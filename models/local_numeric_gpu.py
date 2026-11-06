"""Shared validated local inputs, device cost curves and the existing DP kernel."""
from functools import lru_cache, reduce
from math import gcd
import numpy as np

from models.dp_solver import _prepare_local_parameters, finish_local_solution
from models.dp_gpu import _buffer_bytes, _MAX_DEVICE_BYTES, _INT_MAX, _launch_device_dp


@lru_cache(maxsize=1)
def _kernels():
  from numba import cuda
  from numba.cuda import libdevice as ld
  import math
  globals()['cuda'] = cuda

  @cuda.jit(device=True)
  def alternative(p, restricted, reject, k):
    load, capacity = p[0],p[1]
    alpha,delta,gamma,pi = p[2],p[3],p[4],p[5]
    inbound,ub,omega_fixed,max_x = p[6],p[7],p[8],p[9]
    lower_bound,fixed_x,fixed_r = p[12]>0,p[13]>0,p[14]>0
    r = p[10] if fixed_r else k
    if fixed_x:
      x = p[11]
      r = max(0.,math.ceil(ld.dadd_rn(ld.ddiv_rn(ld.dadd_rn(x,inbound),capacity),-1e-9)))
      lo,hi = x,x
    else:
      lo = max(0.,math.ceil(ld.dadd_rn(ld.dadd_rn(ld.dmul_rn(r-1.,capacity),-inbound),-1e-9))) if lower_bound else 0.
      hi = min(math.floor(ld.dadd_rn(max_x,1e-9)),math.floor(ld.dadd_rn(ld.dadd_rn(ld.dmul_rn(r,capacity),-inbound),1e-9)))
    # Descending {lo, hi, clipped offload breakpoint}; duplicates cannot win a strict tie.
    middle = max(lo,min(hi,ld.dadd_rn(load,-ub)))
    best,bx,bo,bz = float('inf'),0.,0.,0.
    for candidate in range(3):
      x = hi if candidate==0 else middle if candidate==1 and not restricted else lo
      if lo<=x<=hi:
        remaining=ld.dadd_rn(load,-x)
        advantage=ld.dadd_rn(ld.dadd_rn(delta,-pi),gamma)
        omega = omega_fixed if restricted else remaining if not reject else min(remaining,ub) if advantage>0 else 0.
        z=ld.dadd_rn(remaining,-omega)
        if z < -1e-9 or (reject and not restricted and abs(z-round(z))>1e-9):
          continue
        numerator=ld.dadd_rn(ld.dadd_rn(ld.dmul_rn(alpha,x),ld.dmul_rn(ld.dadd_rn(delta,-pi),omega)),-ld.dmul_rn(gamma,z))
        cost=ld.ddiv_rn(-numerator,load if load else 1.)
        if cost<best:
          best,bx,bo,bz=cost,x,omega,z
    return best,bx,bo,bz,r

  @cuda.jit
  def curves(parameters,flags,lengths,nfs,costs):
    index=cuda.grid(1)
    nl,nf=costs.shape[2],costs.shape[1]
    n,f,k=index//(nf*nl),(index//nl)%nf,index%nl
    if n<costs.shape[0]:
      if f<nfs[n] and k<lengths[n,f]:
        costs[n,f,k]=alternative(parameters[n,f],flags[n,0],flags[n,1],k)[0]
      else:
        costs[n,f,k]=float('inf')

  @cuda.jit
  def decode(parameters,flags,nfs,choices,valid,output):
    index=cuda.grid(1)
    n,f=index//output.shape[1],index%output.shape[1]
    if n<output.shape[0] and f<nfs[n] and valid[n]:
      cost,x,omega,z,r=alternative(parameters[n,f],flags[n,0],flags[n,1],choices[n,f])
      output[n,f,0]=cost;output[n,f,1]=x;output[n,f,2]=omega
      output[n,f,3]=z;output[n,f,4]=r
  return cuda,curves,decode


def _shape(parsed):
  problem,rows=parsed
  scale=reduce(gcd,map(int,problem.memory))
  lengths=[1 if row[13] or row[14] else int(row[10])+1 for row in rows]
  return len(rows),max(lengths),int(problem.ram_cap)//scale+1


def _bytes(count,nf,nl,nb):
  # DP arrays plus numeric inputs, flags, compact outputs; include all new buffers.
  return _buffer_bytes(count,nf,nl,nb)+count*(15*nf*8+2+5*nf*8)


def _run_chunk(items):
  cuda,curves,decode=_kernels()
  count=len(items);nf,nl,nb=map(max,zip(*[_shape(p) for p in items]))
  parameters=np.zeros((count,nf,15),dtype=np.float64)
  memory=np.ones((count,nf),dtype=np.int32);lengths=np.zeros_like(memory)
  nfs=np.zeros(count,dtype=np.int32);budgets=np.zeros(count,dtype=np.int32)
  flags=np.zeros((count,2),dtype=np.bool_)
  direct=items[0][0].direct
  for n,(p,rows) in enumerate(items):
    scale=reduce(gcd,map(int,p.memory));fs=len(rows)
    parameters[n,:fs]=rows;memory[n,:fs]=[int(m)//scale for m in p.memory]
    lengths[n,:fs]=[1 if row[13] or row[14] else int(row[10])+1 for row in rows]
    nfs[n],budgets[n],flags[n]=fs,int(p.ram_cap)//scale,(p.restricted,p.reject)
  dp,df,dl,dn=[cuda.to_device(a) for a in (parameters,flags,lengths,nfs)]
  if direct:
    selected=cuda.to_device(np.zeros((count,nf),dtype=np.int32))
    valid=cuda.to_device(np.ones(count,dtype=np.int32))
  else:
    costs=cuda.device_array((count,nf,nl),dtype=np.float64)
    curves[(count*nf*nl+127)//128,128](dp,df,dl,dn,costs)
    selected,valid,scratch=_launch_device_dp(costs,cuda.to_device(memory),dl,
                                           cuda.to_device(budgets),dn,count,nf,nb)
  output=cuda.device_array((count,nf,5),dtype=np.float64)
  decode[(count*nf+127)//128,128](dp,df,dn,selected,valid,output)
  values,feasible=output.copy_to_host(),valid.copy_to_host()
  cuda.synchronize()
  from utils.compute_backend import record_operation
  record_operation('local_numeric','cuda',count,shape=[count,nf,nl,nb],
                   device_bytes=_bytes(count,nf,nl,nb))
  results=[]
  for n,(p,rows) in enumerate(items):
    row=values[n,:len(rows)]
    if (not feasible[n] or not np.all(np.isfinite(row))
        or sum(a[4]*m for a,m in zip(row,p.memory))>p.ram_cap):
      results.append(None)
      continue
    p.costs=[[float(a[0])] for a in row]
    p.levels=[[(float(a[1]),float(a[2]),float(a[3]),int(a[4]))] for a in row]
    results.append(finish_local_solution(p,[0]*len(rows),runtime=0.))
  return results


def try_solve_local_batch_cuda(model,data_batch):
  """Ordered compact solutions; unsupported/infeasible entries are None, errors propagate."""
  from utils import step_time
  step_time.mark_shared_device()
  parsed=[_prepare_local_parameters(model,d) for d in data_batch]
  for i, problem in enumerate(parsed):
    # NumPy scalar arithmetic keeps its precision on CPU. Promoting a float32
    # capacity here can change floor/ceil decisions, even with float64 kernels.
    if problem is not None and any(
        isinstance(value, np.floating) and value.dtype != np.dtype('float64')
        for row in problem[1] for value in row):
      parsed[i] = None
  result=[None]*len(parsed)
  if sum(p is not None for p in parsed)<2:
    from utils.compute_backend import record_operation
    record_operation('local_numeric','cpu',len(parsed),reason='fewer than two eligible problems (unsupported data/dtype excluded)')
    return result
  # A model fixes direct/non-direct; retain input order and bounded padded chunks.
  chunk,indices,shape=[],[],(0,0,0)
  def flush():
    if chunk:
      for i,solution in zip(indices,_run_chunk(chunk)):
        result[i]=solution
  for i,p in enumerate(parsed):
    if p is None:
      continue
    item_shape=_shape(p)
    scale=reduce(gcd,map(int,p[0].memory))
    if (max(item_shape)>_INT_MAX or any(int(m)//scale>_INT_MAX for m in p[0].memory)
        or _bytes(1,*item_shape)>_MAX_DEVICE_BYTES):
      continue
    padded=tuple(map(max,zip(shape,item_shape)))
    if chunk and _bytes(len(chunk)+1,*padded)>_MAX_DEVICE_BYTES:
      flush();chunk,indices,shape=[],[],(0,0,0)
      padded=item_shape
    chunk.append(p);indices.append(i);shape=padded
  flush()
  from utils.compute_backend import record_operation
  unsupported=sum(p is None or max(_shape(p))>_INT_MAX or _bytes(1,*_shape(p))>_MAX_DEVICE_BYTES for p in parsed)
  if unsupported:
    record_operation('local_numeric','cpu',unsupported,reason='unsupported data/dtype or buffer/index limits')
  return result
