"""Seller-local replica policies, sequential within each independent seller."""
from functools import lru_cache
import numpy as np
from utils import step_time
from utils.compute_backend import record_operation


@lru_cache(maxsize=1)
def _kernels():
  from numba import cuda
  from numba.cuda import libdevice as ld
  import math
  globals()['cuda']=cuda

  @cuda.jit(device=True)
  def inverse(capacity,target):
    count=max(0,int(math.ceil(ld.ddiv_rn(target,capacity))))
    while count and ld.dmul_rn(count-1,capacity)>=target:count-=1
    while ld.dmul_rn(count,capacity)<target:count+=1
    return count

  @cuda.jit(device=True)
  def productive(replicas,capacity):
    served=int(math.floor(ld.dmul_rn(replicas,capacity)))
    count=max(1,inverse(capacity,served+1)-int(replicas))
    gain=int(math.floor(ld.dmul_rn(ld.dadd_rn(replicas,count),capacity)))-served
    return count,gain

  @cuda.jit(device=True)
  def floor_divide(a,b):
    # NumPy/Python positive float divmod corrects division near integer bounds.
    remainder=ld.fmod(a,b)
    div=ld.ddiv_rn(ld.dadd_rn(a,-remainder),b)
    result=math.floor(div)
    if ld.dadd_rn(div,-result)>.5:result+=1
    return int(result)

  @cuda.jit
  def proportional(r,rho,memory,capacity,fractions,additional,residual,allocated):
    j=cuda.grid(1)
    if j>=r.shape[0] or rho[j]<=0:return
    seller=rho[j]
    for f in range(r.shape[1]):
      frac=fractions[j,f]
      if frac<=0:continue
      ram=memory[f];cap=capacity[j,f]
      count=floor_divide(ld.dmul_rn(seller,frac),ram)
      before=int(math.floor(ld.dmul_rn(r[j,f],cap)))
      after=int(math.floor(ld.dmul_rn(ld.dadd_rn(r[j,f],count),cap)))
      needed=max(0,inverse(cap,after)-int(r[j,f]))
      count=min(count,needed) if after>before else 0
      spent=ld.dmul_rn(ram,count)
      residual[j]=ld.dadd_rn(residual[j],-int(spent))
      additional[j,f]=count;allocated[j,f]=spent
    while True:
      best=-1;best_deficit=-float('inf');best_fraction=-1.;best_count=0;best_memory=0.
      for f in range(r.shape[1]):
        frac=fractions[j,f]
        if frac<=0:continue
        deficit=ld.dadd_rn(ld.dmul_rn(seller,frac),-allocated[j,f])
        count,_=productive(ld.dadd_rn(r[j,f],additional[j,f]),capacity[j,f])
        mem=ld.dmul_rn(count,memory[f])
        # max(deficit, fraction, -f, f, count, memory), unique f per candidate.
        if mem<=residual[j] and (best<0 or deficit>best_deficit or
            (deficit==best_deficit and (frac>best_fraction or (frac==best_fraction and f<best)))):
          best,best_deficit,best_fraction,best_count,best_memory=f,deficit,frac,count,mem
      if best<0:break
      additional[j,best]=ld.dadd_rn(additional[j,best],best_count)
      allocated[j,best]=ld.dadd_rn(allocated[j,best],best_memory)
      residual[j]=ld.dadd_rn(residual[j],-best_memory)

  @cuda.jit
  def by_value(r,rho,memory,capacity,offsets,values,demands,heads,additional,residual):
    j=cuda.grid(1)
    if j>=r.shape[0] or rho[j]<=0:return
    free=rho[j];nf=r.shape[1]
    while True:
      best=-1;best_ratio=0.;best_count=0;best_capacity=0.;best_memory=0.
      for f in range(nf):
        index=j*nf+f
        if heads[index]>=offsets[index+1]:continue
        count,gain=productive(ld.dadd_rn(r[j,f],additional[j,f]),capacity[j,f])
        mem=ld.dmul_rn(count,memory[f])
        if mem>free:continue
        value=0.;left=float(gain)
        for q in range(heads[index],offsets[index+1]):
          take=min(demands[q],left)
          value=ld.dadd_rn(value,ld.dmul_rn(values[q],take))
          left=ld.dadd_rn(left,-take)
          if left<=0:break
        ratio=ld.ddiv_rn(value,mem)
        if value>0 and (best<0 or ratio>best_ratio):
          best,best_ratio,best_count,best_capacity,best_memory=f,ratio,count,gain,mem
      if best<0:break
      additional[j,best]=ld.dadd_rn(additional[j,best],best_count)
      free=ld.dadd_rn(free,-best_memory)
      residual[j]=ld.dadd_rn(residual[j],-best_memory)
      left=float(best_capacity);index=j*nf+best
      while heads[index]<offsets[index+1] and left>0:
        q=heads[index];take=min(demands[q],left)
        demands[q]=ld.dadd_rn(demands[q],-take);left=ld.dadd_rn(left,-take)
        if demands[q]<=1e-9:heads[index]+=1
  return cuda,proportional,by_value


def _prepare(memory_bids,r,data,rho,value_policy):
  nn,nf=r.shape;v=data[None]
  capacity=np.ones((nn,nf));memory=np.ones(nf);fractions=np.zeros((nn,nf));queues={}
  for j,bids in memory_bids.groupby('j'):
    if rho[j]<=0:continue
    if value_policy:
      groups=[]
      for f,group in bids.groupby('f'):
        requests=sorted(((float(val),float(d)) for val,d in zip(group['v'],group['d']) if d>0 and val>0),reverse=True)
        if requests:groups.append((f,requests));queues[int(j)*nf+int(f)]=requests
    else:
      groups=list(bids['f'].value_counts(normalize=True).sort_index().items())
    for f,quantity in groups:
      memory[f]=v['memory_requirement'][f+1]
      capacity[j,f]=v['max_utilization'][f+1]/v['demand'][j+1,f+1]
      fractions[j,f]=1. if value_policy else quantity
  offsets=[0];values=[];demands=[]
  for index in range(nn*nf):
    for value,demand in queues.get(index,[]):values.append(value);demands.append(demand)
    offsets.append(len(values))
  return capacity,memory,fractions,np.asarray(offsets,dtype=np.int64),np.asarray(values),np.asarray(demands)


def _solve(memory_bids,r,data,rho,value_policy):
  if value_policy:
    raise ValueError("Value-based replica allocation is not part of this artifact")
  from run_faasmadea import VAR_TYPE,start_additional_replicas
  if VAR_TYPE is not int:
    fn=start_additional_replicas
    return fn.__wrapped__(memory_bids,r,data,rho)
  if len(memory_bids)==0:
    record_operation('replicas_value' if value_policy else 'replicas_proportional','cpu',0,reason='empty bids')
    return np.zeros(r.shape),rho.copy()
  step_time.mark_shared_device()
  capacity,memory,fractions,offsets,values,demands=_prepare(memory_bids,r,data,rho,value_policy)
  cuda,proportional,by_value=_kernels();nn,nf=r.shape
  arrays=[np.array(a,dtype=np.float64,order='C',copy=True) for a in (r,rho,memory,capacity)]
  inputs=[cuda.to_device(a) for a in arrays]
  additional=cuda.to_device(np.zeros(r.shape));residual=cuda.to_device(np.array(rho,dtype=np.float64,copy=True))
  if value_policy:
    buffers=[offsets,values,demands,offsets[:-1].copy()]
    by_value[(nn+127)//128,128](*inputs,*[cuda.to_device(a) for a in buffers],additional,residual)
  else:
    buffers=[fractions,np.zeros(r.shape)]
    proportional[(nn+127)//128,128](*inputs,cuda.to_device(fractions),additional,residual,cuda.to_device(buffers[1]))
  result=additional.copy_to_host(),residual.copy_to_host();cuda.synchronize()
  record_operation('replicas_value' if value_policy else 'replicas_proportional','cuda',int(np.count_nonzero(fractions.any(axis=1))),
                   shape=[nn,nf,len(values)],device_bytes=sum(a.nbytes for a in arrays+buffers)+nn*nf*8+nn*8)
  return result


def start_additional_replicas_cuda(memory_bids,r,data,rho):
  return _solve(memory_bids,r,data,rho,False)


def start_replicas_by_value_cuda(memory_bids,r,data,rho):
  return _solve(memory_bids,r,data,rho,True)


def replica_batch_eligible(memory_bids,r,rho,*,value_policy):
  from run_faasmadea import VAR_TYPE
  if VAR_TYPE is not int:return False,'continuous request domain',0
  if len(memory_bids)==0:return False,'empty bids',0
  if any(not isinstance(memory_bids[c].dtype,np.dtype) or memory_bids[c].dtype.kind not in 'iu'
         for c in ('i','j','f')):
    return False,'outside native integer grouping domain',0
  sellers=np.unique(memory_bids['j'].to_numpy())
  count=int(np.count_nonzero(rho[sellers]>0))
  minimum=30 if value_policy else 100
  rows=20 if value_policy else 5
  bound=5*r.size*8+2*rho.size*8+8*r.shape[1]+16*len(memory_bids)+16*(r.size+1)
  if (count<minimum or r.shape[1]<5 or len(memory_bids)<rows*count):
    return False,f'performance gate: >={minimum} positive-budget sellers, >=5 functions, >={rows} rows/seller',count
  if r.dtype!=np.float64 or rho.dtype!=np.float64 or bound>64*1024*1024:
    return False,'outside float64 or 64 MiB buffer domain',count
  return True,'',count
