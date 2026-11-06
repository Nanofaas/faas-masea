"""Lazy float64 CUDA kernels for shared numerical phases; no device state cache."""
import numpy as np

_residual_kernel = None
_source_sum_device = None


def _residual():
  global _residual_kernel, cuda, _source_sum_device
  if _residual_kernel is None:
    from numba import cuda
    from numba import float64, int64
    from numba.cuda import libdevice

    @cuda.jit(device=True)
    def source_sum(y, n, f):
      # Match NumPy's float64 pairwise sum (PW_BLOCKSIZE=128), including
      # its eight accumulators and balanced splits. No unordered atomics.
      starts = cuda.local.array(32, int64)
      sizes = cuda.local.array(32, int64)
      lefts = cuda.local.array(32, float64)
      states = cuda.local.array(32, int64)
      begin, size, depth = 0, y.shape[0], 0
      while True:
        if size > 128:
          left = size // 2
          left -= left % 8
          starts[depth], sizes[depth], states[depth] = begin + left, size - left, 0
          depth += 1
          size = left
          continue
        if size < 8:
          value = -0.
          for i in range(size):
            value = libdevice.dadd_rn(value, y[begin+i,n,f])
        else:
          a=y[begin,n,f]; b=y[begin+1,n,f]; c=y[begin+2,n,f]; d=y[begin+3,n,f]
          e=y[begin+4,n,f]; g=y[begin+5,n,f]; h=y[begin+6,n,f]; j=y[begin+7,n,f]
          end = size - size % 8
          for i in range(8, end, 8):
            a=libdevice.dadd_rn(a,y[begin+i,n,f]); b=libdevice.dadd_rn(b,y[begin+i+1,n,f])
            c=libdevice.dadd_rn(c,y[begin+i+2,n,f]); d=libdevice.dadd_rn(d,y[begin+i+3,n,f])
            e=libdevice.dadd_rn(e,y[begin+i+4,n,f]); g=libdevice.dadd_rn(g,y[begin+i+5,n,f])
            h=libdevice.dadd_rn(h,y[begin+i+6,n,f]); j=libdevice.dadd_rn(j,y[begin+i+7,n,f])
          value = libdevice.dadd_rn(
            libdevice.dadd_rn(libdevice.dadd_rn(a,b),libdevice.dadd_rn(c,d)),
            libdevice.dadd_rn(libdevice.dadd_rn(e,g),libdevice.dadd_rn(h,j)))
          for i in range(end,size):
            value=libdevice.dadd_rn(value,y[begin+i,n,f])
        go_right = False
        while depth:
          top = depth - 1
          if states[top] == 0:
            lefts[top], states[top] = value, 1
            begin, size = starts[top], sizes[top]
            go_right = True
            break
          value = libdevice.dadd_rn(lefts[top], value)
          depth -= 1
        if not go_right:
          return value

    _source_sum_device = source_sum

    @cuda.jit
    def kernel(x, y, r, demand, utilization, cap, residual, ell):
      k = cuda.grid(1)
      nf = x.shape[1]
      n, f = k // nf, k % nf
      if n < x.shape[0]:
        incoming = source_sum(y,n,f)
        load = libdevice.dadd_rn(x[n,f], incoming)
        capacity = libdevice.dmul_rn(r[n,f], libdevice.ddiv_rn(utilization[f], demand[n,f]))
        cap[n,f] = capacity
        ell[n,f] = load
        residual[n,f] = max(0., libdevice.dadd_rn(capacity, -load))
    _residual_kernel = kernel
  return _residual_kernel


def _source_reducer():
  """Reuse the exact one-dimensional NumPy reduction without another launch."""
  _residual()
  return _source_sum_device


def residual_capacity_cuda(x, y, r, demand, utilization):
  """Synchronous full phase, with source reduction in the CPU loop order."""
  from utils import step_time
  step_time.mark_shared_device()
  kernel = _residual()
  arrays = [np.array(a, dtype=np.float64, order='C', copy=True) for a in (x,y,r,demand,utilization)]
  device = [cuda.to_device(a) for a in arrays]
  outputs = [cuda.device_array(arrays[0].shape, dtype=np.float64) for _ in range(3)]
  size = arrays[0].size
  if size:
    kernel[(size+127)//128,128](*device,*outputs)
  result = tuple(a.copy_to_host() for a in outputs)
  cuda.synchronize()
  from utils.compute_backend import record_operation
  record_operation('residual_capacity','cuda',arrays[0].shape[0],shape=list(arrays[0].shape),
                   device_bytes=sum(a.nbytes for a in arrays)+3*arrays[0].nbytes)
  return result


def _group_kernels():
  global _groups_cached,cuda
  if '_groups_cached' in globals():return _groups_cached
  from numba import cuda,float64,int64
  from numba.cuda import libdevice as ld

  @cuda.jit(device=True)
  def pairwise(values,members,begin,size):
    starts=cuda.local.array(32,int64);sizes=cuda.local.array(32,int64)
    lefts=cuda.local.array(32,float64);states=cuda.local.array(32,int64)
    depth=0
    while True:
      if size>128:
        left=size//2;left-=left%8
        starts[depth],sizes[depth],states[depth]=begin+left,size-left,0
        depth+=1;size=left
        continue
      if size<8:
        value=-0.
        for i in range(size):value=ld.dadd_rn(value,values[members[begin+i],0])
      else:
        a=values[members[begin],0];b=values[members[begin+1],0]
        c=values[members[begin+2],0];d=values[members[begin+3],0]
        e=values[members[begin+4],0];g=values[members[begin+5],0]
        h=values[members[begin+6],0];j=values[members[begin+7],0]
        end=size-size%8
        for i in range(8,end,8):
          a=ld.dadd_rn(a,values[members[begin+i],0]);b=ld.dadd_rn(b,values[members[begin+i+1],0])
          c=ld.dadd_rn(c,values[members[begin+i+2],0]);d=ld.dadd_rn(d,values[members[begin+i+3],0])
          e=ld.dadd_rn(e,values[members[begin+i+4],0]);g=ld.dadd_rn(g,values[members[begin+i+5],0])
          h=ld.dadd_rn(h,values[members[begin+i+6],0]);j=ld.dadd_rn(j,values[members[begin+i+7],0])
        value=ld.dadd_rn(ld.dadd_rn(ld.dadd_rn(a,b),ld.dadd_rn(c,d)),ld.dadd_rn(ld.dadd_rn(e,g),ld.dadd_rn(h,j)))
        for i in range(end,size):value=ld.dadd_rn(value,values[members[begin+i],0])
      right=False
      while depth:
        top=depth-1
        if states[top]==0:
          lefts[top],states[top]=value,1;begin,size=starts[top],sizes[top];right=True
          break
        value=ld.dadd_rn(lefts[top],value);depth-=1
      if not right:return value

  @cuda.jit
  def groups(values,members,offsets,out):
    k=cuda.grid(1);nf=values.shape[1];group,f=k//nf,k%nf
    if group<out.shape[0]:
      if nf==1:
        out[group,f]=pairwise(values,members,offsets[group],offsets[group+1]-offsets[group])
      else:
        # NumPy's non-fast axis reduction walks rows in order, independently per f.
        value=0.
        for i in range(offsets[group],offsets[group+1]):value=ld.dadd_rn(value,values[members[i],f])
        out[group,f]=value

  @cuda.jit
  def integer_groups(values,members,offsets,out):
    k=cuda.grid(1);nf=values.shape[1];group,f=k//nf,k%nf
    if group<out.shape[0]:
      value=int64(0)
      for i in range(offsets[group],offsets[group+1]):value+=values[members[i],f]
      out[group,f]=value

  @cuda.jit
  def price(lam,alpha,gradient,out):
    k=cuda.grid(1)
    if k<out.size:
      value=ld.dadd_rn(lam[k],ld.dmul_rn(alpha,gradient[k]))
      out[k]=value if value!=value else max(0.,value)
  _groups_cached=cuda,groups,price,integer_groups
  return _groups_cached


def aggregate_groups_cuda(values,member_ids,offsets):
  from utils import step_time
  from utils.compute_backend import record_operation
  dtype=np.int64 if np.issubdtype(values.dtype,np.signedinteger) else np.float64
  shape=(len(offsets)-1,values.shape[1])
  if not shape[0]:
    record_operation('aggregate_groups','cpu',0,shape=list(shape),reason='empty groups')
    return np.empty(shape,dtype=dtype)
  step_time.mark_shared_device()
  values=np.array(values,dtype=dtype,order='C',copy=True)
  members=np.array(member_ids,dtype=np.int64,order='C',copy=True)
  offsets=np.array(offsets,dtype=np.int64,order='C',copy=True)
  cuda,kernel,_,integer_kernel=_group_kernels()
  if dtype==np.int64:kernel=integer_kernel
  inputs=[cuda.to_device(a) for a in (values,members,offsets)]
  out=cuda.device_array(shape,dtype=dtype)
  kernel[(out.size+127)//128,128](*inputs,out)
  result=out.copy_to_host();cuda.synchronize()
  record_operation('aggregate_groups','cuda',shape[0],shape=[*shape,len(members)],
                   device_bytes=values.nbytes+members.nbytes+offsets.nbytes+result.nbytes)
  return result


def dual_price_cuda(lam,alpha,subgradient):
  from utils import step_time
  from utils.compute_backend import record_operation
  step_time.mark_shared_device()
  cuda,_,kernel,_=_group_kernels()
  lam=np.array(lam,dtype=np.float64,order='C',copy=True)
  gradient=np.array(subgradient,dtype=np.float64,order='C',copy=True)
  out=cuda.device_array(lam.size,dtype=np.float64)
  kernel[(out.size+127)//128,128](cuda.to_device(lam.ravel()),alpha,cuda.to_device(gradient.ravel()),out)
  result=out.copy_to_host().reshape(lam.shape);cuda.synchronize()
  record_operation('dual_price','cuda',len(lam),shape=list(lam.shape),device_bytes=3*lam.nbytes)
  return result


def _confirmation():
  global _confirmation_cached,cuda
  if '_confirmation_cached' in globals():return _confirmation_cached
  from numba import cuda
  @cuda.jit
  def kernel(ids,functions,omega,tolerance,mask,demand):
    k=cuda.grid(1)
    if k<ids.size:
      value=omega[ids[k],functions[k]]
      mask[k]=value>tolerance;demand[k]=value
  _confirmation_cached=kernel
  return kernel


def confirm_memory_rows_cuda(ids,functions,omega,tolerance):
  from utils import step_time
  from utils.compute_backend import record_operation
  step_time.mark_shared_device()
  kernel=_confirmation()
  arrays=[np.array(a,dtype=dtype,order='C',copy=True) for a,dtype in ((ids,np.int64),(functions,np.int64),(omega,np.float64))]
  inputs=[cuda.to_device(a) for a in arrays]
  mask=cuda.device_array(len(ids),dtype=np.bool_);demand=cuda.device_array(len(ids),dtype=np.float64)
  kernel[(len(ids)+127)//128,128](*inputs,tolerance,mask,demand)
  result=mask.copy_to_host(),demand.copy_to_host();cuda.synchronize()
  with step_time.untimed():
    record_operation('memory_confirmation','cuda',len(np.unique(ids)),shape=[len(ids),*omega.shape],
                     device_bytes=sum(a.nbytes for a in arrays)+9*len(ids))
  return result
