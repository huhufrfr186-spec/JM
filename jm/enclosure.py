"""Spatial interval enclosures for original composite objective comparisons."""
import heapq
import numpy as np
from .polynomial import coefficients, quadratic_range

def cell(mu,prec,opacity,colors,bg,rect):
    m=(rect[:2]+rect[2:])/2; h=(rect[2:]-rect[:2])/2
    lo=np.asarray(bg).copy(); hi=lo.copy()
    for i in range(len(mu)-1,-1,-1):
        a,b=quadratic_range(coefficients(mu[i],prec[i],np.log(opacity[i]),m,h))
        amin,amax=np.exp(a),np.exp(b)
        # B enters monotonically; alpha extrema may depend on channel.
        lo=np.minimum(amin*colors[i]+(1-amin)*lo,amax*colors[i]+(1-amax)*lo)
        hi=np.maximum(amin*colors[i]+(1-amin)*hi,amax*colors[i]+(1-amax)*hi)
    # Floating point padding; spatial ranges, not quadrature differences.
    pad=64*np.finfo(float).eps*(len(mu)+1)*(1+np.maximum(abs(lo),abs(hi)))
    return lo-pad,hi+pad

def enclose(mu,prec,opacity,colors,bg,rect,tolerance=1e-4,max_cells=4096):
    rect=np.asarray(rect,float); area=np.prod(rect[2:]-rect[:2]); serial=0
    def entry(r):
        nonlocal serial
        l,h=cell(mu,prec,opacity,colors,bg,r); weight=np.prod(r[2:]-r[:2])/area
        serial+=1
        return (-float(np.max(h-l)*weight),serial,r,weight*l,weight*h)
    first=entry(rect); heap=[first]; low=first[3].copy(); high=first[4].copy()
    while np.max(high-low)>2*tolerance and len(heap)<max_cells:
        _,_,r,l,h=heapq.heappop(heap); low-=l; high-=h
        axis=int(np.argmax(r[2:]-r[:2])); mid=(r[axis]+r[axis+2])/2
        a=r.copy(); b=r.copy(); a[axis+2]=mid; b[axis]=mid
        for child in (a,b):
            item=entry(child); heapq.heappush(heap,item); low+=item[3]; high+=item[4]
    return low,high

def squared_loss_interval(low,high,target):
    minimum=np.maximum(np.maximum(low-target,target-high),0.)
    maximum=np.maximum(abs(low-target),abs(high-target))
    return .5*np.sum(minimum**2),.5*np.sum(maximum**2)
