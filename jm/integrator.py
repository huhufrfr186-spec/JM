"""Full-degree shared moments and original-parameter response contraction."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from .polynomial import *

@dataclass
class Settings:
    color_tol: float = 1e-4
    response_tol: float = 1e-3
    quadrature_tol: float = 1e-8
    order: int = 8
    max_order: int = 64
    max_depth: int = 4
    direct_layers: int = 24

def nodes(rect, order):
    z,w=leggauss(order)
    xx,yy=np.meshgrid(z,z,indexing='ij')
    uv=np.stack([xx.ravel(),yy.ravel()],1)
    weights=np.outer(w,w).ravel()/4
    center=(rect[:2]+rect[2:])/2; half=(rect[2:]-rect[:2])/2
    return center+uv*half,uv,weights,center,half

def direct(mu, precision, opacity, colors, background, rect, order=32):
    u,uv,w,m,h=nodes(np.asarray(rect),order)
    d=u[None]-mu[:,None]
    alpha=opacity[:,None]*np.exp(-.5*np.einsum('nqi,nij,nqj->nq',d,precision,d))
    n=len(mu); c=np.broadcast_to(background,(len(u),len(background))).copy()
    behind=[]
    for i in range(n-1,-1,-1):
        behind.append(c.copy()); c=alpha[i,:,None]*colors[i]+(1-alpha[i,:,None])*c
    behind=behind[::-1]; trans=np.ones(len(u))
    response=np.zeros((n,6,len(background))); appearance=np.zeros(n)
    basis=np.array([uv[:,0]**x*uv[:,1]**y for x,y in BASIS])
    for i in range(n):
        s=(alpha[i]*trans)[:,None]*(colors[i]-behind[i])
        response[i]=np.einsum('q,bq,qc->bc',w,basis,s)
        appearance[i]=w@(trans*alpha[i]); trans*=1-alpha[i]
    return w@c,response,appearance,float(w@trans)

def build(mu, precision, opacity, colors, background, rect):
    m=(rect[:2]+rect[2:])/2; h=(rect[2:]-rect[:2])/2
    w=opacity/opacity.sum(); mean=w@mu; prec=np.einsum('n,nij->ij',w,precision)
    logg=coefficients(mean,prec,0.,m,h)
    _,gmax=quadratic_range(logg)
    logs=np.array([coefficients(a,b,np.log(o),m,h) for a,b,o in zip(mu,precision,opacity)])
    loga=np.minimum(logs[:,0]-logg[0],-gmax)
    residual=logs-logg; residual[:,0]-=loga
    delta=sum(max(abs(a),abs(b)) for a,b in map(quadratic_range,residual))
    B=constant(background); R=constant(np.zeros_like(background)); behind=[]
    for i in range(len(mu)-1,-1,-1):
        behind.append((B,R)); ag={(1,0,0):np.exp(loga[i])}; r=spatial(residual[i])
        gap=add(constant(colors[i]),scale(B,-1))
        R=add(R,mul(ag,add(mul(r,gap),scale(R,-1))))
        B=add(B,mul(ag,gap))
    T=constant(1.); Tr=constant(0.); responses=[]; appearance=[]
    for i,(back,br) in enumerate(behind[::-1]):
        ag={(1,0,0):np.exp(loga[i])}; r=spatial(residual[i]); gap=add(constant(colors[i]),scale(back,-1))
        # W + D_r W, transposed once for all six original log directions.
        S=mul(ag,add(mul(add(T,Tr),gap),mul(T,add(mul(r,gap),scale(br,-1)))))
        responses.append([shift(S,x,y) for x,y in BASIS])
        appearance.append(mul(ag,add(T,Tr,mul(T,r))))
        Tr=add(Tr,scale(mul(ag,add(Tr,mul(T,r))),-1))
        T=add(T,scale(mul(ag,T),-1))
    return add(B,R),responses,appearance,add(T,Tr),logg,delta

def shared(local, order):
    color,response,appearance,terminal,logg,_=local
    _,uv,w,_,_=nodes(np.array([-1.,-1.,1.,1.]),order)
    basis=np.array([uv[:,0]**x*uv[:,1]**y for x,y in BASIS])
    g=np.exp(logg@basis)
    polynomials=[color,terminal,*appearance,*[p for row in response for p in row]]
    keys=set().union(*(p.keys() for p in polynomials))
    # One numerical moment for each distinct power and spatial monomial.
    moments={key:w@(g**key[0]*uv[:,0]**key[1]*uv[:,1]**key[2]) for key in keys}
    return (contract(color,moments),np.array([[contract(p,moments) for p in row] for row in response]),
            np.array([contract(p,moments) for p in appearance]),float(contract(terminal,moments)))

def integrate(mu, precision, opacity, colors, background, rect, settings=None, stats=None, _depth=0):
    s=settings or Settings(); rect=np.asarray(rect,dtype=float)
    mu,precision,opacity,colors,background=map(lambda a:np.asarray(a,dtype=float),(mu,precision,opacity,colors,background))
    if len(mu)==0: return background.copy(),np.zeros((0,6,len(background))),np.zeros(0),1.
    if np.any(opacity<=0) or np.any(opacity>1): raise ValueError('Opacity must be in (0,1].')
    if np.any(np.linalg.eigvalsh(precision)<=0): raise ValueError('Precision must be positive definite.')
    route=len(mu)>=s.direct_layers
    local=None if route else build(mu,precision,opacity,colors,background,rect)
    delta=np.inf if route else local[-1]
    allowed=delta*delta<=s.color_tol and 2*delta*delta<=s.response_tol
    if not allowed and _depth<s.max_depth and not route:
        mid=(rect[:2]+rect[2:])/2
        children=[np.array([x0,y0,x1,y1]) for x0,x1 in [(rect[0],mid[0]),(mid[0],rect[2])] for y0,y1 in [(rect[1],mid[1]),(mid[1],rect[3])]]
        values=[]
        for child in children:
            v=list(integrate(mu,precision,opacity,colors,background,child,s,stats,_depth+1))
            # Convert child log-direction basis moments to the parent basis.
            cm=(child[:2]+child[2:])/2; ch=(child[2:]-child[:2])/2
            m=(rect[:2]+rect[2:])/2; h=(rect[2:]-rect[:2])/2
            a=(cm-m)/h; b=ch/h; j=v[1].copy()
            v[1][:,0]=j[:,0]
            v[1][:,1]=a[0]*j[:,0]+b[0]*j[:,1]
            v[1][:,2]=a[1]*j[:,0]+b[1]*j[:,2]
            v[1][:,3]=a[0]**2*j[:,0]+2*a[0]*b[0]*j[:,1]+b[0]**2*j[:,3]
            v[1][:,4]=a[0]*a[1]*j[:,0]+b[0]*a[1]*j[:,1]+a[0]*b[1]*j[:,2]+b[0]*b[1]*j[:,4]
            v[1][:,5]=a[1]**2*j[:,0]+2*a[1]*b[1]*j[:,2]+b[1]**2*j[:,5]
            values.append(v)
        return tuple(sum(v[k] for v in values)/4 for k in range(4))
    evaluator=(lambda q: shared(local,q)) if allowed and not route else (lambda q: direct(mu,precision,opacity,colors,background,rect,q))
    key='jm' if allowed and not route else 'direct'
    if stats is not None: stats[key]=stats.get(key,0)+1
    q=s.order; old=evaluator(q)
    while q*2<=s.max_order:
        q*=2; new=evaluator(q)
        error=max(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0) for a,b in zip(new,old))
        if error<s.quadrature_tol: return new
        old=new
    raise RuntimeError('Quadrature tolerance unresolved; increase max_order.')
