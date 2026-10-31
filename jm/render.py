"""Camera projection and PyTorch pullback of independent JM responses."""
import numpy as np
import torch
from .integrator import integrate, Settings
from .enclosure import enclose, squared_loss_interval

class Regional(torch.autograd.Function):
    @staticmethod
    def forward(ctx,mu,precision,opacity,colors,bg,rect,settings):
        arrays=[a.detach().cpu().double().numpy() for a in (mu,precision,opacity,colors,bg)]
        value,j,app,terminal=integrate(*arrays,rect,settings)
        ctx.arrays=arrays; ctx.j=j; ctx.app=app; ctx.terminal=terminal
        ctx.rect=np.asarray(rect); ctx.device=mu.device; ctx.dtype=mu.dtype
        return mu.new_tensor(value)

    @staticmethod
    def backward(ctx,grad):
        mu,prec,opacity,colors,bg=ctx.arrays
        cov=np.einsum('nbc,c->nb',ctx.j,grad.detach().cpu().double().numpy())
        m=(ctx.rect[:2]+ctx.rect[2:])/2; h=(ctx.rect[2:]-ctx.rect[:2])/2
        d=m-mu
        # p_mu = (u-mu)^T Lambda dmu
        gmu=np.einsum('nij,nj->ni',prec,d*cov[:,0,None]+cov[:,1:3]*h)
        # p_Lambda = -1/2 (u-mu)^T dLambda (u-mu)
        second=np.empty((len(mu),2,2))
        second[:,0,0]=d[:,0]**2*cov[:,0]+2*d[:,0]*h[0]*cov[:,1]+h[0]**2*cov[:,3]
        second[:,1,1]=d[:,1]**2*cov[:,0]+2*d[:,1]*h[1]*cov[:,2]+h[1]**2*cov[:,5]
        second[:,0,1]=second[:,1,0]=d[:,0]*d[:,1]*cov[:,0]+d[:,1]*h[0]*cov[:,1]+d[:,0]*h[1]*cov[:,2]+h[0]*h[1]*cov[:,4]
        gc=ctx.app[:,None]*grad.detach().cpu().double().numpy()
        gb=ctx.terminal*grad.detach().cpu().double().numpy()
        convert=lambda a:torch.as_tensor(a,device=ctx.device,dtype=ctx.dtype)
        return convert(gmu),convert(-.5*second),convert(cov[:,0]/opacity),convert(gc),convert(gb),None,None

def project(scene,camera):
    xyz=scene['xyz']; R=camera['w2c'][:3,:3]; t=camera['w2c'][:3,3]; K=camera['K']
    p=xyz@R.T+t; z=p[:,2]; safe=z.clamp_min(1e-6)
    mu=torch.stack((K[0,0]*p[:,0]/safe+K[0,2],K[1,1]*p[:,1]/safe+K[1,2]),1)
    J=torch.zeros((len(p),2,3),device=p.device,dtype=p.dtype)
    J[:,0,0]=K[0,0]/safe; J[:,1,1]=K[1,1]/safe
    J[:,0,2]=-K[0,0]*p[:,0]/safe**2; J[:,1,2]=-K[1,1]*p[:,1]/safe**2
    L=torch.tril(scene['shape'],diagonal=-1)+torch.diag_embed(torch.exp(torch.diagonal(scene['shape'],dim1=-2,dim2=-1)))
    cov=L@L.transpose(-1,-2); cc=R@cov@R.T
    cov2=J@cc@J.transpose(-1,-2)+torch.eye(2,device=p.device,dtype=p.dtype)*1e-8
    order=torch.argsort(z); order=order[z[order]>camera.get('near',1e-3)]
    return mu[order],torch.linalg.inv(cov2[order]),torch.sigmoid(scene['opacity'][order]),torch.sigmoid(scene['color'][order]),z[order],order

def render(scene,camera,settings=None,background=None,depth=False):
    mu,prec,opacity,colors,z,_=project(scene,camera)
    bg=mu.new_zeros(3) if background is None else background
    if depth:
        depth_scale=z.detach().max().clamp_min(1.) if len(z) else mu.new_tensor(1.)
        colors=torch.cat((colors,z[:,None]/depth_scale,torch.ones_like(z[:,None])),1)
        bg=torch.cat((bg,bg.new_zeros(2)))
    pixels=[]
    for y in range(camera['height']):
        for x in range(camera['width']):
            pixels.append(Regional.apply(mu,prec,opacity,colors,bg,np.array([x-.5,y-.5,x+.5,y+.5]),settings or Settings()))
    result=torch.stack(pixels).reshape(camera['height'],camera['width'],-1)
    if depth:
        result=torch.cat((result[...,:3],result[...,3:4]*depth_scale,result[...,4:5]),-1)
    return result

def objective_interval(scene,cameras,images,tolerance,max_cells,regularizer=0.):
    lo=hi=float(regularizer)
    for cam,image in zip(cameras,images):
        mu,prec,opacity,color,_,_=project(scene,cam)
        arrays=[a.detach().cpu().numpy() for a in (mu,prec,opacity,color)]
        ltotal=htotal=0.
        for y in range(cam['height']):
            for x in range(cam['width']):
                l,h=enclose(*arrays,np.zeros(3),[x-.5,y-.5,x+.5,y+.5],tolerance,max_cells)
                a,b=squared_loss_interval(l,h,image[y,x].detach().cpu().numpy())
                ltotal+=a; htotal+=b
        norm=len(cameras)*cam['height']*cam['width']; lo+=ltotal/norm; hi+=htotal/norm
    return lo,hi
