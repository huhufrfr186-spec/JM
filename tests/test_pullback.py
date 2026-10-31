import numpy as np
import torch
from jm.render import Regional
from jm.integrator import Settings,direct

def test_original_projected_parameter_pullback():
    mu=torch.tensor([[.03,-.02],[.03,-.02]],dtype=torch.float64,requires_grad=True)
    precision=torch.tensor([[[1.3,.1],[.1,1.2]]]*2,dtype=torch.float64,requires_grad=True)
    opacity=torch.tensor([.3,.4],dtype=torch.float64,requires_grad=True)
    colors=torch.tensor([[.2,.5,.7],[.8,.2,.3]],dtype=torch.float64,requires_grad=True)
    bg=torch.tensor([.1,.2,.1],dtype=torch.float64,requires_grad=True)
    rect=np.array([-.5,-.5,.5,.5]); args=[mu,precision,opacity,colors,bg]
    Regional.apply(*args,rect,Settings()).sum().backward()
    original=[x.detach().numpy().copy() for x in args]
    for k,param in enumerate(args):
        for index in np.ndindex(original[k].shape):
            plus=[x.copy() for x in original]; minus=[x.copy() for x in original]
            eps=1e-5; plus[k][index]+=eps; minus[k][index]-=eps
            fd=(direct(*plus,rect,64)[0].sum()-direct(*minus,rect,64)[0].sum())/(2*eps)
            np.testing.assert_allclose(param.grad[index].item(),fd,atol=2e-8)
