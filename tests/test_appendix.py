"""Independent Appendix D checks, evaluated without the shared integrator."""
import numpy as np
from numpy.polynomial.legendre import leggauss

def test_finite_gaussian_visibility():
    z,w=leggauss(96); x,y=np.meshgrid(z,z,indexing='ij'); weights=np.outer(w,w)/4
    left=.65*np.exp(-.5*((x+.55)**2/.22**2+y*y/.65**2))
    right=np.exp(-.5*((x-.30)**2/.38**2+y*y/.75**2))
    opacity=np.sum(weights*x*(1-left))/np.sum(weights*x*(1-left)*right)
    T=(1-left)*(1-opacity*right); beta=.5*np.exp(-.5*(x*x+y*y))
    derivative=np.sum(weights*T*x*beta)
    np.testing.assert_allclose(opacity,.669333928262,atol=1e-12)
    np.testing.assert_allclose(derivative,-1.375710635329e-4,atol=1e-14)

def test_joint_residual_second_order():
    z,w=leggauss(96); x,y=np.meshgrid(z,z,indexing='ij'); weights=np.outer(w,w)/4
    basis=np.array([np.ones_like(x),x,y,x*x,x*y,y*y])
    r=np.array([[.01,.02,-.01,.03,.01,-.02],[-.02,.01,.03,-.01,.02,.01],[.01,-.01,.02,.01,-.03,.02],[.02,.01,-.02,.02,.01,-.01]])
    p=np.array([[.1,.2,-.1,.1,.2,-.1],[-.1,.1,.2,.2,-.1,.1],[.2,-.2,.1,-.1,.1,.2],[.1,.1,-.2,.2,.1,-.1]])
    rr=np.einsum('nb,bxy->nxy',r,basis); pp=np.einsum('nb,bxy->nxy',p,basis)
    g=np.exp(-2*((x-.17)**2+1.3*(y+.11)**2)); alpha=np.array([.22,.31,.18,.27])[:,None,None]*g
    colors=[.2,.8,.4,.6]; errors=[]
    for scale in [.5,1.]:
        B=np.full_like(x,.13); R=np.zeros_like(x); G=R.copy(); U=R.copy(); C=B.copy(); D=R.copy()
        for i in range(3,-1,-1):
            a=alpha[i]; r=scale*rr[i]; p=pp[i]; gap=colors[i]-B
            B,R,G,U=(a*colors[i]+(1-a)*B,a*r*gap+(1-a)*R,a*p*gap+(1-a)*G,a*r*p*gap-a*p*R-a*r*G+(1-a)*U)
            original=a*np.exp(r); D=original*p*(colors[i]-C)+(1-original)*D; C=original*colors[i]+(1-original)*C
        errors.append((abs(np.sum(weights*(B+R-C))),abs(np.sum(weights*(G+U-D)))))
    np.testing.assert_allclose(errors[1],[3.129459e-5,4.749888e-7],rtol=1e-6)
    assert np.all(np.array(errors[1])/errors[0]>3.8)
