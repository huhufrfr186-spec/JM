import numpy as np
from jm.integrator import *
from jm.enclosure import enclose

def state():
    return (np.array([[.05,-.03],[-.02,.04]]),np.array([[[1.2,.1],[.1,1.4]],[[1.3,0],[0,1.2]]]),np.array([.3,.4]),np.array([[.2,.7,.5],[.8,.3,.6]]),np.array([.1,.1,.1]),np.array([-.5,-.5,.5,.5]))

def test_shared_against_original():
    args=state(); s=Settings(color_tol=1.,response_tol=1.)
    v,j,_,_=integrate(*args,s)
    d,dj,_,_=direct(*args,64)
    delta=build(*args)[-1]
    assert np.max(abs(v-d))<delta**2
    assert np.max(abs(j-dj))<2*delta**2

def test_exact_profile():
    args=list(state()); args[0][:]=args[0][0]; args[1][:]=args[1][0]
    v=integrate(*args); d=direct(*args,64)
    for a,b in zip(v,d): np.testing.assert_allclose(a,b,atol=2e-11)

def test_direct_finite_difference():
    args=list(state()); base=direct(*args,64); eps=1e-5
    plus=[a.copy() for a in args]; minus=[a.copy() for a in args]
    plus[0][0,0]+=eps; minus[0][0,0]-=eps
    fd=(direct(*plus,64)[0]-direct(*minus,64)[0])/(2*eps)
    j=base[1][0]; m=(args[-1][:2]+args[-1][2:])/2; h=(args[-1][2:]-args[-1][:2])/2
    expected=(args[1][0]@(m-args[0][0]))[0]*j[0]+args[1][0,0,0]*h[0]*j[1]+args[1][0,1,0]*h[1]*j[2]
    np.testing.assert_allclose(fd,expected,atol=1e-10)

def test_subdivision_basis_conversion():
    args=state(); s=Settings(color_tol=0.,response_tol=0.,max_depth=2)
    a=integrate(*args,s); b=direct(*args,64)
    for x,y in zip(a,b): np.testing.assert_allclose(x,y,atol=1e-10)

def test_enclosure():
    args=state(); l,h=enclose(*args,tolerance=1e-3,max_cells=512)
    v=direct(*args,64)[0]
    assert np.all(l<=v) and np.all(v<=h)
