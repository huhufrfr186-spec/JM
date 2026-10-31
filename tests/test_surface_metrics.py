import numpy as np
from jm.metrics import surface_metrics,paired_interval
from jm.surface import fuse,consistent_depths

def test_metrics_and_building_interval():
    cloud=np.array([[0.,0.,0.],[1.,0.,0.]])
    score=surface_metrics(cloud,cloud,.1)
    assert score['cd_l1']==0 and score['f_score']==1
    interval=paired_interval(np.ones((6,3))*2,np.ones((6,3)))
    assert interval=={'mean':1.,'lower':1.,'upper':1.}

def test_plane_fusion():
    K=np.array([[10.,0.,3.5],[0.,10.,3.5],[0.,0.,1.]])
    cam=dict(K=K,w2c=np.eye(4))
    d=np.ones((8,8))*2
    depths=consistent_depths([d,d],[cam,cam],.02)
    vertices,faces=fuse(depths,[cam,cam],np.array([[-.1,-.1,1.9],[.1,.1,2.1]]),voxel=.02)
    assert len(faces)>0
    assert np.max(abs(vertices[:,2]-2))<1e-6
