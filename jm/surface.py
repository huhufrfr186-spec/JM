"""Multiview-consistent center depth fusion and TSDF surface extraction."""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

def consistent_depths(depths,cameras,threshold):
    output=[]
    for i,(depth,cam) in enumerate(zip(depths,cameras)):
        yy,xx=np.indices(depth.shape); pix=np.stack((xx,yy,np.ones_like(xx)),-1)
        local=(pix@np.linalg.inv(cam['K']).T)*depth[...,None]
        R=cam['w2c'][:3,:3]; t=cam['w2c'][:3,3]; world=(local-t)@R
        count=np.zeros(depth.shape,int)
        for j,(other,c) in enumerate(zip(depths,cameras)):
            if i==j: continue
            p=world@c['w2c'][:3,:3].T+c['w2c'][:3,3]
            projected=p@c['K'].T; z=p[...,2]; uv=projected[...,:2]/np.maximum(z[...,None],1e-12)
            sampled=map_coordinates(other,[uv[...,1],uv[...,0]],order=0,mode='constant',cval=0.)
            valid=(uv[...,0]>=0)&(uv[...,0]<=other.shape[1]-1)&(uv[...,1]>=0)&(uv[...,1]<=other.shape[0]-1)&(z>0)&(sampled>0)
            count+=valid&(abs(sampled-z)<=threshold)
        valid=(depth>0)&(count>=min(1,len(depths)-1))
        output.append(np.where(valid,depth,0.))
    return output

def fuse(depths,cameras,xyz,voxel=.02,truncation_voxels=4,max_voxels=20000000):
    trunc=voxel*truncation_voxels
    low=xyz.min(0)-trunc; high=xyz.max(0)+trunc
    shape=np.ceil((high-low)/voxel).astype(int)+1
    if np.prod(shape)>max_voxels: raise ValueError('TSDF grid exceeds max_voxels; select a building bounding region or increase max_voxels.')
    volume=np.ones(np.prod(shape),np.float32); observed=np.zeros(len(volume),bool)
    for start in range(0,len(volume),200000):
        flat=np.arange(start,min(start+200000,len(volume)))
        grid=np.stack(np.unravel_index(flat,tuple(shape)),1)*voxel+low
        accum=np.zeros(len(grid)); weight=np.zeros(len(grid))
        for depth,cam in zip(depths,cameras):
            p=grid@cam['w2c'][:3,:3].T+cam['w2c'][:3,3]; proj=p@cam['K'].T
            uv=proj[:,:2]/np.maximum(p[:,2,None],1e-12)
            measured=map_coordinates(depth,[uv[:,1],uv[:,0]],order=0,mode='constant',cval=0.)
            sdf=measured-p[:,2]
            valid=(measured>0)&(p[:,2]>0)&(sdf>=-trunc)
            accum[valid]+=np.clip(sdf[valid]/trunc,-1,1); weight[valid]+=1
        mask=weight>0; volume[flat[mask]]=accum[mask]/weight[mask]; observed[flat[mask]]=True
    v,f,_,_=marching_cubes(volume.reshape(tuple(shape)),level=0,spacing=(voxel,)*3,mask=observed.reshape(tuple(shape)))
    return v+low,f

def write_ply(path,vertices,faces=None):
    faces=np.empty((0,3),int) if faces is None else faces
    with open(path,'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        np.savetxt(f,vertices,fmt='%.8f')
        for a,b,c in faces: f.write(f'3 {a} {b} {c}\n')
