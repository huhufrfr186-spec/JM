"""Convert calibrated COLMAP text models and RGB images to a JM dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--images',required=True); p.add_argument('--out',required=True); p.add_argument('--train-views',type=int,default=12); p.add_argument('--test-views',type=int,default=4); p.add_argument('--width',type=int,default=1500); p.add_argument('--height',type=int,default=1000); p.add_argument('--seed',type=int,default=20261031); p.add_argument('--max-points',type=int,default=10000)
    a=p.parse_args(); model=Path(a.model); out=Path(a.out); (out/'images').mkdir(parents=True,exist_ok=True)
    cameras={}
    for line in (model/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        fields=line.split(); idx=int(fields[0]); name=fields[1]; w,h=map(int,fields[2:4]); v=list(map(float,fields[4:]))
        if name=='PINHOLE': fx,fy,cx,cy=v
        elif name=='SIMPLE_PINHOLE': fx,cx,cy=v; fy=fx
        else: raise ValueError('Use COLMAP image_undistorter and export a PINHOLE text model before conversion.')
        sx,sy=a.width/w,a.height/h
        cameras[idx]=[[fx*sx,0,(cx+.5)*sx-.5],[0,fy*sy,(cy+.5)*sy-.5],[0,0,1]]
    lines=[l for l in (model/'images.txt').read_text().splitlines() if not l.startswith('#')]
    records=[]; i=0
    while i<len(lines):
        if not lines[i].strip(): i+=1; continue
        f=lines[i].split(); i+=2
        qw,qx,qy,qz=map(float,f[1:5]); W=np.eye(4); W[:3,:3]=Rotation.from_quat([qx,qy,qz,qw]).as_matrix(); W[:3,3]=list(map(float,f[5:8]))
        records.append((int(f[0]),int(f[8]),' '.join(f[9:]),W))
    records.sort(key=lambda r:r[0]); rng=np.random.default_rng(a.seed)
    needed=a.train_views+a.test_views
    if len(records)<needed: raise ValueError(f'Need {needed} calibrated views; found {len(records)}.')
    selected=sorted(rng.choice(len(records),needed,replace=False)); train=set(rng.choice(selected,a.train_views,replace=False).tolist())
    views=[]
    for j in selected:
        idx,cam,name,W=records[j]; filename=f'images/{idx:06d}.png'
        Image.open(Path(a.images)/name).convert('RGB').resize((a.width,a.height),Image.Resampling.LANCZOS).save(out/filename)
        views.append(dict(image=filename,K=cameras[cam],w2c=W.tolist(),split='train' if j in train else 'test'))
    points=[]; rgb=[]
    for line in (model/'points3D.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        f=line.split(); points.append(list(map(float,f[1:4]))); rgb.append(np.array(list(map(float,f[4:7])))/255)
    points=np.asarray(points); rgb=np.asarray(rgb)
    if len(points)>a.max_points:
        sel=rng.choice(len(points),a.max_points,replace=False); points=points[sel]; rgb=rgb[sel]
    if len(points)<2: raise ValueError('Initialization requires at least two points.')
    spacing=cKDTree(points).query(points,k=2)[0][:,1].clip(1e-5)
    scale=spacing[:,None]*np.array([[1.,1.,.3]])
    np.savez_compressed(out/'init.npz',xyz=points,rgb=rgb,scale=scale,opacity=np.full(len(points),.2))
    (out/'scene.json').write_text(json.dumps(dict(initialization='init.npz',views=views),indent=2))

if __name__=='__main__': main()
