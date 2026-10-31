import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from .integrator import Settings
from .data import load_dataset,save_scene
from .render import render,objective_interval

def pairs(scene,k=6):
    xyz=scene['xyz'].detach().cpu().numpy()
    if len(xyz)<2: return None
    dist,idx=cKDTree(xyz).query(xyz,k=min(k+1,len(xyz)))
    a=np.repeat(np.arange(len(xyz)),idx.shape[1]-1); b=idx[:,1:].ravel(); spacing=dist[:,1:].ravel()
    valid=spacing>1e-10; a=a[valid]; b=b[valid]; spacing=spacing[valid]
    return tuple(torch.as_tensor(v,device=scene['xyz'].device) for v in (a,b,spacing))

def regularizer(scene,neighbors):
    if neighbors is None or len(neighbors[0])==0: return scene['xyz'].sum()*0
    shape=scene['shape']; L=torch.tril(shape,diagonal=-1)+torch.diag_embed(torch.exp(torch.diagonal(shape,dim1=-2,dim2=-1)))
    eigenvalue,eigenvector=torch.linalg.eigh(L@L.transpose(-1,-2))
    a,b,spacing=neighbors
    # Degenerate eigenspaces have undefined normals and are excluded.
    valid=((eigenvalue[a,1]-eigenvalue[a,0])>1e-10)&((eigenvalue[b,1]-eigenvalue[b,0])>1e-10)
    if not valid.any(): return scene['xyz'].sum()*0
    a=a[valid]; b=b[valid]; spacing=spacing[valid]
    normal=eigenvector[:,:,0]; delta=scene['xyz'][a]-scene['xyz'][b]
    value=((normal[a]*delta).sum(1)**2+(normal[b]*delta).sum(1)**2)/(2*spacing**2)+1-(normal[a]*normal[b]).sum(1)**2
    return value.mean()

def train(dataset,config,out,device='cpu'):
    cfg=json.loads(Path(config).read_text()); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    scene,cameras,images,splits=load_dataset(dataset,device)
    cameras=[c for c,s in zip(cameras,splits) if s=='train']; images=[i for i,s in zip(images,splits) if s=='train']
    settings=Settings(**cfg['integration']); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)); start=time.perf_counter(); accepted=0; attempted=0
    with (out/'training.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['attempt','accepted','elapsed_seconds','loss_lower','loss_upper','step'])
        while accepted<cfg['updates'] and time.perf_counter()-start<cfg['time_seconds']:
            attempted+=1; neighbors=pairs(scene); reg=cfg['lambda_geometry']*regularizer(scene,neighbors)
            loss=reg
            for c,target in zip(cameras,images):
                prediction=render(scene,c,settings)
                loss=loss+.5*((prediction-target)**2).sum(-1).mean()/len(cameras)
            for p in scene.values(): p.grad=None
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in scene.values()): raise FloatingPointError('Nonfinite scene gradient.')
            before={k:p.detach().clone() for k,p in scene.items()}
            step=cfg['learning_rate']; success=False
            for trial in range(cfg['line_search_steps']):
                # New camera projection and depth sorting happen at every evaluation.
                with torch.no_grad():
                    for k,p in scene.items():
                        p.copy_(before[k] if p.grad is None else before[k]-step*p.grad)
                    newreg=float(cfg['lambda_geometry']*regularizer(scene,neighbors))
                tol=cfg['objective_tolerance']; old_interval=new_interval=None
                for refinement in range(cfg['objective_refinements']):
                    with torch.no_grad():
                        new_interval=objective_interval(scene,cameras,images,tol,cfg['interval_cells']*2**refinement,newreg)
                        candidate={k:p.detach().clone() for k,p in scene.items()}
                        for k,p in scene.items(): p.copy_(before[k])
                        old_interval=objective_interval(scene,cameras,images,tol,cfg['interval_cells']*2**refinement,float(reg.detach()))
                        for k,p in scene.items(): p.copy_(candidate[k])
                    if new_interval[1]<old_interval[0]: success=True; break
                    if new_interval[0]>=old_interval[1]: break
                    tol*=.25
                if success: break
                step*=.5
            if success: accepted+=1
            else:
                with torch.no_grad():
                    for k,p in scene.items(): p.copy_(before[k])
            current=new_interval if success else old_interval
            writer.writerow([attempted,accepted,time.perf_counter()-start,*current,step]); f.flush()
            save_scene(scene,out/'scene.npz')
            if not success: break
    (out/'run.json').write_text(json.dumps({'accepted_updates':accepted,'attempts':attempted,'elapsed_seconds':time.perf_counter()-start},indent=2))
