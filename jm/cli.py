import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(prog='jm'); sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo'); demo.add_argument('--out',default='data/demo')
    train=sub.add_parser('train'); train.add_argument('--data',required=True); train.add_argument('--config',default='configs/T.json'); train.add_argument('--out',default='runs/jm'); train.add_argument('--device',default='cpu')
    for command in ('render','extract'):
        q=sub.add_parser(command); q.add_argument('--data',required=True); q.add_argument('--checkpoint',required=True); q.add_argument('--config',default='configs/T.json'); q.add_argument('--out',required=True); q.add_argument('--device',default='cpu')
    metric=sub.add_parser('metrics'); metric.add_argument('--prediction',required=True); metric.add_argument('--reference',required=True); metric.add_argument('--threshold',type=float,default=.1); metric.add_argument('--out',default='metrics.json')
    args=p.parse_args()
    if args.command=='metrics':
        from .metrics import surface_metrics
        result=surface_metrics(np.load(args.prediction),np.load(args.reference),args.threshold)
        Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return
    if args.command=='train':
        from .train import train
        train(args.data,args.config,args.out,args.device); return
    import torch
    from .data import load_dataset,restore_scene
    from .render import render
    from .integrator import Settings
    from PIL import Image
    if args.command=='demo':
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
        xyz=np.array([[-.08,0,2],[.09,.02,2.12]])
        np.savez(out/'init.npz',xyz=xyz+np.array([.015,0,0]),rgb=np.array([[.8,.2,.1],[.1,.6,.8]]),scale=np.array([[.12,.1,.02],[.13,.11,.025]]),opacity=np.array([.6,.55]))
        manifest={'initialization':'init.npz','views':[]}
        for i,offset in enumerate([-.05,.05,0.]):
            W=np.eye(4); W[0,3]=offset
            manifest['views'].append(dict(image=f'{i}.png',K=[[8,0,1.5],[0,8,1.5],[0,0,1]],w2c=W.tolist(),split='train' if i<2 else 'test'))
            Image.fromarray(np.zeros((4,4,3),np.uint8)).save(out/f'{i}.png')
        (out/'scene.json').write_text(json.dumps(manifest,indent=2))
        scene,cameras,_,_=load_dataset(out); scene['xyz'].data.copy_(torch.tensor(xyz))
        for i,cam in enumerate(cameras):
            with torch.no_grad(): image=render(scene,cam)
            Image.fromarray((image.numpy().clip(0,1)*255).round().astype(np.uint8)).save(out/f'{i}.png')
        return
    cfg=json.loads(Path(args.config).read_text()); settings=Settings(**cfg['integration'])
    scene=restore_scene(args.checkpoint,args.device); _,cameras,_,splits=load_dataset(args.data,args.device)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); depths=[]; arraycams=[]
    for i,cam in enumerate(cameras):
        with torch.no_grad(): image=render(scene,cam,settings,depth=True).cpu().numpy()
        Image.fromarray((image[...,:3].clip(0,1)*255).round().astype(np.uint8)).save(out/f'{i:04d}.png')
        coverage=image[...,4]; depth=np.where(coverage>=cfg['coverage_threshold'],image[...,3]/np.maximum(coverage,1e-12),0.)
        np.save(out/f'{i:04d}_depth.npy',depth); np.save(out/f'{i:04d}_coverage.npy',coverage)
        if splits[i]=='train':
            depths.append(depth); arraycams.append({k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in cam.items()})
    if args.command=='extract':
        from .surface import consistent_depths,fuse,write_ply
        depths=consistent_depths(depths,arraycams,4*cfg['voxel_size'])
        vertices,faces=fuse(depths,arraycams,scene['xyz'].detach().cpu().numpy(),cfg['voxel_size'],cfg['truncation_voxels'])
        write_ply(out/'surface.ply',vertices,faces)

if __name__=='__main__': main()
