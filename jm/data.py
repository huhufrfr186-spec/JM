import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load_dataset(path,device='cpu'):
    root=Path(path); manifest=json.loads((root/'scene.json').read_text(encoding='utf-8'))
    cameras=[]; images=[]; splits=[]
    for view in manifest['views']:
        image=np.asarray(Image.open(root/view['image']).convert('RGB'),dtype=np.float64)/255.
        cameras.append({'K':torch.tensor(view['K'],dtype=torch.float64,device=device),'w2c':torch.tensor(view['w2c'],dtype=torch.float64,device=device),'height':image.shape[0],'width':image.shape[1]})
        images.append(torch.tensor(image,dtype=torch.float64,device=device)); splits.append(view['split'])
    a=np.load(root/manifest['initialization'])
    xyz=np.asarray(a['xyz']); n=len(xyz)
    scale=np.asarray(a['scale']) if 'scale' in a else np.full((n,3),.02)
    if scale.shape==(n,): scale=np.repeat(scale[:,None],3,axis=1)
    shape=np.zeros((n,3,3)); shape[:,range(3),range(3)]=np.log(scale)
    rgb=np.asarray(a['rgb']) if 'rgb' in a else np.full((n,3),.5)
    opacity=np.asarray(a['opacity']) if 'opacity' in a else np.full(n,.2)
    logit=lambda x:np.log(np.clip(x,1e-6,1-1e-6)/np.clip(1-x,1e-6,1))
    scene={k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float64,device=device)) for k,v in {'xyz':xyz,'shape':shape,'color':logit(rgb),'opacity':logit(opacity)}.items()}
    if not any(x=='train' for x in splits): raise ValueError('At least one training view is required.')
    return scene,cameras,images,splits

def save_scene(scene,path):
    np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in scene.items()})

def restore_scene(path,device='cpu'):
    return {k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float64,device=device)) for k,v in np.load(path).items()}
