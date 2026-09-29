#!/usr/bin/env python3
"""Optional frozen DINOv3 CLS/patch-mean audit; gated access may be required.
No images are uploaded. Downloads model weights on first use. Not Mac-tested here.
This script is independent of the small C4 autoencoder.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor,AutoModel
from orbitlab import choose_device,environment


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images",required=True); p.add_argument("--out",required=True)
    p.add_argument("--device",choices=("auto","cpu","mps"),default="auto")
    p.add_argument("--model",default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--limit",type=int,default=200)
    a=p.parse_args(); out=Path(a.out)
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    files=sorted(f for f in Path(a.images).rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})[:a.limit]
    if not files: raise FileNotFoundError("No image files found")
    dev=choose_device(a.device)
    processor=AutoImageProcessor.from_pretrained(a.model)
    # Eager attention avoids assuming a CUDA-only optimized attention path.
    model=AutoModel.from_pretrained(a.model,attn_implementation="eager").to(dev).eval()
    model.requires_grad_(False)
    cls=[]; mean=[]; names=[]
    registers=int(getattr(model.config,"num_register_tokens",0))
    layers=(3,6,9,12)
    for file in files:
        with Image.open(file) as im:
            # A controlled square field of view BEFORE applying exact C4 rotations.
            im=im.convert("RGB"); side=min(im.size)
            left=(im.width-side)//2; top=(im.height-side)//2
            im=im.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.BICUBIC)
        transposes=(None,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270)
        views=[im if t is None else im.transpose(t) for t in transposes]
        inputs=processor(images=views,return_tensors="pt")
        inputs={k:v.to(dev) for k,v in inputs.items()}
        with torch.inference_mode(): result=model(**inputs,output_hidden_states=True)
        chosen=[i for i in layers if i<len(result.hidden_states)]
        cls.append(torch.stack([result.hidden_states[i][:,0] for i in chosen],1).cpu().numpy())
        mean.append(torch.stack([result.hidden_states[i][:,1+registers:].mean(1) for i in chosen],1).cpu().numpy())
        names.append(str(file))
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,cls=np.stack(cls),patch_mean=np.stack(mean),paths=np.array(names),layers=np.array(chosen))
    out.with_suffix('.json').write_text(json.dumps({'model':a.model,'environment':environment(),
        'shape':'[base image, rotation, layer, hidden dimension]',
        'num_register_tokens':registers,'note':'This is feature extraction, not proof of invariance or semantic disentanglement.'},indent=2))
    print(f"Saved {out}")

if __name__=='__main__': main()
