#!/usr/bin/env python3
"""Small C4 representation + conditional latent-flow laboratory.

No pretrained weights, external dataset, custom kernels or CUDA are required.
A working starter, not a reproduction of RAE, JEPA, or Flow Equivariant World Models.
C4 acts on the ENTIRE image about its center. Its latent action is a cyclic shift.
"""
from __future__ import annotations
import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ("L", "T", "arrow", "zigzag")
# Synthetic data factors, not chart styling.
PALETTE = ((225, 70, 65), (65, 190, 110), (65, 115, 225),
           (235, 180, 60), (185, 90, 215), (60, 195, 210))
POLYGONS = (
    [(-1,-1),(-.3,-1),(-.3,.3),(1,.3),(1,1),(-1,1)],
    [(-1,-1),(1,-1),(1,-.3),(.3,-.3),(.3,1),(-.3,1),(-.3,-.3),(-1,-.3)],
    [(-1,-.4),(.15,-.4),(.15,-1),(1,0),(.15,1),(.15,.4),(-1,.4)],
    [(-1,-1),(.3,-1),(.3,-.3),(1,-.3),(1,1),(-.3,1),(-.3,.3),(-1,.3)],
)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Use native arm64 Python on a supported Mac, or --device cpu.")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "mps": torch.mps.synchronize()


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "numpy": np.__version__}


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_output(path: str) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new path.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def render_base(index: int, seed: int, size: int, split: str) -> tuple[np.ndarray, tuple[int,int], dict]:
    """Split seeds separate base scenes; held-out kind/color pairs test composition.

    train/val/test use 20 of 24 factor pairs. ood holds out (k,k) for k=0..3.
    C4 variants are made AFTER base-scene splitting. Canonical angle is k=0.
    """
    offsets = {"train": 0, "val": 1, "test": 2, "ood": 3}
    rng = np.random.default_rng(np.random.SeedSequence([seed, offsets[split], index]))
    pairs = [(k,c) for k in range(4) for c in range(6) if ((k == c) == (split == "ood"))]
    kind, color = pairs[int(rng.integers(len(pairs)))]
    radius = float(rng.uniform(.13, .24)) * size
    cx, cy = (rng.uniform(.32, .68, size=2) * size).tolist()
    # Supersampling anti-aliases canonical images; rotations themselves are exact rot90.
    ss = 2
    img = Image.new("RGB", (size*ss, size*ss), (0,0,0))
    vertices = [((cx+x*radius)*ss, (cy+y*radius)*ss) for x,y in POLYGONS[kind]]
    ImageDraw.Draw(img).polygon(vertices, fill=PALETTE[color])
    img = img.resize((size,size), Image.Resampling.LANCZOS)
    info = {"base_id": f"{seed}:{split}:{index}", "kind": kind, "color": color,
            "cx": cx/size, "cy": cy/size, "radius": radius/size}
    return np.asarray(img).copy(), (kind,color), info


def dataset(n: int, seed: int, size: int, split: str) -> tuple[torch.Tensor,torch.Tensor,list[dict]]:
    if n < 1: raise ValueError("Dataset size must be positive")
    items = [render_base(i, seed, size, split) for i in range(n)]
    images = torch.from_numpy(np.stack([x[0] for x in items])).permute(0,3,1,2).float()/255
    labels = torch.tensor([x[1] for x in items], dtype=torch.long)
    return images, labels, [x[2] for x in items]


def rotate(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k % 4, dims=(-2,-1)).contiguous()


def rho(z: torch.Tensor, k: int) -> torch.Tensor:
    return torch.roll(z, shifts=k % 4, dims=1)


class Encoder(nn.Module):
    def __init__(self, size: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.SiLU(),
            nn.Conv2d(16,32,4,2,1),nn.SiLU(),nn.Conv2d(32,32,4,2,1),nn.SiLU(),
            nn.Flatten(),nn.Linear(32*(size//8)**2,latent))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


class Decoder(nn.Module):
    def __init__(self, size: int, latent: int):
        super().__init__(); self.side = size//8
        self.fc = nn.Linear(latent, 32*self.side*self.side)
        self.net = nn.Sequential(nn.ConvTranspose2d(32,32,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU(),nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z).reshape(-1,32,self.side,self.side))


class Autoencoder(nn.Module):
    def __init__(self, size: int=64, model: str="equivariant", channels: int=16):
        super().__init__()
        if size % 8 != 0: raise ValueError("Image size must be divisible by 8")
        self.model, self.channels = model, channels
        latent = channels*4 if model in ("plain","aug") else channels
        self.encoder, self.decoder = Encoder(size,latent), Decoder(size,latent)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model in ("plain","aug"):
            return self.encoder(x).reshape(-1,4,self.channels)
        # E_k(x)=h(R_{-k} x); E(R_r x)=rho_r E(x) exactly in exact arithmetic.
        batch = x.shape[0]
        z = self.encoder(torch.cat([rotate(x,-k) for k in range(4)],dim=0))
        z = z.reshape(4,batch,self.channels).transpose(0,1).contiguous()
        if self.model == "invariant": z = z.mean(1,keepdim=True).expand(-1,4,-1)
        return z
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.model in ("plain","aug"):
            return self.decoder(z.reshape(z.shape[0],-1))
        # D(z)=1/4 sum_k R_k d(z_k), so D(rho_r z)=R_r D(z).
        decoded = self.decoder(z.reshape(-1,self.channels)).reshape(z.shape[0],4,3,-1)
        side = math.isqrt(decoded.shape[-1])
        decoded = decoded.reshape(z.shape[0],4,3,side,side)
        return torch.stack([rotate(decoded[:,k],k) for k in range(4)]).mean(0)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.decode(self.encode(x))


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Foreground weighting avoids rewarding all-black output; no factor labels used.
    weight = 1 + 4*(target.amax(1,keepdim=True) > .05).to(target.dtype)
    return ((pred-target).square()*weight).mean()


def grid(images: torch.Tensor, path: Path, columns: int=8) -> None:
    arr = (images.detach().cpu().clamp(0,1).permute(0,2,3,1).numpy()*255).astype(np.uint8)
    n,h,w,_ = arr.shape
    canvas = Image.new("RGB",(w*columns,h*math.ceil(n/columns)))
    for i,a in enumerate(arr): canvas.paste(Image.fromarray(a),((i%columns)*w,(i//columns)*h))
    canvas.save(path)


def load_ae(path: str, device: torch.device) -> tuple[Autoencoder,dict]:
    # Load only locally generated / otherwise trusted checkpoints.
    ck = torch.load(path,map_location="cpu",weights_only=True)
    model = Autoencoder(ck["size"],ck["model"],ck["channels"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model,ck


@torch.no_grad()
def diagnostic(model: Autoencoder, x: torch.Tensor) -> dict:
    z = model.encode(x); rotated_z = model.encode(rotate(x,1))
    pred = model.decode(z); truth = rotate(x,1)
    edited = model.decode(rho(z,1))
    denom = rotated_z.square().mean().clamp_min(1e-8)
    equiv = ((rotated_z-rho(z,1)).square().mean()/denom).item()
    # A fixed rho is meaningful by construction ONLY for the C4 models.
    fft = np.fft.fft(z.cpu().numpy(),axis=1,norm="ortho")
    bands = (np.abs(fft)**2).mean(axis=(0,2))
    mse = (pred-x).square().mean().item()
    mask = x.amax(1,keepdim=True) > .05
    fg_mae = (((pred-x).abs()*mask).sum()/(mask.sum()*3).clamp_min(1)).item()
    return {"reconstruction_mse":mse,"psnr_db":-10*math.log10(max(mse,1e-12)),
            "foreground_mae":fg_mae,"fixed_rho_equivariance_relative_mse":equiv,
            "latent_edit_target_mse":(edited-truth).square().mean().item(),
            "decoder_commutation_mse":(edited-rotate(pred,1)).square().mean().item(),
            "latent_std_mean":z.flatten(1).std(0,unbiased=False).mean().item(),
            "c4_fourier_band_energy":bands.tolist(),
            "note":"Fixed-rho metrics for plain/aug are NOT their best learned-action metrics. Fit A90 separately before comparing representation quality."}


def train_ae(args: argparse.Namespace) -> None:
    seed_all(args.seed); dev=choose_device(args.device); out=new_output(args.out)
    x,_,meta=dataset(args.n,args.data_seed,args.size,"train")
    model=Autoencoder(args.size,args.model,args.channels).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    logs=[]; sync(dev); started=time.perf_counter()
    model.train()
    for step in range(args.steps):
        idx=torch.randint(len(x),(args.batch,))
        batch=x[idx].to(dev)
        if args.model in ("aug","invariant"):
            batch=rotate(batch,random.randrange(4))
        opt.zero_grad(set_to_none=True)
        loss=reconstruction_loss(model(batch),batch)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss. Try FP32/CPU parity check.")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step % max(1,args.steps//20)==0 or step==args.steps-1:
            logs.append({"step":step+1,"loss":float(loss.detach().cpu())})
    sync(dev); elapsed=time.perf_counter()-started
    ck={"state_dict":model.cpu().state_dict(),"size":args.size,"model":args.model,
        "channels":args.channels,"seed":args.seed,"data_seed":args.data_seed,"n":args.n}
    torch.save(ck,out/"ae.pt"); model.to(dev).eval()
    v,_,_=dataset(min(args.eval_n,128),args.data_seed,args.size,"val")
    with torch.no_grad(): metrics=diagnostic(model,v.to(dev))
    dump_json(out/"run.json",{"environment":environment(),"arguments":vars(args),
        "parameters":sum(p.numel() for p in model.parameters()),"train_wall_seconds":elapsed,
        "seconds_per_step_including_periodic_logging":elapsed/args.steps,"loss_log":logs,"validation":metrics})
    dump_json(out/"data_manifest.json",{"split":"train","base_scenes":meta})
    with torch.no_grad(): grid(torch.cat([v[:8],model(v[:8].to(dev)).cpu()]),out/"reconstruction.png")
    print(json.dumps({"out":str(out),"validation":metrics},indent=2))


class Velocity(nn.Module):
    def __init__(self,channels: int=16,equivariant: bool=True):
        super().__init__(); self.channels=channels; self.equivariant=equivariant
        d=4*channels
        self.net=nn.Sequential(nn.Linear(d+1+10,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,d))
    def raw(self,z: torch.Tensor,t: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        cond=torch.cat([F.one_hot(label[:,0],4),F.one_hot(label[:,1],6)],1).to(z.dtype)
        return self.net(torch.cat([z.flatten(1),t[:,None],cond],1)).reshape_as(z)
    def forward(self,z: torch.Tensor,t: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        if not self.equivariant: return self.raw(z,t,label)
        # Invariant conditioning (shape/color) only. Coordinates require transformed c.
        return torch.stack([rho(self.raw(rho(z,k),t,label),-k) for k in range(4)]).mean(0)


def train_flow(args: argparse.Namespace) -> None:
    seed_all(args.seed); dev=choose_device(args.device); out=new_output(args.out)
    ae,ack=load_ae(args.ae,dev)
    if ack["model"] == "invariant": raise ValueError("Invariant-only flow is not part of the starter generation comparison.")
    x,label,_=dataset(args.n,ack["data_seed"],ack["size"],"train")
    codes=[]; labels=[]
    with torch.no_grad():
        for i in range(0,len(x),args.batch):
            b=x[i:i+args.batch].to(dev); y=label[i:i+args.batch]
            # All rotations, staying inside the TRAIN base-scene split.
            for k in range(4): codes.append(ae.encode(rotate(b,k)).cpu()); labels.append(y)
    z=torch.cat(codes); label=torch.cat(labels)
    # Same mean/std across group slots preserves the regular representation.
    mean=z.mean((0,1),keepdim=True); std=z.std((0,1),keepdim=True,unbiased=False).clamp_min(.02)
    z=(z-mean)/std
    flow=Velocity(ack["channels"],ack["model"]=="equivariant").to(dev)
    opt=torch.optim.AdamW(flow.parameters(),lr=args.lr,weight_decay=1e-4)
    logs=[]; sync(dev); started=time.perf_counter()
    for step in range(args.steps):
        idx=torch.randint(len(z),(args.batch,)); target=z[idx].to(dev); y=label[idx].to(dev)
        noise=torch.randn_like(target); t=torch.rand(len(target),device=dev)
        point=(1-t[:,None,None])*noise+t[:,None,None]*target
        opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(flow(point,t,y),target-noise)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite flow loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(),1.0); opt.step()
        if step % max(1,args.steps//20)==0 or step==args.steps-1: logs.append({"step":step+1,"loss":float(loss.detach().cpu())})
    sync(dev); elapsed=time.perf_counter()-started
    torch.save({"state_dict":flow.cpu().state_dict(),"channels":ack["channels"],
        "equivariant":ack["model"]=="equivariant","mean":mean,"std":std,
        "ae_checkpoint":ack,"seed":args.seed},out/"flow.pt")
    dump_json(out/"run.json",{"environment":environment(),"arguments":vars(args),
        "parameters":sum(p.numel() for p in flow.parameters()),"train_wall_seconds":elapsed,"loss_log":logs,
        "note":"Training loss is not a generative-quality result. Evaluate new samples and nearest neighbors separately."})
    print(f"Saved {out/'flow.pt'}")


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    seed_all(args.seed); dev=choose_device(args.device); out=new_output(args.out)
    ck=torch.load(args.flow,map_location="cpu",weights_only=True); ack=ck["ae_checkpoint"]
    ae=Autoencoder(ack["size"],ack["model"],ack["channels"]).to(dev)
    ae.load_state_dict(ack["state_dict"]); ae.eval()
    flow=Velocity(ck["channels"],ck["equivariant"]).to(dev); flow.load_state_dict(ck["state_dict"]); flow.eval()
    y=torch.tensor([[i%4,(i//4)%6] for i in range(args.n)],device=dev)
    # With fixed random seed, make paired equivariance tests possible.
    z=torch.randn(args.n,4,ck["channels"],device=dev)
    eps=z.cpu().clone()
    for i in range(args.ode_steps):
        t=torch.full((args.n,),i/args.ode_steps,device=dev)
        z=z+flow(z,t,y)/args.ode_steps
    raw=z*ck["std"].to(dev)+ck["mean"].to(dev)
    images=ae.decode(raw)
    grid(images,out/"new_samples.png")
    dump_json(out/"labels.json",{"labels":y.cpu().tolist(),"seed":args.seed,"ode_steps":args.ode_steps,
        "note":"Pairs (kind==color) were held out. Their appearance is a test, not guaranteed compositional success."})
    torch.save({"noise":eps,"normalized_latent":z.cpu(),"labels":y.cpu()},out/"samples.pt")
    for i in range(min(4,args.n)):
        frames=[Image.fromarray((ae.decode(rho(raw[i:i+1],k))[0].cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)) for k in range(4)]
        frames[0].save(out/f"latent_rotation_{i}.gif",save_all=True,append_images=frames[1:],duration=500,loop=0)
    print(f"Saved samples in {out}; inspect quality before claiming success.")


@torch.no_grad()
def analyze(args: argparse.Namespace) -> None:
    dev=choose_device(args.device); out=new_output(args.out); ae,ck=load_ae(args.ae,dev)
    results={}
    for split in ("train","val","test","ood"):
        x,labels,meta=dataset(args.n,ck["data_seed"],ck["size"],split)
        # Evaluate every orientation rather than leaking a random frame split.
        split_metrics=[]; all_z=[]
        for k in range(4):
            entries=[]
            for i in range(0,len(x),args.batch):
                b=rotate(x[i:i+args.batch].to(dev),k); z=ae.encode(b)
                entries.append((len(b),diagnostic(ae,b))); all_z.append(z.cpu().numpy())
            total=sum(n for n,_ in entries)
            split_metrics.append({key:sum(n*m[key] for n,m in entries)/total for key in entries[0][1] if isinstance(entries[0][1][key],(int,float))})
        results[split]={"by_rotation":split_metrics,"note":"Per-batch diagnostics; PSNR averaged in dB. Use pooled per-example metrics for the final report."}
        np.savez_compressed(out/f"{split}_latents.npz",z=np.concatenate(all_z),labels=np.tile(labels.numpy(),(4,1)),
            rotation=np.repeat(np.arange(4),len(x)))
        dump_json(out/f"{split}_manifest.json",{"base_scenes":meta})
        if split=="test":
            b=x[:8].to(dev); z=ae.encode(b)
            # Rows: source, reconstruction, latent edited, true rotated target.
            grid(torch.cat([b,ae.decode(z),ae.decode(rho(z,1)),rotate(b,1)]),out/"intervention.png")
            group_mean=z.mean(1,keepdim=True).expand_as(z)
            grid(torch.cat([ae.decode(z),ae.decode(group_mean)]),out/"remove_noninvariant_bands.png")
    dump_json(out/"metrics.json",results); print(f"Saved diagnostics in {out}")


def doctor(args: argparse.Namespace) -> None:
    dev=choose_device(args.device); seed_all(0)
    report=environment(); report["selected_device"]=str(dev)
    cpu=Autoencoder(32,"equivariant",4).eval(); x=torch.rand(2,3,32,32)
    reference=cpu(x).detach(); other=Autoencoder(32,"equivariant",4).to(dev).eval(); other.load_state_dict(cpu.state_dict())
    observed=other(x.to(dev)).detach().cpu()
    err=(reference-observed).abs().max().item(); report["cpu_device_forward_max_abs_error"]=err
    report["forward_parity_pass"]=bool(err < 1e-4)
    if not report["forward_parity_pass"]: raise AssertionError(report)
    a=Autoencoder(32,"equivariant",4).to(dev); a.load_state_dict(cpu.state_dict())
    c=Autoencoder(32,"equivariant",4); c.load_state_dict(cpu.state_dict())
    c(x).square().mean().backward(); a(x.to(dev)).square().mean().backward()
    grad_err=max((pc.grad-pa.grad.cpu()).abs().max().item() for pc,pa in zip(c.parameters(),a.parameters()))
    report["cpu_device_gradient_max_abs_error"]=grad_err; report["gradient_parity_pass"]=bool(grad_err < 1e-4)
    if not report["gradient_parity_pass"]: raise AssertionError(report)
    if args.out: dump_json(Path(args.out),report)
    print(json.dumps(report,indent=2))


def smoke(args: argparse.Namespace) -> None:
    dev=choose_device(args.device); seed_all(0); report={"environment":environment(),"device":str(dev)}
    x,_,_=dataset(4,42,32,"train"); x=x.to(dev)
    ae=Autoencoder(32,"equivariant",4).to(dev)
    z=ae.encode(x)
    errors={"encoder":(ae.encode(rotate(x,1))-rho(z,1)).abs().max().item(),
            "decoder":(ae.decode(rho(z,1))-rotate(ae.decode(z),1)).abs().max().item(),
            "group_closure":(rho(rho(z,1),3)-z).abs().max().item()}
    inv=Autoencoder(32,"invariant",4).to(dev)
    errors["invariant_encoder"]=(inv.encode(rotate(x,1))-inv.encode(x)).abs().max().item()
    flow=Velocity(4,True).to(dev); t=torch.rand(4,device=dev); y=torch.tensor([[0,1]]*4,device=dev)
    errors["flow"]=(flow(rho(z,1),t,y)-rho(flow(z,t,y),1)).abs().max().item()
    for name,err in errors.items():
        if err>1e-4: raise AssertionError(f"{name} equivariance check failed: {err}")
    opt=torch.optim.AdamW(ae.parameters(),lr=1e-3)
    before=next(ae.parameters()).detach().clone()
    loss=reconstruction_loss(ae(x),x); loss.backward(); opt.step()
    assert torch.isfinite(loss) and not torch.equal(before,next(ae.parameters()).detach())
    arr=z.detach().cpu().numpy(); fft=np.fft.fft(arr,axis=1,norm="ortho")
    restored=np.fft.ifft(fft,axis=1,norm="ortho").real
    assert np.allclose(restored,arr,atol=1e-6)
    seen={(k,c) for k in range(4) for c in range(6) if k!=c}; held={(k,k) for k in range(4)}
    assert not (seen & held)
    report.update({"passed":True,"max_absolute_errors":errors,"finite_backward_and_update":True,
        "fourier_roundtrip":True,"factor_pair_sets_disjoint":True,
        "scope":"Correctness/smoke checks only. No trained quality or Mac speed claim."})
    if args.out: dump_json(Path(args.out),report)
    print(json.dumps(report,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("doctor","smoke","train-ae","train-flow","sample","analyze"):
        p=sub.add_parser(name); p.add_argument("--device",choices=("auto","cpu","mps"),default="auto")
        p.add_argument("--out",default=None if name in ("doctor","smoke") else f"runs/{name}")
        p.add_argument("--seed",type=int,default=0)
        if name in ("train-ae","train-flow","analyze"):
            p.add_argument("--n",type=int,default=2048 if name!="analyze" else 128)
            p.add_argument("--batch",type=int,default=32)
        if name in ("train-ae","train-flow"):
            p.add_argument("--steps",type=int,default=1000); p.add_argument("--lr",type=float,default=1e-3)
        if name=="train-ae":
            p.add_argument("--model",choices=("plain","aug","equivariant","invariant"),default="equivariant")
            p.add_argument("--size",type=int,default=64); p.add_argument("--channels",type=int,default=16)
            p.add_argument("--data-seed",type=int,default=42); p.add_argument("--eval-n",type=int,default=32)
        if name in ("train-flow","analyze"): p.add_argument("--ae",required=True)
        if name=="sample":
            p.add_argument("--flow",required=True); p.add_argument("--n",type=int,default=48)
            p.add_argument("--ode-steps",type=int,default=64)
    args=parser.parse_args()
    for attr in ("n","batch","steps","ode_steps","channels","eval_n"):
        if hasattr(args,attr) and getattr(args,attr)<1: parser.error(f"--{attr.replace('_','-')} must be >=1")
    # A small thread count also makes laptop-scale CPU checks responsive.
    torch.set_num_threads(min(4,torch.get_num_threads()))
    functions={"doctor":doctor,"smoke":smoke,"train-ae":train_ae,"train-flow":train_flow,"sample":sample,"analyze":analyze}
    functions[args.command](args)

if __name__=="__main__": main()
