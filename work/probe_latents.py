#!/usr/bin/env python3
"""Fit linear probes and a learned 90-degree action. Never fit on test/OOD.

Requires: pip install scikit-learn
Input: train/val/test/ood_latents.npz produced by orbitlab.py analyze.
CIs and matching model/compute budgets are intentionally left to the research phase.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(); root=Path(args.input); out=Path(args.out)
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    data={s:np.load(root/f"{s}_latents.npz") for s in ("train","val","test","ood")}
    raw={s:d["z"].reshape(len(d["z"]),-1).astype(np.float64) for s,d in data.items()}
    scaler=StandardScaler().fit(raw["train"])
    features={s:scaler.transform(z) for s,z in raw.items()}
    result={"split_policy":"Fit on train, choose hyperparameters on val, evaluate test and held-out factor pairs."}
    result["probes"]={}
    for target in ("shape","color","rotation"):
        labels={s:(d["rotation"] if target=="rotation" else d["labels"][:,0 if target=="shape" else 1]) for s,d in data.items()}
        best=None
        for c in (.01,.1,1.,10.):
            clf=LogisticRegression(C=c,max_iter=1000).fit(features["train"],labels["train"])
            acc=accuracy_score(labels["val"],clf.predict(features["val"]))
            if best is None or acc>best[0]: best=(acc,c,clf)
        result["probes"][target]={"C":best[1],"validation_accuracy":best[0],
            **{f"{s}_accuracy":accuracy_score(labels[s],best[2].predict(features[s])) for s in ("test","ood")}}
    def pairs(z: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        if len(z)%4: raise ValueError("Expected four equally sized orientation blocks")
        shaped=z.reshape(4,-1,z.shape[1])
        return z,np.roll(shaped,-1,axis=0).reshape(z.shape)
    x,y=pairs(features["train"]); vx,vy=pairs(features["val"])
    best=None
    for alpha in (1e-4,1e-2,1.,100.):
        a=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y)
        err=float(np.mean((vx@a-vy)**2))
        if best is None or err<best[0]: best=(err,alpha,a)
    a=best[2]; dim=a.shape[0]
    result["learned_action"]={"ridge_alpha":best[1],"validation_mse":best[0],
        "A4_identity_relative_frobenius":float(np.linalg.norm(np.linalg.matrix_power(a,4)-np.eye(dim))/np.sqrt(dim)),
        "note":"Closure in poorly observed dimensions is not reliable; inspect effective rank. This operator is fitted, unlike the hard-coded C4 shift."}
    for s in ("test","ood"):
        xx,yy=pairs(features[s]); result["learned_action"][f"{s}_relative_mse"]=float(np.mean((xx@a-yy)**2)/max(np.mean(yy**2),1e-12))
    centered=raw["train"]-raw["train"].mean(0)
    eig=np.linalg.svd(centered,compute_uv=False)**2
    probability=eig/max(eig.sum(),1e-12)
    entropy=-sum(p*np.log(p) for p in probability if p>0)
    result["raw_train_effective_rank"]=float(np.exp(entropy))
    result["raw_train_std_mean"]=float(raw["train"].std(0).mean())
    result["note"]="Point estimates only. Bootstrap by base scene, not by rotation or adjacent frame; run >=3 initialization seeds."
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
