import csv,json,os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'reports/figures';dest.mkdir(exist_ok=True)
rows=list(csv.DictReader((ROOT/'reports/ae_results.csv').open()));colors={'aug':'#2166AC','equivariant':'#D6604D'}
fig,ax=plt.subplots(figsize=(7.2,4.3))
for model in colors:
 means=[];sds=[]
 for n in [256,1024,4096]:
  family='primary' if n==1024 else 'data';a=[float(r['test_foreground_mae']) for r in rows if r['model']==model and r['family']==family and int(r['n'])==n]
  means.append(np.mean(a));sds.append(np.std(a,ddof=1));ax.scatter([n]*len(a),a,color=colors[model],alpha=.65,s=22)
 ax.errorbar([256,1024,4096],means,yerr=sds,label=model,color=colors[model],marker='o',capsize=4)
ax.set_xscale('log',base=4);ax.set_xticks([256,1024,4096],['256','1,024','4,096']);ax.set_xlabel('Unique training base scenes');ax.set_ylabel('Test foreground MAE (lower is better)');ax.set_title('Same 3,000 optimizer steps; 3 initialization seeds');ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/'data_efficiency.png',dpi=180);fig.savefig(dest/'data_efficiency.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.3));groups=[('primary','aug','B1'),('primary','equivariant','B2'),('parameter','equivariant','B2 matched\nparameters'),('time','aug','B1 matched\ntime'),('time','equivariant','B2 matched\ntime')]
for i,(family,model,label) in enumerate(groups):
 a=[float(r['test_foreground_mae']) for r in rows if r['family']==family and r['model']==model];ax.bar(i,np.mean(a),color=colors[model],alpha=.7);ax.errorbar(i,np.mean(a),yerr=np.std(a,ddof=1),color='black',capsize=4);ax.scatter([i]*len(a),a,color='black',s=15)
ax.set_xticks(range(len(groups)),[g[2] for g in groups]);ax.set_ylabel('Test foreground MAE');ax.set_title('Parameter and time controls are separate comparisons');fig.tight_layout();fig.savefig(dest/'fairness.png',dpi=180);plt.close(fig)
if (ROOT/'reports/generation_results.csv').exists():
 rows=list(csv.DictReader((ROOT/'reports/generation_results.csv').open()));fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=True)
 for ax,split in zip(axs,['seen','ood']):
  for model in ['aug','equivariant']:
   means=[];sd=[]
   for steps in [0,16,32,64]:
    a=[float(r['joint_accuracy']) for r in rows if r['run'].startswith('flow_'+model+'_') and int(r['steps'])==steps and r['split']==split];means.append(np.mean(a));sd.append(np.std(a,ddof=1))
   ax.errorbar(range(4),means,yerr=sd,label=model,color=colors[model],marker='o',capsize=3)
  ax.set_xticks(range(4),['Gaussian','16','32','64']);ax.set_title(split+' conditions');ax.set_xlabel('Euler ODE steps');ax.set_ylim(0,1.02);ax.grid(alpha=.2)
 axs[0].set_ylabel('Joint shape/color agreement with condition');axs[1].legend(frameon=False);fig.suptitle('Renderer-template evaluator; mean ± seed SD');fig.tight_layout();fig.savefig(dest/'generation_conditions.png',dpi=180);plt.close(fig)
