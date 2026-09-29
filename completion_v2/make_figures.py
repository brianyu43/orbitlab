"""Static scientific figures from verified final aggregate; ranges are seed ranges."""
from pathlib import Path
import os,json
B=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR','/tmp/orbitlab-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=json.loads((B/'aggregate.json').read_text());assert r['all_experiments_and_primary_verification_complete']
out=B/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
def save(fig,name):
    fig.tight_layout();fig.savefig(out/f'{name}.png',bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def metric(v,key):return v['metrics'][key]['mean']
splits=['test','ood','count3','count4','occlusion'];labels=['2 objects','Unseen pairs','3 objects','4 objects','Occlusion']
f,axes=plt.subplots(1,2,figsize=(12,4.3))
for ax,branch,arms in zip(axes,['perception','perception_detail'],[['global','crop'],['binary','alpha']]):
    for i,arm in enumerate(arms):
        vv=[r[branch][f'{arm}/{s}']['metrics']['end_to_end_success'] for s in splits];y=np.array([v['mean'] for v in vv])*100
        ax.bar(np.arange(5)+(i-.5)*.34,y,.34,label=arm,color=colors[i],yerr=np.array([[v['mean']-v['min'] for v in vv],[v['max']-v['mean'] for v in vv]])*100,capsize=3)
    ax.set_xticks(np.arange(5),labels,rotation=20);ax.set_ylim(0,105);ax.set_ylabel('Strict edit success (%)');ax.legend();ax.set_title('Initial crop study' if branch=='perception' else 'Fresh full-pose detail study')
f.suptitle('Known-palette perception; range across evaluation-data seed means',y=1.03);save(f,'perception')
f,axes=plt.subplots(1,2,figsize=(11,4))
for i,(variant,v) in enumerate(r['generation'].items()):
    y=[metric(v,f'{s}_strict_accepted_and_joint')*100 for s in ['seen','ood']]
    axes[0].bar(np.arange(2)+(i-.5)*.32,y,.32,label=variant.split('/')[0],color=colors[i])
axes[0].set_xticks([0,1],['Seen pairs','Unseen pairs']);axes[0].set_ylim(0,100);axes[0].set_ylabel('Strict generation success (%)');axes[0].legend();axes[0].set_title('Fresh confirmation means')
d=json.loads((B/'generation_diagnosis/summary.json').read_text())['groups']
axes[1].bar(['Small','Medium','Large'],[100*d[f'test/shape2/size{i}']['strict'] for i in range(3)],color=colors[0]);axes[1].set_ylim(0,100);axes[1].set_ylabel('Strict reconstruction success (%)');axes[1].set_title('Historical arrow reconstruction diagnosis')
save(f,'generation')
f,axes=plt.subplots(2,2,figsize=(11,8))
variants=['one','multi','multi_bounded','force_wall']
for ax,split in zip(axes[0],['test','count4']):
    for i,v in enumerate(variants):
        a=r['dynamics'][f'variable_force/oracle/{v}/{split}'];y=[metric(a,f'h{h}_position_mae') for h in [1,16,61]]
        ax.plot([1,16,61],y,'o-',color=colors[i],label=v)
    ax.set_yscale('log');ax.set_xlabel('Autonomous prediction horizon');ax.set_ylabel('Position MAE (pixels, log scale)');ax.set_title('Two objects' if split=='test' else 'Four objects');ax.legend()
for ax,inp in zip(axes[1],['oracle','rgb_measurement']):
    for i,v in enumerate(variants):
        y=[]
        for split in ['test','count3','count4']:
            a=r['dynamics'][f'variable_force/{inp}/{v}/{split}'];y.append(100*(metric(a,'h61_nonfinite')+metric(a,'h61_finite_blowup')))
        ax.plot([2,3,4],y,'o-',color=colors[i],label=v)
    ax.set_ylim(-2,102);ax.set_xticks([2,3,4]);ax.set_xlabel('Object count');ax.set_ylabel('61-step failure episodes (%)');ax.set_title(inp);ax.legend()
f.suptitle('Variable-force environment: accuracy and catastrophic failure are separate',y=1.01);save(f,'dynamics')
f,axes=plt.subplots(1,2,figsize=(10,4))
for ax,key,title in zip(axes,['mse','changed_pixel_mse_on_changed_scenes'],['Full-image MSE','Changed-pixel MSE (changed scenes)']):
    y=[metric(r['svib'][kind],key) for kind in ['plain','c4']];ax.bar(['plain','C4'],y,color=colors[:2]);ax.set_title(title)
    for i,kind in enumerate(['plain','c4']):
        vals=[v[key] for v in r['svib'][kind]['per_data_seed'].values()];ax.scatter(np.full(len(vals),i),vals,s=22,color='black',zorder=3)
    if key=='mse':ax.axhline(metric(r['svib']['plain'],'identity_mse'),color='gray',ls='--',label='Input copy');ax.legend()
f.suptitle('Official SVIB Hard dSprites Single Atomic; three initializations',y=1.03);save(f,'svib')
print('saved four PNG/PDF figures')
