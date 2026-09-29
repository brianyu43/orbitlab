"""Final static research figure, preserving every seed and source binding."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import HERE, dump, r

OUT=HERE/'reports/generation_cost_figure_v1'
COLORS=['#245b8a','#ba812c','#8a557b']


def main():
    OUT.mkdir(exist_ok=False)
    rows=[];sources={}
    def read(rel):
        sources[rel]=r.sha(HERE/rel)
        return json.loads((HERE/rel).read_text())
    def add(panel,category,series,unit,terms,scale=1):
        values=[]
        for rel,pointer in terms:
            value=read(rel)
            for key in pointer:value=value[key]
            values.append(float(value))
        row={'panel':panel,'category':category,'series':series,'unit':unit,
             'value':sum(values)*scale,'scale':scale,'source_terms':terms}
        rows.append(row)
        return row['value']
    fig,axs=plt.subplots(2,3,figsize=(16,9.3),layout='constrained')
    modes=['plain','augment','equivariant'];labels=['Plain','Augmented','C4']
    def dots(ax,panel,categories,series,labels_y,title,ylabel):
        for j,s in enumerate(series):
            for i,c in enumerate(categories):
                v=[x['value'] for x in rows if x['panel']==panel and x['category']==c and x['series']==s]
                x=i+(j-(len(series)-1)/2)*.22
                ax.scatter(x+np.linspace(-.045,.045,len(v)),v,marker=['o','s','^'][j],s=28,
                           color=COLORS[j],alpha=.85,label=labels_y[j] if i==0 else None)
                ax.plot([x-.07,x+.07],[np.mean(v)]*2,color=COLORS[j],lw=2)
        ax.set_xticks(range(len(categories)),categories);ax.set(title=title,ylabel=ylabel)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    for tag in ['fixed','time']:
        for mode in modes:
            for seed in range(3):
                root=f'runs/flow_{tag}_s{seed}_{mode}'
                for split in ['seen','ood']:
                    add('quality_'+tag,split,mode,str(seed),[(root+'/samples/metrics.json',['64',split,'strict_accepted_and_joint'])],100)
                add('train',tag,mode,str(seed),[(root+'/run.json',['train_wall_seconds'])])
                add('sample',tag,mode,str(seed),[(root+'/samples/metrics.json',['64','sample_and_decode_seconds'])])
    dots(axs[0,0],'quality_fixed',['seen','ood'],modes,labels,'Fixed 4,000 flow updates','Strict automatic success (%)')
    dots(axs[0,1],'quality_time',['seen','ood'],modes,labels,'Matched flow training time','Strict automatic success (%)')
    dots(axs[0,2],'train',['fixed','time'],modes,labels,'Flow training cost','Wall time (seconds)')
    dots(axs[1,0],'sample',['fixed','time'],modes,labels,'576 images, 64 integration steps','Sampling + decoding (seconds)')
    summary=read('reports/confirmation_v1/summary.json')
    cmodes=['plain_fixed','plain_time','equivariant_fixed'];clabels=['Plain, fixed','Plain, time','C4, fixed']
    for mode in cmodes:
        for split in ['seen','ood']:
            for i,ds in enumerate(summary['new_data_seeds']):
                add('confirmation',split,mode,str(ds),[('reports/confirmation_v1/summary.json',['summary',mode,split+'_strict_accepted_and_joint','per_data_seed',i])],100)
        for ds in summary['new_data_seeds']:
            indices=[i for i,x in enumerate(summary['runs']) if x['data_seed']==ds and x['mode']==mode]
            assert len(indices)==3
            add('total',mode,'ae_plus_flow',str(ds),[('reports/confirmation_v1/summary.json',['runs',i,key]) for i in indices for key in ['ae_seconds','flow_seconds']],1/3)
    dots(axs[1,1],'confirmation',['seen','ood'],cmodes,clabels,'Independent confirmation: 3 data seeds','Strict automatic success (%)')
    dots(axs[1,2],'total',cmodes,['ae_plus_flow'],['AE + flow'],'Confirmation: full training budget','Wall time (seconds)')
    axs[1,2].set_xticks(range(3),['Plain\nfixed','Plain\ntime','C4\nfixed'])
    fig.suptitle('Generation quality and computation\nTop row and bottom-left: 1 data seed × 3 initializations. Bottom-middle/right: 3 data seeds, each averaged over 3 initializations.',fontsize=13)
    fig.supxlabel('Dots are observed runs/data means; horizontal ticks are averages, not confidence intervals. Shared-host timings; no hardware speedup claim. Human review pending.',fontsize=10)
    for ext in ['png','svg']:fig.savefig(OUT/f'generation_quality_cost.{ext}',dpi=160)
    plt.close(fig)
    dump(OUT/'chart_data.json',{'rows':rows,'source_sha256':sources})
    with (OUT/'chart_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','category','series','unit','value']);writer.writeheader()
        writer.writerows({k:x[k] for k in writer.fieldnames} for x in rows)
    dump(OUT/'manifest.json',{'source_sha256':r.sha(__file__),'source_files':sources,'rows':len(rows),
        'files':{p.name:r.sha(p) for p in OUT.iterdir()},'scope':'Six fixed scientific panels; all seeds shown. Cost includes AE+flow in the confirmation panel; excludes data creation and evaluation. Sampling panel is all 576 images, not 64 images. Timings are observed and not a device benchmark.'})
    print('generation cost figure',len(rows),'bound points',flush=True)


if __name__=='__main__':main()
