"""Complete public-preview report, plots, and fixed unselected image examples."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, dump, r
from svib_preview_data import NAME, ALPHAS, alpha_name


def main():
    root=HERE/f'reports/{NAME}';aggregate=root/'aggregate_v1';summary=json.loads((aggregate/'summary.json').read_text())
    audit=json.loads((aggregate/'verification.json').read_text());assert audit['all_passed'] and audit['summary_sha256']==r.sha(aggregate/'summary.json')
    assert audit['verifier_sha256']==r.sha(HERE/'verify_svib_preview_aggregate.py')
    out=root/'report_v1'
    if out.exists():raise RuntimeError(f'Preserve existing report: {out}')
    out.mkdir();bindings=[];example_records=[]
    def lookup(alpha,method,metric='pixel_mse',category='all',split='heldout'):
        matches=[x for x in summary['conditions'] if (x['alpha'],x['method'],x['metric'],x['category'],x['split'])==(alpha,method,metric,category,split)]
        assert len(matches)==1;return matches[0]
    def bind(x,label):
        item={'label':label,'kind':'paired' if 'left' in x else 'condition','data':x};bindings.append(item);return x
    def mean(alpha,method,**kw):return lookup(alpha,method,**kw)['mean']
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    colors={'identity':'#777777','train_target_mean':'#a66b32','plain':'#2463a6','c4':'#d14836'}
    panels=[('pixel_mse','all','heldout','External test: all 100 scenes'),
            ('changed_pixel_mse','changed','heldout','Changed pixels: 74 changed scenes'),
            ('pixel_mse','unchanged','heldout','No-change test: 26 scenes'),
            ('pixel_mse','all','train','Training: 70 scenes per alpha')]
    for ax,(metric,category,split,title) in zip(axes.flat,panels):
        methods=['plain','c4'] if category=='unchanged' else ['identity','train_target_mean','plain','c4']
        for method in methods:
            stats=[bind(lookup(a,method,metric,category,split),f'plot/{title}/{method}/{a}') for a in ALPHAS]
            y=np.array([x['mean'] for x in stats])*1000;interval=np.array([x['scene_ci95'] for x in stats])*1000
            ax.plot([float(a) for a in ALPHAS],y,'o-',label=method,color=colors[method]);ax.fill_between([float(a) for a in ALPHAS],interval[:,0],interval[:,1],alpha=.10,color=colors[method])
        ax.set(title=title,xlabel='Alpha',ylabel='Pixel MSE x 1000 (lower is better)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];paired=[]
    for a in ALPHAS:
        x=next(x for x in summary['paired_conditions'] if (x['alpha'],x['left'],x['right'],x['split'],x['category'],x['metric'])==(a,'c4','plain','heldout','all','pixel_mse'))
        paired.append(bind(x,f'plot/paired/{a}'))
    ax.axhline(0,color='black',lw=1)
    for i,x in enumerate(paired):
        a=float(ALPHAS[i]);ax.plot(a,x['mean']*1000,'o',color=colors['c4']);ax.vlines(a,*np.array(x['scene_ci95'])*1000,color=colors['c4'])
    ax.set(title='Same-scene difference: C4 minus plain',xlabel='Alpha',ylabel='MSE difference x 1000');ax.grid(alpha=.2)
    ax=axes[1,2];costs=summary['costs']
    for kind in ['plain','c4']:
        rows=[[x['training_seconds'] for x in costs if x['alpha']==a and x['method']==kind] for a in ALPHAS]
        ax.plot([float(a) for a in ALPHAS],np.mean(rows,axis=1),'o-',label=kind,color=colors[kind])
        for a,values in zip(ALPHAS,rows):ax.vlines(float(a),min(values),max(values),color=colors[kind],alpha=.5)
    ax.set(title='Actual local training time (shared host)',xlabel='Alpha',ylabel='Seconds for 2,000 steps');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('SVIB public preview: 70 training pairs per alpha, one shared 100-pair external test\nBands: scene bootstrap after averaging fixed initializations; not independent-dataset intervals',fontsize=13)
    fig.savefig(out/'preview_comparison.png',dpi=150);fig.savefig(out/'preview_comparison.svg');plt.close(fig)
    with np.load(HERE/f'data/{NAME}/heldout/inputs.npz') as a:source=a['source_rgb'].copy()
    with np.load(HERE/f'data/{NAME}/heldout/labels.npz') as a:target=a['target_rgb'].copy();changed=a['changed_mask'].any(axis=(1,2))
    indices=np.r_[np.flatnonzero(changed)[:3],np.flatnonzero(~changed)[:2]].tolist()
    for alpha in [ALPHAS[0],ALPHAS[-1]]:
        methods=['train_target_mean','plain','c4'];arrays=[];rows=[];paths=[]
        for method in methods:
            folder=HERE/(f'reports/{NAME}/baselines/{alpha_name(alpha)}/heldout/{method}' if method=='train_target_mean' else f'runs/{NAME}/{alpha_name(alpha)}/{method}_s0/heldout')
            with np.load(folder/'predictions.npz') as a:arrays.append(a['predictions'].copy().transpose(0,2,3,1))
            rows.append(list(csv.DictReader((folder/'rows.csv').open())));paths.append(folder)
        fig,axes=plt.subplots(5,5,figsize=(11,11.7),constrained_layout=True)
        for row,index in enumerate(indices):
            images=[source[index],target[index]]+[x[index] for x in arrays]
            for col,image in enumerate(images):
                ax=axes[row,col];ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:ax.set_title(['Source','Target','Train-target mean','Plain (seed 0)','C4 (seed 0)'][col],fontsize=10)
                if col==0:ax.set_ylabel(f'Scene {index:03d}\n'+('changed' if changed[index] else 'no change'),fontsize=9)
                if col>=2:
                    value=float(rows[col-2][index]['pixel_mse']);ax.set_xlabel(f'MSE {value:.5f}',fontsize=8)
                    example_records.append({'alpha':alpha,'method':methods[col-2],'seed':None if col==2 else 0,'episode':index,'pixel_mse':value,
                         'prediction_path':str((paths[col-2]/'predictions.npz').relative_to(HERE)),
                         'prediction_sha256':r.sha(paths[col-2]/'predictions.npz'),'rows_path':str((paths[col-2]/'rows.csv').relative_to(HERE)),
                         'rows_sha256':r.sha(paths[col-2]/'rows.csv')})
        fig.suptitle(f'Published preview, alpha={alpha}; fixed first 3 changed + first 2 no-change scenes\nFinal checkpoint, initialization 0; no selection by prediction quality',fontsize=12)
        fig.savefig(out/f'examples_{alpha_name(alpha)}.png',dpi=140);plt.close(fig)
    lines=['# External task connection in SVIB public example','',
        'We trained a small RGB predictor from scratch using 500 publicly available Shape-Swap examples from official dSprites / Single Atomic. We completed validation of all evaluation predictions and scores for 24 models. This experiment is an external task exploration, and the results do not replicate the overall SVIB benchmark performance.','',
        'Input is a single image from the previous image, and output is the image after transformation. The correct object properties, masks, and transformation rules were not entered. A C4 model was compared with a predictor applied to four rotation directions, similar to a general model. The transfer performance of the existing OrbitLab checkpoint was not measured.','',
        '## Comparison conditions with materials','',
        'The 100 pairs per alpha were trained using official sorting and split methods, divided into 70 (training) pairs, 15 (validation) pairs, and 15 (ID evaluation) pairs. All models share a separate Test 100 pair. The correct approach is to keep the 74 pairs where the pixels actually change and leave the 26 pairs unchanged. Different alphas and initializations are not counted separately in the independent data repetition.','',
        'The same parameters were used for 1,266,467 iterations, with the same initial weights and mini-batch order, and were fixed by 2,000 updates. All checkpoints of the 2 × alpha 4 × initialization 3 method were evaluated, and no selection based on scores was made. C4 averages the four baseline predictions, so it is not a comparison of identical computational loads.','',
        '## External 100 pairs of pixel error','',
        'The unit of the table is pixel MSE × 1,000, and the lower it is, the better. General/C4 uses three fixed initializations as the mean. The meaning of the success rate in shape exchange is not meaningful, and one cannot claim that the object structure has been understood only with a low pixel error.','',
        '| alpha | Keep it as is | Average learning answer | General | C4 |','| --- | ---: | ---: | ---: | ---: |']
    for alpha in ALPHAS:
        cells=[bind(lookup(alpha,k),f'table/main/{alpha}/{k}') for k in ['identity','train_target_mean','plain','c4']]
        lines.append('| '+alpha+' | '+' | '.join(f"{x['mean']*1000:.3f}" for x in cells)+' |')
    wins=sum(v<0 for x in paired for v in x['per_seed_mean'])
    means_better=sum(x['mean']<0 for x in paired)
    beats_identity={k:sum(mean(a,k)<mean(a,'identity') for a in ALPHAS) for k in ['plain','c4']}
    lines+=['',f'The overall pixel error of C4 is lower than that of the general model, which is the alpha average.{means_better}/4 pieces, pair comparison of the same initialization{wins}/It was 12. The alpha average is lower than the comparison of keeping it as it is, which is normal.{beats_identity["plain"]}/4 pieces, C4{beats_identity["c4"]}/4. This number is a technical statistic that reuses the same small external evaluation set.','',
        '## Actual areas to change and scenes to preserve','',
        '| alpha | Changed pixel error: Normal | Changed pixel error: C4 | Unchanged scene error: Normal | Unchanged scene error: C4 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for a in ALPHAS:
        cells=[bind(lookup(a,k,metric='changed_pixel_mse',category='changed'),f'table/changed/{a}/{k}') for k in ['plain','c4']]
        cells += [bind(lookup(a,k,category='unchanged'),f'table/nochange/{a}/{k}') for k in ['plain','c4']]
        lines.append('| '+a+' | '+' | '.join(f"{x['mean']*1000:.3f}" for x in cells)+' |')
    lines+=['','The change region is a location where the pixels of the source and target differ and is used only for scoring. The 26 pairs of change region points that did not change are left as missing. The contrast error of keeping unchanged in a scene with no change is 0. The amount of the original parts that should be maintained has also been separately stored as the unchanged-pixel indicator for each scene.','',
        '## Learning cost and rotation consistency','',
        '| alpha | General learning beginner | C4 learning beginner | General 100-page CPU inference beginner | C4 100-page CPU inference beginner |',
        '| --- | ---: | ---: | ---: | ---: |']
    for a in ALPHAS:
        row=[]
        for key in ['training_seconds','cpu_forward_seconds_100_images']:
            for kind in ['plain','c4']:row.append(float(np.mean([x[key] for x in costs if x['alpha']==a and x['method']==kind])))
        lines.append('| '+a+' | '+' | '.join(f'{x:.3f}' for x in row)+' |')
    rotation=max(x['rotation_maximum_absolute_gap'] for x in costs if x['method']=='c4')
    lines+=['',f'The maximum pixel consistency error for all external inputs of the C4 model over three rotations is{rotation:.3e}This is a consistency check of output rotation and not a guarantee of the accuracy of the target image. Training was performed on local MPS, and storage checkpoint inference was recorded on the CPU. Since we shared the other research work and computers, the time in the table is a measure of this execution and not a general speed ranking.','',
        '## Difference from the official protocol','',
        '| Item | Official material·code | This execution |','| --- | --- | --- |',
        '| Scope | 12 tasks of multiple domains and rules | dSprites / Single Atomic public preview one task |',
        '| Material size | 64,000 learning pool·8,000 tests per assignment | 100 pairs of pool·common tests per alpha |',
        '| Split | After sorting 70%/15%/15% | Pairs of 70/15/15 with the same rule |',
        '| Evaluation sample | Code basic batch/drop_last/max_images setting | All 100 pairs up to the last bundle |',
        '| MSE | Divide pixel/channel error sum into image count | Save both element-wise MSE and image-wise sum; 49,152 times correlation verification |',
        '| LPIPS | Included in official evaluation indicators | Unmeasured |',
        '| Model | Models compared in the paper and learning settings | Small residual CNN general/C4, fixed 2,000 updates |','',
        'We confirmed the separation between the learning source combinations and test source combinations, but some combinations of the learning target overlap with the test source. We recorded the exposure of source and target separately, and we do not interpret this fact as pixel duplication or a leakage of the entire official benchmark. We preserved the original stored RGB channels and pixels.','',
        '## Verification and interpretation scope','',
        'We reproduced the 4,800 predictions and 7,200 rotation input predictions of 24 models. We examined all pixel/region scores, 1,536 counts, and 1,920 pair differences, as well as the learning initial values, sample order, and final optimizer state. This is not a re-run of the entire MPS training set. The intervals are conditional bootstrap conditions after re-sampling unique scenes from the initialization mean, and they do not represent uncertainty about new datasets.','',
        'In the first baseline verification, we found approximately 10⁻¹⁷ residual values in the average computation order, and v2, which was tested within the existing allowable range, passed. In the separate rotation formula verification, we separated the float32 batch computation differences of approximately 5.6×10⁻⁶ into float64 formula verification. The allowable criteria for stored prediction/indicators and actual output playback/rotation verification were not changed, and we preserved the failure codes and logs.','',
        'Overfitting is possible in the small learning materials of the public examples. Only with this result can one claim superiority over the latest methods in natural language editing, stable object manipulation, external transfer of existing checkpoints, or otherwise. Executing the entire official data and the comparison model of the original author using the same protocol was not included in this reduced experiment.','',
        '![All alpha comparison](report_v1/preview_comparison.png)','',
        '![Fixed example alpha 0](report_v1/examples_alpha_0p0.png)','',
        '![Fixed example alpha 0.6](report_v1/examples_alpha_0p6.png)','',
        '[Official project](https://systematic-visual-imagination.github.io/) · [Execution plan](../../planning_B06_SVIB_PREVIEW_KO.md) · [Data verification](data_verification.json) · [Model verification](model_verification.json) · [Overall aggregation](aggregate_v1/summary.json) · [Aggregation verification](aggregate_v1/verification.json)']
    report=root/'RESULTS_KO.md';report.write_text('\n'.join(lines)+'\n')
    dump(out/'manifest.json',{'summary_sha256':r.sha(aggregate/'summary.json'),'aggregate_verification_sha256':r.sha(aggregate/'verification.json'),
        'source_sha256':r.sha(__file__),'report_sha256':r.sha(report),'bindings':bindings,'cost_records':costs,
        'fixed_example_selection':{'changed':indices[:3],'unchanged':indices[3:],'alphas':[ALPHAS[0],ALPHAS[-1]],'seed':0},
        'examples':example_records,'all_example_input_sha256':r.sha(HERE/f'data/{NAME}/heldout/inputs.npz'),
        'all_example_target_sha256':r.sha(HERE/f'data/{NAME}/heldout/labels.npz'),
        'derived_counts':{'c4_better_alpha_means':means_better,'c4_better_seed_pairs':wins,'model_better_than_identity':beats_identity},
        'files':{p.name:r.sha(p) for p in out.iterdir() if p.is_file()}})
    print('SVIB report created',len(bindings),'bound statistics,',len(example_records),'example prediction cells',flush=True)


if __name__=='__main__':main()
