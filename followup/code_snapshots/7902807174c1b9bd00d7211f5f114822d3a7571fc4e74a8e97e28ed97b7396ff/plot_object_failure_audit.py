"""Export source-backed held-out segmentation and visibility-bin figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,dump,r


def main():
    source=HERE/'reports/object_layers_heldout_v1/summary.json';failure=HERE/'reports/object_failure_audit_v1/summary.json'
    results=json.loads(source.read_text())['results'];bins=json.loads(failure.read_text())['occlusion_bins']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));colors=['#4a5568','#2477b3','#a54f9d']
    methods=[('component','native','Connected regions'),('flat','decoder_rgb_foreground','Flat + RGB foreground'),('slot','decoder_rgb_foreground','Slot attention + RGB foreground')]
    splits=['test','ood','count3','count4','occlusion'];locations=np.arange(5);width=.25
    for j,(kind,variant,label) in enumerate(methods):
        selected=[next(v for v in results if v['split']==split and v['kind']==kind and v['variant']==variant) for split in splits]
        means=np.array([v['metrics']['matched_visible_iou']['mean'] for v in selected]);cis=np.array([v['metrics']['matched_visible_iou']['base_scene_ci95'] for v in selected]).T
        axes[0].bar(locations+(j-1)*width,means,width,yerr=[means-cis[0],cis[1]-means],color=colors[j],label=label,capsize=2)
        selected=[v for v in bins if v['kind']==kind and v['variant']==variant];means=np.array([v['matched_visible_iou']['mean'] for v in selected]);cis=np.array([v['matched_visible_iou']['base_scene_ci95'] for v in selected]).T
        axes[1].errorbar(np.arange(4),means,yerr=[means-cis[0],cis[1]-means],color=colors[j],marker='o',capsize=3,label=label)
    axes[0].set_xticks(locations,['2 objects','Unseen\nshape-color','3 objects','4 objects','Occluded'])
    axes[0].set_title('Held-out scenes: fixed models and postprocessing')
    counts=[v['scenes'] for v in bins if v['kind']=='component']
    axes[1].set_xticks(np.arange(4),[f'{label}\n(n={n})' for label,n in zip(['<50%','50–75%','75–90%','90–100%'],counts)])
    axes[1].set_xlabel('Least-visible object: fraction of original area visible')
    axes[1].set_title('Occluded scenes: visible-region score by severity')
    for ax in axes:
        ax.set_ylim(0,1.04);ax.set_ylabel('Matched visible-object IoU (higher is better)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False)
    fig.suptitle('Object segmentation: strong simple baseline, unresolved occlusion',fontsize=14)
    fig.tight_layout(rect=(0,.06,1,.94))
    out=HERE/'reports/object_failure_audit_v1';fig.savefig(out/'heldout_segmentation.png',dpi=170,bbox_inches='tight');fig.savefig(out/'heldout_segmentation.svg',bbox_inches='tight');plt.close(fig)
    dump(out/'figure_manifest.json',{'data_sources':{str(p.relative_to(HERE)):r.sha(p) for p in [source,failure]},
        'files':{name:r.sha(out/name) for name in ['heldout_segmentation.png','heldout_segmentation.svg']},
        'script_sha256':r.sha(__file__),'uncertainty':'95% resampling intervals over base scenes, conditional on one learned initialization; not seed uncertainty.',
        'caution':'The occlusion bins contain different scenes. A merged-region baseline can score higher when object visible areas are very unequal; this is not successful occlusion recovery.'})


if __name__=='__main__':main()
