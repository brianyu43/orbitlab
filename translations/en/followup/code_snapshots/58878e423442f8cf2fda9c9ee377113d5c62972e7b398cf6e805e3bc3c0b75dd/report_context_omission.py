"""Report the verified information-omission effects and their limits."""
import csv
import json
from common import HERE,dump,r
from dynamics_context_omission import NAME,DATA


def main():
    root=HERE/f'reports/{NAME}';out=root/'across_data';evidence={}
    for path in [root/'verification.json',out/'verification.json',root/'identifiability/verification.json']:
        check=json.loads(path.read_text());assert check['all_passed'];evidence[str(path.relative_to(HERE))]=r.sha(path)
    aggregate=json.loads((out/'verification.json').read_text());assert aggregate['manifest_sha256']==r.sha(out/'manifest.json')
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['files']['metrics.csv']==r.sha(out/'metrics.csv')
    index={tuple(row[k] for k in ['condition','kind','mode','split','category','metric']):row for row in csv.DictReader((out/'metrics.csv').open())}
    used=[]
    def get(condition,kind,split,metric='velocity_mae_pixels_per_frame'):
        key=(condition,kind,'variable_force',split,'all',metric);row=index[key]
        values=[float(row[f'data_{ds}_mean']) for ds in DATA]
        used.append({'key':list(key),'data_seeds':DATA,'values':values})
        return values
    def cells(v):return ' | '.join(f'{x:.5f}' for x in v)
    conditions={'full':'All information','no_net':'Hide the combined years','no_drag':'Resistance concealment','no_context':'Both are hidden'}
    split_names={'test':'General test','count4':'Your object','force_ood':'Non-aesthetic external force'}
    values={(condition,split):get(condition,'joint_exact',split) for split in split_names for condition in conditions}
    ratios={split:[100*(b-a)/a for a,b in zip(values['full',split],values['no_net',split])] for split in ['test','force_ood']}
    assert all(x>0 for vv in ratios.values() for x in vv)
    wrong_full=get('full','wrong_exact','force_ood');wrong_hidden=get('no_net','wrong_exact','force_ood')
    assert all(a>b for a,b in zip(wrong_full,wrong_hidden))
    v=json.loads((root/'verification.json').read_text());same=v['zero_net_structure_training_comparisons']
    assert len(same)==18 and all(x['weights_exactly_equal'] for x in same)
    lines=['# Exercise prediction when environmental information is hidden','',
        f'In a model that rotates both state and external forces together, if the combined effect (gravity + wind) is masked and then learned again, the speed error in a single step of the general test is data-specific.{", ".join(f"{x:.1f}%" for x in ratios["test"])}It has increased. Even in non-learning external forces, both sets of data have deteriorated. This is the observation that environmental directional information contributes to the one-step gain of this model.','',
        'The effect of masking only resistance varied depending on the data and evaluation conditions. Furthermore, models that did not rotate the external force direction together were actually better at hiding the combined effect of non-learned external forces. The performance of finite learning models did not always improve just because they had more information.','',
        '## What did you keep the same?','',
        'Input the exact current object state and behavior. It is not a video recognition or long-term prediction score. It was re-learned from scratch in each of the three conditions that cover both convergence and resistance, and the same masking was applied to both learning and evaluation. The 27 total information conditions reused the already verified variable_force weights. We completed the evaluation of 81 new learning instances and 108 total conditions.','',
        'All models were trained only on variable_force data and were applied directly to other environments. Each model followed 12,000 steps, initial weights, sample selection, and object order changes. Each table\'s row is a single independent data generation seed, and the numbers are the means of three fixed initializations. The independent data consists of three sets, not nine sets.','',
        f'Prediction of transfer under condition 108{v["predictions_replayed"]:,}Dog, scene·contact-specific acts{v["episode_rows_verified"]:,}The dog and counting were replayed. Since it was evaluated repeatedly on the same scene under multiple conditions, the number of independent samples differs from the actual sample size. All environmental, contact categories, and train/validation diagnoses were preserved in the CSV, and the table below shows the hold-out condition for variable_force.','',
        '## Co-rotating model: The effect of environmental information','',
        'Speed average absolute error (px/frame), the lower it is, the better. It is a one-step evaluation that provides the correct state again every moment, and differs from autonomous prediction that accumulates errors over time.','',
        '| Evaluation conditions / Provided information | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |']
    for split,label in split_names.items():
        for condition,info in conditions.items():lines.append(f'| {label} / {info} | {cells(values[condition,split])} |')
    lines+=['',f'If the combined effort is hidden, the error of unlearned external forces is per data.{", ".join(f"{x:.1f}%" for x in ratios["force_ood"])}It increased. The changes when resistance was hidden were inconsistent. The learning resistance range, fixed budget, and results are limited to that structure, and this does not mean that resistance information is generally unnecessary.','',
        '## In the incorrectly applied rotation restrictions, there are other phenomena','',
        'The two models below use the same rotational average structure. The model that rotates only under a state condition fixes the external force vector, while the coupled rotation model rotates both the external force and the state together. They were compared under non-learning external force conditions.','',
        '| Model / Provided information | 640031 | 997101 | 997102 |','| --- | ---: | ---: | ---: |',
        f'| Only status rotates / Full information |{cells(wrong_full)} |',
        f'| Only status rotates / Compulsory hidden |{cells(wrong_hidden)} |',
        f'| Joint rotation / Full information |{cells(values["full","force_ood"])} |',
        f'| Co-rotation / Combined hidden |{cells(values["no_net","force_ood"])} |','',
        'When the combined force is applied, the input external force vector becomes zero, so the function of the two rotational structures becomes identical. In the 18 corresponding learning pairs between combined force concealment and overall environment concealment, the weights were completely the same, and this equivalence was also confirmed in prediction comparison. At this point, the superiority of the two structures is not interpreted as separate experimental effects. It does not mean that the concealed actual external force is zero or that the evaluation distribution itself is rotational symmetric.\n\nThe phenomenon of information being erased in a structure that only rotates in','',
        'and scores improving shows that the structure may be using the provided directional information incorrectly or not generalizing it appropriately. It does not extend to conclusions that the value of information has become negative in the optimal predictor or to a proof of the cause for all models.','',
        '## Cases where you can\'t solve because you don\'t have information and cases where you failed to learn','',
        'In the three pairs separately constructed, even if the current state, behavior, and visible environmental inputs were the same, the next answer differed depending on the hidden environment. A model that gives the same answer to the same input cannot achieve both answers. The minimum squared error for each pair was tested using closed‑form physical equations and numerical methods.','',
        'These three pairs are a concrete example of ambiguity caused by information deficiency. They do not indicate how frequently such cases occur in random evaluation data or calculate the optimal error limit for the entire dataset. Even in the current state, traces of past environments may remain. Therefore, the entire average error difference cannot be explained as indeterminability.','',
        '## Next connection','',
        'The learning, evaluation, and aggregation of environmental information missing-comparison has been completed. By linking the currently ongoing observation lengths 1/2/4/8 experiments, we verify which information is actually recovered by looking at the past more closely. The C07 entire set, SVIB external evaluation, comparison with prior research, actual human judgment, and final research bundle are still not completed.','',
        '[Overall statistics](across_data/metrics.csv) · [Difference in the same initialization](across_data/paired.csv) · [Aggregation verification](across_data/verification.json) · [Same input, different answer examples](identifiability/EXPLANATION_KO.md)']
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    colors=['#3275a8','#d68a34','#5b986a'];plot_data=[]
    for ax,split,title in zip(axes,split_names,['Standard','4 objects','Unseen force']):
        for j,ds in enumerate(DATA):
            y=[values[c,split][j] for c in conditions];ax.plot(range(4),y,'o-',color=colors[j],label=f'Data {ds}')
            plot_data.append({'split':split,'data_seed':ds,'conditions':list(conditions),'values':y})
        ax.set_xticks(range(4),['Full','No force','No drag','Neither']);ax.tick_params(axis='x',labelsize=9)
        ax.set(title=title,ylabel='One-step velocity MAE (px/frame)');ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Joint rotation dynamics: masked inputs in both training and evaluation\nEach point averages 3 fixed initializations; true current state and action provided',fontsize=12)
    for ext in ['png','svg']:fig.savefig(root/f'context_information_effect.{ext}',dpi=160)
    plt.close(fig)
    dump(root/'report_data.json',{'table_values':used,'plot_values':plot_data,'relative_error_increase_percent':ratios})
    dump(root/'report_manifest.json',{'evidence_sha256':evidence,'metrics_sha256':r.sha(out/'metrics.csv'),'source_sha256':r.sha(__file__),
        'files':{f:r.sha(root/f) for f in ['RESULTS_KO.md','context_information_effect.png','context_information_effect.svg','report_data.json']},
        'scope':'Verified context omission study. C07 observation-length study and full research objective still pending.'})
    print('context report and figure written',flush=True)


if __name__=='__main__':main()
