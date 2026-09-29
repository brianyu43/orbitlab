"""Build source-backed summary tables and presentation inputs from saved runs."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text())
def avg(xs): return {'mean':float(np.mean(xs)), 'seed_sd':float(np.std(xs,ddof=1)), 'values':xs}
def fmt(x, digits=4): return f"{x['mean']:.{digits}f} ± {x['seed_sd']:.{digits}f}"
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

ae=read('reports/ae_summary.json'); gen={}; flows={}
for model in ['aug','equivariant']:
    ds=[read(f'runs/flow_{model}_s{s}/samples/metrics.json') for s in range(3)]
    flows[model]=ds
    for steps in [0,16,32,64]:
        for split in ['seen','ood']:
            group=[d[str(steps)][split] for d in ds]
            entry={k:avg([d[k]['mean'] for d in group]) for k in ['shape_correct','color_correct','joint_correct','valid','nn_mse','occupancy']}
            entry['valid_and_joint']=avg([d['valid_correct_n']/d['n'] for d in group])
            entry['coverage']=avg([d['valid_correct_centroid_grid_coverage'] for d in group])
            gen[f'{model}/{steps}/{split}']=entry
summary={'groups':gen,'scope':'3 initialization seeds, one data seed. Each model/seed/step has 576 independent noise vectors: 480 seen and 96 OOD. Same noise is paired across models and ODE steps.'}
(ROOT/'reports/generation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

rep=[]; actions=[]
for task in read('configs/experiment_matrix.json')['tasks']:
    root=ROOT/task['out'];p=root/'analysis/probes.json'
    if not p.exists(): continue
    d=json.loads(p.read_text())
    for space,values in d['features'].items():
        for target in ['shape','color','rotation']:
            for split in ['test','ood']:
                score=values[target][split]
                rep.append({'run':task['id'],'space':space,'target':target,'split':split,'accuracy':score['mean'],'ci_low':score['base_scene_ci95'][0],'ci_high':score['base_scene_ci95'][1],'C':values[target]['C'],'effective_rank':values['effective_rank'],'probe_train_base_scenes':256})
    actions.append([task['id'],f"{d['learned_A90']['test']['relative_mse']['mean']:.3g}",f"{d['learned_A90']['test']['in_data_closure_relative_mse']['mean']:.3g}",f"{d['learned_A90']['A4_identity_relative_frobenius']:.3g}",f"{d['learned_A90']['test']['decoded_target_mse']['mean']:.5f}"])
with (ROOT/'reports/representation_results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rep[0]));w.writeheader();w.writerows(rep)

mainkeys=['primary/aug/n1024','primary/equivariant/n1024','parameter/equivariant/n1024','time/aug/n1024','time/equivariant/n1024']
labels=['B1 rotation enhancement','B2 C4','B2 parameter control','B1 time control','B2 time control']
ae_table=table(['comparison','Parameter','steps average','At the beginning of learning','MAE overview before the test ↓','test IoU ↑','OOD overview MAE ↓'],[[label,f"{ae['groups'][k]['parameters']['mean']:,.0f}",f"{ae['groups'][k]['steps']['mean']:,.0f}",fmt(ae['groups'][k]['train_seconds'],2),fmt(ae['groups'][k]['test_foreground_mae']),fmt(ae['groups'][k]['test_mask_iou']),fmt(ae['groups'][k]['ood_foreground_mae'])] for k,label in zip(mainkeys,labels)])
data_table=table(['Number of scenes','B1 test overview MAE','B2 test overview MAE','B2−B1 paired average'],[[n,fmt(ae['groups'][f"{'primary' if n==1024 else 'data'}/aug/n{n}"]['test_foreground_mae']),fmt(ae['groups'][f"{'primary' if n==1024 else 'data'}/equivariant/n{n}"]['test_foreground_mae']),f"{ae['paired'][str(n)]['mean']:+.5f}"] for n in [256,1024,4096]])
gen_table=table(['Model / conditions','shape','a sack','Shape + color','Form valid','Valid & shape+color','Center coverage'],[[f"{'B1' if model=='aug' else 'B2'} / {split}"]+[fmt(gen[f'{model}/64/{split}'][metric],3) for metric in ['shape_correct','color_correct','joint_correct','valid','valid_and_joint','coverage']] for model in ['aug','equivariant'] for split in ['seen','ood']])
ode_table=table(['ODE steps','B1 seen','B2 seen','B1 OOD','B2 OOD'],[[('Gaussian' if s==0 else s)]+[fmt(gen[f'{m}/{s}/{sp}']['joint_correct'],3) for sp in ['seen','ood'] for m in ['aug','equivariant']] for s in [0,16,32,64]])
control=[]
for model in ['aug','equivariant']:
    run=read(f'runs/flow_time_{model}_s0/run.json');d=read(f'runs/flow_time_{model}_s0/samples/metrics.json')
    control.append([model,run['steps'],f"{run['train_wall_seconds']:.3f}",f"{d['64']['seen']['joint_correct']['mean']:.3f}",f"{d['64']['ood']['joint_correct']['mean']:.3f}"])
nn=read('reports/nn_reference.json')
ae_runs=[read(t['out']+'/run.json') for cfg in ['configs/experiment_matrix.json','configs/generation_ae_repair.json'] for t in read(cfg)['tasks']]
maxrss=max(d['peak_process_rss_bytes'] for d in ae_runs)/2**30
report=f'''# OrbitLab experiment results — 15th round completed report

2026-09-23 · M5 Pro / 64GB · macOS 27 arm64 · Python 3.13.14 · PyTorch 2.14.0 / MPS FP32

**Rotation consistency was confirmed, but the C4 model did not dominate in any indicators of restoration and generation.** In AE learning at the same time, restoration of B1 was good, and in fixed-step flow generation, the condition matching rate of B2 was high. The blurring and shape errors of the generated products are still large. We actually performed the required outputs for 15 task slots, and this does not mean that we learned for 15 days or 15×90 minutes.

## 1. Original and actual scope of actual performance obtained

I copied the `OrbitLab_starter.zip` and `OrbitLab_Mac_ExperimentPlan.docx` files from iCloud Drive to `sources/originals/` and verified their SHA-256, ZIP CRC, and paths. I checked the 11 files in the ZIP and the 299 paragraphs in the Word file. I preserved the [original list and hashes](reports/source_manifest.json), [original audit](reports/CODE_AUDIT_KO.md), and [recovered conversation](sources/conversation_plan.md).

The original `orbitlab.py`, `probe_latents.py`, and `extract_dino.py` were not modified. Additional implementations are located in `research.py`, `probes.py`, and `generation.py`. The actual execution consists of **43 times**, including 29 AE comparisons, 6 AE corrections for generation entry, and 8 flow runs. Separately, the original 200-step pilot run and 1,000-step calibration run were performed, along with CPU replay tests for doctor/smoke/unit tests and the saved checkpoint. DINO and dynamics extensions were not performed within this 15-run range.

## 2. Design, data, entry criteria

We used 4 types of 64×64 RGB shapes and 6 colors. The data seed was fixed to 42. `(shape,color)=(0,0),(1,1),(2,2),(3,3)` was excluded from the training process. The train pool is 4,096, with 256 samples for validation/test/OOD, and the C4 orbit hash for 4,864 scenes is all different. We removed duplicates between 4 train candidates and 1 test candidate. The train 256/1,024/4,096 are used by overlapping the front part of the fixed pool. [Data specification](data/manifest.json)

Initialization seed 0/1/2, AE batch 32, AdamW lr 0.001, FP32, latent total 64 dimensions, fixed final checkpoint rules were set. The external minibatch original of B1/B2 and the rotation noise distribution are identical. B2 evaluates four directions internally. The main experiment runs 6 times, with 3,000 steps; data size, parameters, and time control are separate search comparisons. Parameter control B2 width=31 is +1.15% compared to B1, but the number of parameters is not exactly the same.

The MAE of the original 200-step pilot was blurred to 0.2335. After validation-only calibration, the main comparison was fixed at 3,000 steps. The generation entry criterion was that the validation MSE < 0.5 times the black baseline, the image MAE ≤ 0.10, IoU ≥ 0.70, the accuracy of the shape/color evaluator for the reconstructed image was ≥ 0.90, and the shape preservation of the fixed grid. This **reconstruction image evaluator criterion differs from the latent linear probe 90%.** The latent shape probe failed to achieve 90%.

The restoration color accuracy of 3,000-step B1 seed 2 failed to meet the threshold of 88.28%. Without lowering the threshold, both the two-model × 3 seed models were retrained on 6,000-steps and passed all six models. This supplementary decision was made during validation before opening the test/OOD results. The supplementary models are exclusively for generation and did not replace the 3,000-step AE table below. [Number gate](reports/quality_gate_repair.json), [Actual image review](reports/repair_visual_review.json)

## 3. AE comparison: Data efficiency and cost

Each cell is the average ± sample standard deviation of the 3 initial seeds. test/OOD consists of 256 independent scenes × 4 rotations for each. The raw data CSV and the 95% CI of each model are located in `runs/<id>/analysis/`. Bootstrap first averages the original scenes from four rotations and then resamples them 1,000 times. The seed standard deviation and scene CI measure different levels of uncertainty.{ae_table}The actual learning time for time control is approximately 26.94 seconds on both sides. In this budget, B1 performed about 5,667 steps, and B2 performed about 2,765 steps. The overall MAE is 0.04457 versus 0.07268. The basic B2 has at least slower parameters in this implementation. The overall MAE of the expanded B2 improved, but the learning time increased to 54.79 seconds. These two comparisons are not mixed in a single fairness table.{data_table}In 256 cases, B2 performed worse than all seeds, and in 1,024/4,096 cases, the order of seed performance changed. Therefore, **this experiment does not support the universal data efficiency advantage of the C4 configuration.** The same steps are not the same epoch for different data sizes. Single data seeds and three initializations do not claim population significance.

![Number of data and actual seed value](reports/figures/data_efficiency.png)

## 4. Expression and intervention: distinguishing accurate symmetry and meaning

The test rotation error of this C4 encoder is 0, and the maximum decoder error is approximately 1.19e-7. This is a structural verification of the shared encoder and symmetry-preserving decoder. This design does not replicate all group-convolution layers in the paper. [C4 formulas and limitations](reports/C4_MATH_KO.md), [Cohen·Welling original paper](https://proceedings.mlr.press/v48/cohenc16.html)

The probe was standardized and fitted with **256 scenes × 4 rotations before training**, and C or ridge alpha was selected using 256 validation samples. Do not confuse the n=1,024 for AE training and n=256 for probe fitting. The test/OOD, CI, and rank results for full/m0/m2/m13 were saved in [CSV](reports/representation_results.csv). The B2 full shape accuracy is 42.2/43.0/45.3% for each seed, and the color accuracy is 98.4/96.5/97.7%. The direction probe for m0 is 25% for all three seeds.

When only m0 is left from seed 0, the overall MAE increased from 0.07825 to 0.29146, and the center error increased from 0.00605 to 0.14627. As shown in the bottom row of the figure, a clump with rotational symmetry merged with the contour. The B3 invariant-only also had a test IoU of 0.3435, which was lower. This aligns with the result of losing position and direction. However, removing the band creates latent variables outside the learning distribution, so it cannot be interpreted as a pure meaning separation causal experiment. Even if the linear shape is lower in m1/m3, this does not provide evidence of non‑linear information.

![Input, full restoration, keep only m0](runs/primary_equivariant_n1024_s0/analysis/band_m0.png)

In general AE, the slot shift determined by the human is not treated as the correct rotation, but compared with the A90 learned from the train. The closure below is the four‑fold synthesis of standardized coordinates. The full‑matrix closure is also sensitive to directions that do not occupy data. For example, the in‑data closure of B3 is almost 0 while the matrix closure is 0.866, showing the difference.{table(['practice','A90 relative MSE','A⁴ relative MSE within data','Frobenius relative to matrix A⁴−I','Rotation answer decode MSE'],actions)}Even with a small residual error in B2 A90, the rotation answer decode MSE originally includes the recovery error. The large fixed-rho error in B1 was not used as the basis for performance inferiority.

## 5. Generate flow in fixed AE

We fixed the encoder/decoder of the 6,000-step AE and verified the actual weight invariance. Each flow has 4,000 optimizer steps, a batch size of 128, and the number of parameters on both sides is the same. B2 averages the velocity four times. We did not standardize each slot separately but used the batch+group common channel average and standard deviation. We learned the vector field of the straight path from latent to Gaussian and integrated the Euler ODE. This small experiment uses the principle of [Flow Matching](https://arxiv.org/abs/2210.02747) and is not a replication of large RAEs or world models.

For each model/seed, **the same 576 initial noise** was reused on Gaussian/16/32/64 steps. For each of the 24 conditions, 24 samples were used, resulting in 480 seen samples and 96 OOD samples. The noise SHA between B1/B2 also matches. The total of 8 flow × 4 conditions = 18,432 evaluation rows does not mean there are 18,432 independent noise samples. It means that the same noise was evaluated repeatedly.

In the clean 512-frame × 4 rotation of the independent renderer seed, the evaluator reported 100% for both shape and color, with 0 duplicates in the train orbit. Since the generated output is blurry, we cannot guarantee that this accuracy is transferred to the previous one. `valid` means the silhouette template IoU is ≥0.65 and not empty, which is not the human-verified success rate of the generation.

**64-step results, 3-seed average±SD, ratio 0-1:**{gen_table}Moving from Gaussian to flow improves condition matching. The simultaneous seen matching rate for B2 is 57.0%, while for B1 it is 34.0%. However, when both validity and condition matching are simultaneously required, the rates are approximately 52.2% and 24.7% respectively. Therefore, **the relative improvement in condition control is observed but not considered sufficient for high-quality generation completion.** OOD is not a new shape or color, but rather a combination of four previously observed factors. The condition distributions for OOD and seen are also not identical.{ode_table}The ODE steps increase condition did not always improve accuracy. The latent differences between 16→32 and 32→64 decreased, but since the scales of B1 and B2 latents were different, the raw latent MSE was not used for model quality comparison. The coupled-noise ODE equivariance error for the B2 seed set was 0, and the decoder exchange error was approximately 1.19e-7. The fixed-rho GIF for B1 lacks structural rotation guarantees. The B2 GIF is presented as evidence of rotation control.

![Gaussian/ODE and condition match](reports/figures/generation_conditions.png)

Time control flow is a search comparison of seed 0. It has matched approximately 10.57 seconds of the **flow learning part**. This is not a comparison of the entire pipeline, which also includes the training time of the previous AE or the cost of ODE inference.{table(['a model','Real steps','At the beginning of learning','seen simultaneous matching','OOD simultaneous matching'],control)}## 6. New sample, recent neighbors, failure cases

For the 64-step fixed version, the first 48 samples were saved, and the template suitability was at least 16 samples, with a learning recentness distance of at least 8 samples. In B2, small stains, rounded corners, incorrect shapes, and inaccurate colors remained. In B1, the combination of separate pieces and shapes was more frequently observed. The image review list and range are found in [Visual Review Records](reports/result_visual_review.json).

![B2 fixed first 48 pieces](runs/flow_equivariant_s0/samples/samples_64.png)

![B2 16 templates suitable for the lowest](runs/flow_equivariant_s0/samples/lowest_template_fit.png)

![B1 16 templates suitable for the lowest](runs/flow_aug_s0/samples/lowest_template_fit.png)

![Generated sample above, recent-neighbor learning image below](runs/flow_equivariant_s0/samples/nearest_train_pairs.png)

It is the exact FP32 RGB MSE compared to the four rotations of 1,024 recent train scenes. The average distance of independent real tests is{nn['test']['mean']:.5f}, OOD is{nn['ood']['mean']:.5f}All. The average of generated seen is B1 0.01140, B2 0.00896. No copies less than 1e-7 were observed in the 64-step 3,456 images of the main generation. This pixel-based criterion almost excludes all copied images or cannot prove distribution learning. Especially blur and background can reduce distance.

The valid and condition-matching sample for seen filled out a 4×4 centroid grid for both models. OOD coverage was 0.625 for B1 and 0.958 for B2. Coverage is the position grid summing up all conditions and is not a proof of the diversity of the overall distribution. The ratio of the area outside the reference region and the scale/centroid standard deviation were also preserved in `samples/metrics.json`. The occupancy of the generated seen was 0.0997 for B1 and 0.1094 for B2, which is the actual test's{nn['real_test_state_reference']['occupancy']['mean']:.4f}It was larger. The distribution of observation location and size also does not match at all.

## 7. Resources, reproduction, limitations

The benchmark measured 100-steps three times after warm-up 20 and synchronized MPS. During the training time, model/data setup and final evaluation are not included, and the speed of that segment is not used as the total execution time. The maximum process RSS among the recorded 35 original and supplementary AE is{maxrss:.3f}It's GiB. This is not the peak memory of the entire device including MPS, and the peak RSS of the flow was not measured separately. MPS driver allocation is the final value of AE. Environment fallback was not used.

The range of single data generation seed, three initialization seed, and small synthetic C4 world is maintained. Parameter control is approximation matching, and time control is the value measured on this Mac. The shape evaluation metric for generation was only independently verified on clean data. The probe was limited to 256 train scenes. After testing, the baseline was lowered or only good seeds were replaced. The 3,000-step and 6,000-step AE results are not combined with the same learning budget.

Code, settings, original, checkpoint, CSV, environment lock, and hashes were preserved. [Reproduce command](REPRODUCE.md), [15th session evidence report](reports/SESSION_COMPLETION_KO.md), [Completion check](reports/completion_checks.json), [6-minute presentation](deliverables/OrbitLab_6min.pptx) can be reviewed together. The CPU replay test copies the absolute path of the storage flow from a separate copy and checks the path to generate samples with the same weight. The verification that was performed on the second Mac for the entire 43rd learning session is not a re-execution.

## 8. The decision of the following experiment

The 15-round assessment covers the following: confirming the lack of universal data efficiency superiority in H1, verifying the loss in position and direction of invariant pooling in H2, assessing relative improvement in the alignment of generation conditions in H3 and the incomplete quality, and checking the difference in parameter count and actual costs in H4. Upon further execution, the appropriate order is to first assess the **independence of human evaluation/evaluation tool robustness for the generated outputs**, followed by repeated data seeding and fixing the overall AE+flow cost control. Before attaching larger models or DINO/world models, the current shape errors are used as the target for measurement. This subsequent step is an suggestion and was not included in the number of completed experiments this time.'''
(ROOT/'RESULTS_KO.md').write_text(report)

notes=[
'''[0:00–1:00] OrbitLab's question is whether the structure that handles rotation leads to better restoration and generation in data with fewer rotations. Here, we used a small synthetic world drawn with L, T, arrows, and zigzags. To put it simply, the advantages emerged in different areas. The B1, which used rotation augmentation in a standard AE, restored better at the same time, while the B2, which included a C4 structure, better matched the generation conditions. However, the generation images of B2 also show many blurriness and incorrect shapes. In the presentation, we will verify each of the following: that the structure is correct, that pixel restoration is good, and that the new conditional samples are good. We actually ran 29 AE comparisons, 6 auxiliary AEs for generation, and 8 flow runs. The 15th iteration of the original was interpreted as a list of work slots, and it does not mean that the learning lasted 15 days. [Take a moment to look at the diagram and move to the next slide]

Source: RESULTS_KO.md, configs/experiment_matrix.json, configs/generation_ae_repair.json, reports/generation_status.json''',
'''[1:00–2:00] There are four types of shapes and six types of colors. We removed the four combinations of shape and color from the learning process and left them as OOD combinations. Since the entire screen rotates 90 degrees at a time, the position also rotates. Each row of the drawing represents four rotations of the same scene. The rotations of one original scene were grouped so that they do not mix with the other split, and duplicate raster and rotation orbit hash were also removed. The total of 4,864 scenes differs based on orbit. The main learning is 1,024 scenes, verification and test are each 256 scenes, and OOD is 256 scenes. There is only one data seed and only three initialization seeds, so we do not measure the uncertainty of the overall data distribution. The standard deviation of the three seeds and the scene bootstrap CI are also different statistics. This distinction is necessary to avoid exaggerating the numbers in the following.

Source: data/manifest.json, tests/test_research.py, reports/ae_summary.json''',
'''[2:00–3:00] The vertical axis of this graph should be lower for the foreground MAE. In the main experiment, B1 and the baseline B2 have the same 3,000 steps, but the number of parameters and time differ. B1 has about 315,000 parameters, and B2 has about 118,000 parameters. If we widen B2 to about 319,000 parameters, the error decreases, but the time increases to about 55 seconds. The two bars on the right are controlled separately by time. For about 26.94 seconds, B1 learned about 5,667 steps, and B2 learned about 2,765 steps. Here, the foreground MAE was 0.0446 compared to 0.0727, so B1 was better. Even with 256 small data points, B2 did not consistently outperform. Therefore, the hypothesis that structural equivariance soon becomes a data efficiency advantage in this implementation is not supported. The raw data points and standard deviation are included in the graph of the report.

Source: reports/ae_results.csv, reports/ae_summary.json, reports/figures/fairness.png''',
'''[3:00–4:00] The C4 representation can be divided into Fourier components for your group slot. m0 is a component that does not change even when rotated. This diagram shows the input from top to bottom, the full latent reconstruction, and the reconstruction excluding m0. The results in the rows below, such as stars or clumps, show that the original direction and position have disappeared. From seed 0, the foreground MAE increased from 0.078 to 0.291. However, just by this alone, m0 cannot be said to represent meaning, while other components represent detail. The removed latent may be a distribution that the decoder has not learned. The actual latent linear shape probe was approximately 42 to 45 percent lower, and the colors were well read. The rotation control of the general AE was systematically verified using a suitable A90 during training instead of arbitrary slot shifts. The exact rotation relationship and reconstruction quality must be measured separately in this way.

Source: reports/C4_MATH_KO.md, reports/representation_results.csv, runs/primary_equivariant_n1024_s0/analysis/band_interventions.json. Mathematical background: https://proceedings.mlr.press/v48/cohenc16.html''',
'''[4:00–5:00] We inspected the validation quality standards before generating. One seed of the original B1 failed the color criteria, so both were retrained with 6,000-step processes. We did not overwrite the existing results. After fixing this AE and training the flow, we compared the same 576 noise samples using Gaussian and ODE at 16, 32, and 64 steps. The graphs show the simultaneous matching of shape and color of the combinations allowed during training. At 64 steps, B1 achieved approximately 34 percent and B2 approximately 57 percent. The right side represents the first 48 fixed samples, not the selected successful outputs. The colors are relatively accurate, but there are examples where the contours and shapes are incorrect. The evaluation tool was validated on clean synthetic shapes, but it has limitations on blurry generated images. Therefore, while relative improvements are acknowledged, we did not conclude that high-quality generation was complete. OOD, the worst case, and Gaussian images were also included in the report.

Source: reports/generation_summary.json, reports/quality_gate_repair.json, reports/state_evaluator_validation.json. Flow Matching: https://arxiv.org/abs/2210.02747''',
'''[5:00–6:00] Here are three conclusions. First, C4 symmetry matched at the level of numerical error, but B1 was more advantageous for restoration at the same time. Second, only invariant components were insufficient to fully restore images that required position and direction. Third, although the flow condition consistency was high, the generated quality was still incomplete. In pixel comparisons with recent-learning images, we did not find a complete copy, but this alone cannot eliminate memorization or distribution gaps. Coverage is also limited to the position grid that combines the conditions. This time, we preserved the original data, settings, data, code, checkpoints, CSV, environment lock, and reproducibility commands together. The priority of the next research is not model expansion, but the robustness of the generation image evaluator, repeated data seed, and a comparison of total cost combining AE and flow. We can continue narrowing the questions within the scope shown by this experiment.

Source: RESULTS_KO.md, REPRODUCE.md, reports/nn_reference.json, reports/SESSION_COMPLETION_KO.md'''
]
deck={'cover':'We have verified the accurate rotation structure.\nThe results differed when the recovery efficiency and generation conditions were consistent.',
      'conclusions':['Recovery at the same time: B1 overview MAE 0.0446, B2 0.0727','64-step generation condition match: B1 34.0%, B2 57.0%','There is still an error in the generation shape. Evaluation verification and repeated experiments are the next task.'],
      'flowSeries':[{'name':'B1 seen condition','values':[gen[f'aug/{s}/seen']['joint_correct']['mean'] for s in [0,16,32,64]],'fill':'#2166AC'},{'name':'B2 seen condition','values':[gen[f'equivariant/{s}/seen']['joint_correct']['mean'] for s in [0,16,32,64]],'fill':'#D6604D'}], 'notes':notes}
(ROOT/'reports/deck_data.json').write_text(json.dumps(deck,ensure_ascii=False,indent=2)+'\n')
(ROOT/'deliverables').mkdir(exist_ok=True)
(ROOT/'deliverables/TALK_6MIN_KO.md').write_text('# OrbitLab 6-minute presentation manuscript\n\nThis is a draft with approximately 1 minute allocated per slide. The actual reading time will be adjusted based on the presentation speed and illustration descriptions.\n'+'\n\n'.join(notes)+'\n')
print('Built report, representation CSV, generation summary, six-slide content and speaker script.')
