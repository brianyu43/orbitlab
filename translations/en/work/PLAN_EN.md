# OrbitLab — Macro Local Expression Analysis & Creation Experiment Plan

Submission criteria: 2026-09-23

## OrbitLab

Expression analysis · control · generation experiments starting from Mac

> Key Question | When a statement that preserves rotational information even while recognizing that it is the same object is used, will it recover data better and generate it more consistently when there is less data?

Date of writing: September 23, 2026. Designed based on the M5 Pro · 64GB unified memory Mac you previously provided. Since you did not directly connect to the device at this time, actual environment and processing speed will be measured during the first execution.

### Recommended choices

Complete a small experiment in a single storage. The main experiment consists of C4 symmetric representation → Fourier component analysis → latent variable intervention → conditional latent flow generation. DINOv3 observations are placed in the selection module, and the world model is treated as an extension module after obtaining the key results.

### Why is this topic?

It connects recent areas of interest such as geometric deep learning, invariance and variable representation, and the representation of world models within an experiment. At the same time, it allows you to visually inspect rotation and generative scenes, and to explain how certain assumptions produce certain results numerically. Instead of continuously expanding the topic, it is a choice to complete one experiment to the end.

| Item | Basic plan |
| --- | --- |
| Scope | 64×64 synthesized image / 64-dimensional latent representation / small CNN·MLP |
| Duration | 15 sessions × 90–120 minutes of work slots. Actual learning waiting time is measured separately |
| Resources | Local CPU·MPS. Core experiments are model·data download and paid API unnecessary |
| Final Output | Reproducible code, comparison table, latent intervention GIF, new generated sample, 6 minute presentation |
| Excluded | Pre-learning of large image and video models, 4K video generation, actual protein design and experimental verification |

### The status of this document

It provides a research plan and a working starter. It is not a document that already has obtained scientific results. Structural checks and micro-execution passed on Linux CPU, but the performance of Mac MPS and the quality of the sufficiently learned model have not yet been measured. All target values are pre-defined judgment criteria, not observation results.

Reading order: 2nd page research flow → 3–7th page experimental design → 8–9th page implementation/schedule → 10th page extension → 11th page package scope → 12th page source.

## 01 | Recent research flow

Choose research questions you can bring instead of the latest model name

Below is the relevant flow of the original paper and official implementation as confirmed by September 23, 2026. It does not mean a complete list of all fields or that it represents the “single most up-to-date method.”

| Research·Period | Important changes | What can be taken from Mac |
| --- | --- | --- |
| DINOv3 · 2025-08 Public | Reuses global and patch-level features while fixing the learned visual representation. ViT-S/16 has 21M parameters. [S1,S2] | Extracts layer-by-layer features from small models and investigates what is preserved during rotation and position changes |
| RAE · 2025-10 / T2I extension 2026-01 | Combine the expression encoder and decoder and generate in their latent space. T2I subsequent research experimented with the 0.5B–9.8B range. [S3,S4] | Do not replicate the large DiT, fix the small encoder, and only train the small flow |
| V-JEPA 2.1 · 2026-03 | Enhances dense features with spatial structure and temporal consistency in video and images. Public ViT-B is 80M. [S5] | Instead of training large video models, slightly structure the feature prediction problem in 2D videos |
| Flow Equivariant World Models · 2026-01 | Structures memory and long-term rollout by modeling self-motion and object motion with Lie-group flow. [S6] | Controlled experiment to control the benefits of consistency and long-term prediction in known small symmetry groups |
| LeJEPA · 2025-11 | Proposes a self-supervised learning principle that more explicitly addresses expression distribution and collapse prevention. [S8] | Adds decay diagnosis metrics such as dispersion and effective rank, instead of just looking at the average loss |

### Important distinction

C4 group lifting or Fourier decomposition themselves are not new 2026 technologies. This project is a reduction experiment that connects an old precise mathematical structure with the recent question of “using expressions together with understanding and generation.” The theoretical novelty is not that we used the structure, but that we must obtain it by controlling when it helps and when it fails. [S3,S4,S9]

The “flow” in FEWM refers to the flow of the transformation group over time. Here, the flow matching used for generation learns a vector field that moves from the noise distribution to the data distribution. It may be related, but it is not the same concept or the same algorithm in the same paper. [S6,S10]

## 02 | Questions and hypotheses

Separate to view · preserve · manipulate · create

### If expressed in everyday words

“You must know that the arrow pointing to the right and the arrow pointing up are of the same type. However, when you recreate the drawing, you must also know which direction it was pointing.” The experiment investigates how invariance, which is good for classification, and information preservation necessary for reconfiguration coexist.

### Minimum expressions that connect expression learning and mathematical expression theory

```text
E(R_k x) = rho_k E(x)
D(rho_k z) = R_k D(z)
rho_a rho_b = rho_(a+b mod 4)
```

E is the image encoder, D is the decoder, and R_k is the 90k-degree rotation of the entire image, while rho_k is the transformation that acts on the latent space. The invariant representation holds when rho_k is an identity transformation. This relationship itself is a fundamental concept of group equivariance, and this experiment is precisely tested using the finite group C4. [S9]

| Hypothesis | Verification method | Unacceptable conclusion |
| --- | --- | --- |
| H1 data efficiency | Compare the test reconstruction and intervention errors of augmentation AE and equivariant AE in the same base-scene number | Since equivariance is accurate, it is better in all problems |
| H2 information preservation | Compare the restoration and direction probe of expressions that only leave out the entire C4 expression and the group mean | The invariant component is soon "meaning", and the rest is soon "form" |
| H3 generation and control | Learn flow from fixed AE representations to evaluate new samples, condition matching, rotation control, and nearest neighbors | Since the GIF rotates well, new image generation also succeeded |
| H4 calculation cost | Separate and compare the same step experiment and the same wall-clock experiment | Since the weight is shared, the calculation amount is also the same |

### Definition of success

Regardless of which model wins, achieving project goals involves distinguishing data, computational volume, and expression structure and reproducing the results. Even results contrary to prior expectations are research results that can test loss, capacity, data bias, and domain shift.

### Information provided explicitly

Rotation groups and rotation transformations are determined by the human. In the basic AE, factor labels are not provided, but the reconstruction goal is given. In the generation flow, shape and color conditions are provided. Therefore, the claim that “AI discovered groups or concepts without any assumptions” is not made.

## 03 | Data and division

First create a small world that knows the answer

| Design elements | Specification |
| --- | --- |
| Observation | 64×64 RGB, black background, one shape. pilot only 32×32 |
| Shapes | L / T / Arrow / zigzag 4 types, 6 colors, continuous position and size |
| Symmetric group | C4 = {0°,90°,180°,270°}. Use rot90 after creating canonical raster |
| Basic size | Main experiment base scenes 1,024. Additional data efficiency experiment 256 / 4,096 |
| Split unit | Keep all rotations of the original scene and the same original in one bundle |
| Combination OOD | (shape,color)=(0,0),(1,1),(2,2),(3,3) are excluded from learning |
| Equal distribution evaluation | Generate independent val/test scenes with the remaining 20 combinations allowed for learning |

### Leak prevention rules

train/val/test uses a separate RNG namespace and base_id. augmentation is applied only after the split. Before the final report, it checks for raster duplicates using the canonical image and rotation orbit hashes. The starter implements seed and factor-pair splitting, but raster hash deduplication is a later implementation.

Rotation holdout and combination holdout are different tests. The structural model is guaranteed to handle a 90-degree rotation as per design, but creating a shape and color combination that is not observed is a separate generalization problem. The two results are not combined and are not referred to as "OOD success."

### Not claiming asymmetry incorrectly

Rotation is a transformation of the entire scene centered on the screen. It does not treat arbitrary translations, crops, or resizes that go outside the screen as part of the same exact group. 45 degrees is a transformation outside of C4, and it does not expect exact equivariance from models that do not perform a separate SO(2) expansion.

A figure can have 180-degree symmetry with respect to itself, such as in a zigzag. When evaluating "shape direction," the direction of congruence corresponding to that symmetry is allowed. The C4 orbit, which includes the position of the entire screen, and the pose of the object itself are distinguished.

### Metadata to be stored

base_id, split, shape, color, center coordinates, size, rotation, data_seed, code version, and canonical hash are stored. When using real photos in the selection module, only those with public permission or self-taken data are used, and photos that reveal personal information are excluded.

## 04 | Model and expressive theory analysis

Check the exact small structure directly

| Model | Composition | Role |
| --- | --- | --- |
| B0 Plain AE | General CNN → 64D → decoder, basic direction learning | Diagnosis contrast group without augmentation |
| B1 Augmented AE | Same structure as B0, C4 augmentation | Baseline comparison |
| B2 C4 AE | Apply shared encoder four times → 4×16D; equivariant decoder | Main experimental model |
| B3 Invariant-only | Average and replicate the four group slots of B2 | Direction information removal contrast group; not a main performance competition model |

### Implementation starting without external group-convolution library

```text
z_k = h(R_(-k) x),  k = 0,1,2,3
(rho_r z)_k = z_(k-r mod 4)
D(z) = (1/4) sum_k R_k d(z_k)
```

This implementation is a precise C4 configuration that evaluates the shared encoder h and decoder d from multiple directions. It does not reproduce the entire architecture of the group-convolution paper. Even with shared weights, the computation includes four-directional evaluation, so both the number of parameters and the actual time are recorded. [S9]

### Decompose the group axis with DFT

```text
z_hat_m = (1/2) sum_(k=0..3) z_k exp(-2*pi*i*m*k/4)
```

m=0 is a component that does not rotate. m=1 and m=3 are conjugate pairs in real input, and when grouped by real coordinates, they become 2D rotation components. m=2 changes sign after a single 90-degree rotation. DFT is performed on the CPU to avoid dependence on MPS complex operations in the first experiment.

### Three operations

First, shift the group slot by one unit using rho_1 to decode and compare it with the actual 90-degree rotation ground truth. Second, leave only m=0 and check which information is lost. Third, measure shape/color/direction probes and energy for each band. Removing bands can create latent variables outside the learning distribution, so we distinguish between decoder artifacts and the effect of information removal.

General AE does not have a fixed rho in its potential coordinate system. For B0/B1, A90 is fitted by ridge regression from the train and normalized to val, then test residuals and A90^4 ≈ I are evaluated. General AE does not conclude that it is inferior solely based on the error introduced by applying a fixed slot shift.

## 05 |  Creation experiment

Does not confuse restoration · intervention · creation of new samples

### Experiment G1: Intervention of the potential variable

Encode the input x and apply rho_k, then decode. Compare the results with the actual R_k x. This experiment is a controllable re-generation, and it is not an experiment that generates a new scene from noise.

### Experiment G2: Conditional flow learning in fixed representations

After completing the AE training, both the encoder and decoder are fixed. The training scene is converted into latent z1. z0 is Gaussian noise, and c is the shape and color condition. A small MLP learns the vector field along the straight path below, and during generation, the ODE is integrated to create samples corresponding to z1. This is a simplified design of the flow matching principle. [S10]

```text
z_t = (1-t) z0 + t z1
L_flow = E ||v_theta(z_t,t,c) - (z1-z0)||^2
x_new = D(ODEsolve(v_theta, z0, c))
```

In the C4 model, the vector field is symmetrized by v_G(z,t,c) = (1/4) sum_g rho_g^-1 v(rho_g z,t,c). Condition c only includes shapes and colors that are invariant under rotation. If the position and direction conditions are added, the condition must also be redesigned to be transformed together.

### Attention to normalizing potential variables

Applying different mean and standard deviations for each group slot can break the original simple rho effect. The starter averages the group axis and batch axis together and uses channel-specific common normalization. Even in the extension that uses pretrained DINO features, normalization and decoder training are separately verified.

### Essential comparison and judgment

G2 compares “directly feeding Gaussian latent into the decoder” and “learned flow”. It separates ODE 16/32/64 steps and evaluates more than 512 samples from the same generation seed list. Outputs that are almost copies of the GIF that was rotated only four times, outputs that almost copy the learned samples, or outputs that only produce stable backgrounds are not considered successful.

While it is related to the core idea of RAE, it is not a formal RAE replication. The encoder of this experiment is trained for reconstruction on small synthetic data. RAE is designed using a pre-trained representation encoder, and the difference is explicitly stated in the report title and methodology. [S3,S4]

## 06 | Measurement and conclusion rules

Separate mathematical assurance and experiential performance

| Measurement | What it means | Attention |
| --- | --- | --- |
| E(Rx)−rho E(x) | Is the structure implementation correct | In the accurate B2, it is a correctness test, not a new discovery |
| Reconstruction MSE / PSNR / foreground MAE | How much location, color, and shape information is preserved | Do not trust only the low overall MSE of data with many background pixels |
| Difference between D(rho E(x)) and Rx | Does the correct scene recover after intervention | Even if only D(rho z)=R D(z) is correct, the original may be blurred |
| linear probe / effective rank | What information is read and what latent has decayed | The expression that is causally separated because the probe reads information is not |
| Condition match / centroid / occupancy | Does the new generated sample satisfy the conditions and shape required | Validate the evaluator itself in an independent holdout |
| NN distance / coverage / diversity | Does the learning sample have memorization·mode collapse | Does the high diversity of low-quality noise not succeed in being measured |

### Minimum experiment table

The confirmed comparison is B1/B2 × n=1,024 × initialization seed 0/1/2, totaling 6 times. B0/B3 are first diagnosed individually. The additional experiment with n=256/4,096 only checks the direction with seed 0, and if a meaningful difference is observed, the seed is increased. The exploratory results and confirmatory results are separated in the table.

### Comparison fairness

It aligns the same base scenes, input resolution, data split, latent full dimension, optimizer type, and validation usage rules. It separates the parameter-matched and wall-clock-matched results. If only the same step results are present, it is precisely referred to as a comparison under that condition. The starter parameter count does not automatically equalize.

### Statistics and termination conditions

Display the material points of the three seeds and report the mean ± standard deviation. Bootstrap is performed on the original scene unit rather than the rotation image unit. In the generation process, the base noise seed is fixed. If the reconstruction is not properly performed in 1,024 scenes, it does not proceed to the flow stage.

Self-entry criterion: The held-out reconstruction must be improved compared to the all-black baseline, and the foreground location and shape must be preserved both visually and numerically. As a preliminary operational goal, shape/color probe ≥90% and foreground MAE ≤0.10 are acceptable, but they will be fixed before the main experiment after calibration depending on the data difficulty. This is not the observed performance.

FID alone does not claim the quality of synthesized figures. Since the standard natural image features and small synthetic distribution may not match, we use the correct factor-based indicators as the primary evaluation metric.

## 07 | Mac resources and environment

Instead of writing all the large specifications, create repeatable small experiments.

The hardware configuration is M5 Pro / 64GB unified memory. Currently, there is no available RAM, storage space, or macOS/Python version confirmed. PyTorch provides an MPS backend, but this does not mean that CUDA code automatically runs on the same path. [S11,S12]

| Area | Start-up settings·Operating budget |
| --- | --- |
| Main model | 64×64, latent 4×16, trainable parameters limited to less than millions |
| batch | From AE 16 or 32, from flow 128. Change after actual measurement |
| Memory | Operating budget that prioritizes the internal 8GB main process; not real-time value |
| Storage space | Aim for 10–20GB for the entire project. Recommended to reserve a first 20GB buffer |
| dtype | Starting from FP32 correctness criteria; mixed precision is done after ablation |
| Data memory | 4,096 × 3 × 64 × 64 × FP32 = approx. 0.188 GiB; activation·runtime are separate |
| Avoid | CUDA/Triton/xFormers dependent code, unlimited cache of all patches of all layers |

### Installation

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip freeze > installed-versions.txt
python orbitlab.py doctor --device mps --out doctor_mps.json
python orbitlab.py smoke --device mps --out smoke_mps.json
```

Do not upgrade the existing environment in bulk; use a new arm64 environment. If Python is not installed, you must first prepare the official Python distribution. The starter's auto mode can be switched to CPU, so for performance measurement, specify --device mps.

### How to estimate time

After the warm-up of 20 steps, measure 100 steps three times and record the median. Synchronize the asynchronous queue using torch.mps.synchronize() before and after measurement. Total time budget = runs × steps × median seconds/step + evaluation and storage time. This document does not include the number of images per minute or the “how many minutes to finish” values that were not measured on Mac.

Do not release the memory limit. When an unsupported operator occurs, locate the position using the CPU/MPS minimum example. The execution that is turned on with fallback is recorded separately and not mixed with MPS-only benchmarks. As memory pressure increases, it is reduced by batch, resolution, and cache order. [S12]

## 08 | 15th episode execution schedule

Don't increase your exploration, leave one piece of evidence for each episode.

Each session is a planning slot for 90–120 minutes of focused work. It is adjusted based on implementation experience and equipment measurements, and does not guarantee the completion of learning time. The design ensures that at least a minimum result remains even if only 1–4 sessions are completed.

| Episode | Execution | Completed Output |
| --- | --- | --- |
| 1 | Environmental·MPS forward/reverse transmission parity and smoke test | doctor_mps.json / smoke_mps.json |
| 2 | Check synthetic data and split, view the same original rotation | Data description and 32 input grid |
| 3 | 32×32 small AE pilot, time and memory measurement | First reconstruction + step time |
| 4 | C4 encoder/decoder·DFT·group synthesis explanation | 1st equation explained by my hand and unit tests |
| 5–6 | B1/B2 n=1,024 execution, foreground restoration test | First baseline comparison and failure cases |
| 7 | invariant-only removal experiment and Fourier band probe | band energy / information removal picture |
| 8 | A90 suitable, val selection, test·OOD analysis | linear probe / closure / residual table |
| 9–10 | Weekly seed comparison 1/2 repetition, calculation amount·parameter record | seed comparison table |
| 11 | encoder/decoder fixed, conditional flow pilot | flow loss and Gaussian baseline |
| 12 | Create new sample, 16/32/64-step comparison | Create same seed grid / latent GIF |
| 13 | Condition matching·NN·coverage evaluation and error analysis | Evaluation sheet that presents representative success·failure together |
| 14 | Memory·time·split leakage·conclusion audit | Final metrics and verification checklist |
| 15 | README·6 minutes presentation·reproduce command summary | Other people can run the release |

### What to do today right away

```text
python orbitlab.py train-ae --device mps --model equivariant \
  --size 32 --channels 8 --n 256 --batch 16 --steps 200 \
  --out runs/pilot_eq
```

If you only target one file, it will be runs/pilot_eq/reconstruction.png. Even if the first output is blurry or only shows the background, it is not a “experiment failure” but the starting point of the diagnosis. Before adding a generation model or a new paper, first verify that the data, loss, shape, and gradient are correct.

### The central sentence of the presentation

“We separated and compared the ability to recognize rotation, the ability to not lose rotation information, and the ability to control and generate that information.” If there are no numbers yet, this sentence should be written as a target sentence and not as a performance sentence.

## 09 | Selection extension

After getting the key results, choose only one.

### Extension A — DINOv3 Expression microscope

Fix the official ViT-S/16 21M model and observe the original, 90°, 180°, and 270° rotations on 200–500 images. Compare the CLS and patch mean from layers 3/6/9/12. Dense patches are compared after removing the register token and aligning the grid spatial rotations. Verify the public implementation and the model's access conditions. [S1,S2]

The question is “What are pre-trained representations insensitive to and what preserve?” Synthetic figures and natural photos of self-recorded individuals are distributed separately, and domain shift is indicated. Do not conclude “they do not recognize rotation” solely based on low cosine similarity; instead, consider both learned A90 and aligned patch correspondence.

The basic package extract_dino.py stores the CLS/patch mean. Dense correspondence heatmap, layerwise CKA, and the generated decoder have not yet been implemented. Even if DINO access approval is delayed, the C4 experiments will proceed as planned.

### Extension B — Small dynamical models and counterfactual futures

Expand the generator to allow 1–2 objects to move within a 64×64 scene. The first version maintains C4 symmetry using a square boundary and frictionless, isotropic friction. Speed is inferred from at least two frames, and we compare what future outcomes arise when we change the action. DINO-WM is a related prior research approach that predicts fixed visual features. [S7]

```text
z_(t+1) = F(z_t, z_(t-1), a_t)
F(rho z_t, rho z_(t-1), R a_t) ≈ rho F(z_t,z_(t-1),a_t)
```

Compare the constant-velocity baseline, non-structural latent predictor, and C4 predictor. Evaluate the train 16-step / test 32.64-step rollout, centroid error, collision timing, and the reappearance location behind occlusion. Divide the trajectory into segments and clearly distinguish the video rendered with the known renderer from the video generated by the neural decoder.

If the direction of gravity is fixed, rotational symmetry is broken. The gravitational vector must also be treated as an input and rotate together, or the symmetry group must be reduced. We do not assume that matching the speed and hidden states that cannot be observed in a single image can be solved simply by “expressing them well.” Uncertain futures are evaluated separately using point predictions and conditional distributions.

### Extensions that will not be done in this session

Do not simultaneously add protein/molecular, LLM SAE, NeRF/3D Gaussian, and robot control. Each requires its own separate data, symmetry, and evaluation criteria. The value of the first project is determined by the depth of verification for a single question, rather than the number of application fields.

## 10 | Execution package and verification range

Distinguish between the immediately executable part and the subsequent implementation.

| File | Purpose |
| --- | --- |
| orbitlab.py | doctor / smoke / train-ae / analyze / train-flow / sample |
| probe_latents.py | train Select suitable/val/test/ood evaluation, linear probes and learned A90 |
| extract_dino.py | Select DINOv3 feature extraction. Separate transformers·torchvision required |
| README_KO.md | Installation·Correct execution command·Check points in case of failure |
| IMPLEMENTATION_PROMPT_KO.md | Implementation requirements for the next coding session |
| validation_*.json | CPU smoke·doctor·micro-sized complete path execution log |

### What I checked

On Linux CPU / Python 3.13.5 / torch 2.10.0+cpu, C4 encoder·decoder·flow, GAN synthesis, invariant pooling, DFT bidirectional, backpropagation·parameter update were verified. The encoder max absolute error was 0, and the decoder obtained a smoke result of approximately 5.96×10^-8. This is a verification of the structure implementation and not learned performance.

We also executed the AE 3 step → flow 3 step → 4-step ODE sampling → PNG/GIF → representation metrics → linear probe/learned-action fit path. Since we did not learn enough to evaluate the quality, we do not provide the resulting image as an exemplary case.

### Things that you have not checked yet

Mac MPS execution speed, memory, quality after sufficient learning, all generation evaluation metrics, and final multiple seed statistics are not confirmed. The DINO script only performed syntax checking and did not apply weights or execute inference. The CPU doctor error of 0 is a CPU self-comparison, so it should not be read as MPS verification.

### Additional features to add before the final research report

Remove raster/orbit hash duplicates, set parameter/compute-matched, per-sample CSV and base-scene bootstrap, condition accuracy, centroid, coverage, NN evaluator, best-validation checkpoint rules, or explicit fixed-step comparison, complete lockfile and code commit records per execution. Insert this sequence into the subsequent implementation prompt.

### The result table is not filled out now

The final table columns are model / data_n / seed / parameters / train_seconds / reconstruction / foreground_MAE / learned_action_error / condition_accuracy / coverage. Any cells that have not yet been measured are left as unmeasured. The target values in the document or facts derived from the architecture are not used as performance metrics.

> Completion criteria | A state where another person creates data with the same command in a new environment, obtains results from the same partition, and can verify the difference between the description and the numbers.

## 11 | Source and reading order

Original paper · Official implementation focus / Confirmation date 2026-09-23

The sources below are the basis for the research flow and implementation interface. The experimental size, schedule, entry criteria, and model comparison table are the designs proposed in this plan, and they are not directly copied from the experimental results of the original paper.

[S1] Meta FAIR. DINOv3 official storage. Publiced on 2025-08-14; model size, weight access, license, and 2026 update.
https://github.com/facebookresearch/dinov3

[S2] Hugging Face Transformers. DINOv3 document. CLS/register/patch distinction and example of using official models.
https://huggingface.co/docs/transformers/model_doc/dinov3

[S3] Zheng et al. Diffusion Transformers with Representation Autoencoders. 2025-10.
https://arxiv.org/abs/2510.11690

[S4] Tong et al. Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders. 2026-01-22.
https://arxiv.org/abs/2601.16208

[S5] Mur-Labadia et al. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. 2026-03-15 Submission, official implementation 03-16 Public release.
https://arxiv.org/abs/2603.14482
https://github.com/facebookresearch/vjepa2

[S6] Lillemark et al. Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments. 2026-01-03.
https://arxiv.org/abs/2601.01075

[S7] Zhou et al. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. Revision 2024-11 / 2025-02.
https://arxiv.org/abs/2411.04983

[S8] Balestriero & LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 2025-11.
https://arxiv.org/abs/2511.08544

[S9] Cohen & Welling. Group Equivariant Convolutional Networks. ICML 2016.
https://arxiv.org/abs/1602.07576

[S10] Lipman et al. Flow Matching for Generative Modeling. 2022 / ICLR 2023.
https://arxiv.org/abs/2210.02747

[S11] PyTorch. MPS backend documentation.
https://docs.pytorch.org/docs/2.14/notes/mps.html

[S12] PyTorch. MPS Environment Variables.
https://docs.pytorch.org/docs/2.14/mps_environment_variables.html

Recommended reading: Read only the equivariance definition of S9 before the first experiment. After reconstruction, read the path and vector field definitions of S10 before distinguishing S3's encoder/decoder and implementing the flow. S6, S7, and S8 should be read when extending, and implementation should not be postponed until all are read.
