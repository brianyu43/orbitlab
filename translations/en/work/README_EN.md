# OrbitLab — Macro Local Expression Analysis & Generation Experiment Starter

This package is code designed to immediately start the **small C4 representation experiment and latent flow generation** in the plan. It is not a replication of the RAE / JEPA / Flow Equivariant World Models paper. It only brought the idea from the original paper, which included a symmetric structure and representation space generation, into a small control environment.

## Verification status

Checked the following on Linux CPU, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5.

- Equivariance of C4 encoder/decoder/flow, group synthesis, invariant pooling, Fourier inversion, finite backpropagation and parameter update: pass.
- AE 3 step → flow 3 step → 4-step ODE sampling → PNG/GIF → representation analysis → linear probe/learned-action fit: execution pass.
- **Did not run directly on Mac MPS.** Speed, memory, and sufficiently learned generation quality were not verified. The 3-step test images are not meaningful results, so they are not included.
- The DINOv3 selection script only performed syntax checking and did not execute weighted download/inference. Access authorization and network may be required.

The validation records are `validation_cpu_smoke.json`, `validation_cpu_doctor.json`, and `validation_e2e.json`. The 0 error in the CPU doctor is a CPU self-comparison result, not MPS validation.

## 1. Installation and equipment check

Use a new virtual environment for **arm64 Python 3.12 or later** for Apple Silicon. This package cannot check whether Python is installed.

```bash
cd orbitlab_starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > installed-versions.txt
python orbitlab.py doctor --device mps --out doctor_mps.json
python orbitlab.py smoke --device mps --out smoke_mps.json
```

`--device mps` is designed to fail if MPS is unavailable; it does not silently switch to CPU. Specify `--device cpu` only when you intentionally run a CPU check. The default `auto` selects CPU when MPS is unavailable.

Initially, maintain FP32. If an unsupported operator appears, first check only that specific operation against the CPU/MPS minimum example. Record the result of enabling CPU fallback separately and do not mix it with the MPS benchmark. Do not use environment variables that exceed memory limits.

## 2. First execution — Pipeline verification from small data

```bash
python orbitlab.py train-ae --device mps --model equivariant \
  --size 32 --channels 8 --n 256 --batch 16 --steps 200 \
  --out runs/pilot_eq
python orbitlab.py analyze --device mps --ae runs/pilot_eq/ae.pt \
  --n 64 --batch 16 --out runs/pilot_eq_analysis
```

The first line of `reconstruction.png` is input, and the second line is reconstruction. `intervention.png` is the correct order of decoding, where input / reconstruction / latent are rotated by 90 degrees, and actual input is rotated by 90 degrees. The rotation is a **scene-wide rotation centered on the image**, not an independent rotation that fixes the object center.

## 3. Comparison of the means — augmentation and structural equivariance

```bash
python orbitlab.py train-ae --device mps --model aug \
  --size 64 --channels 16 --n 1024 --batch 32 --steps 1000 \
  --seed 0 --data-seed 42 --out runs/aug_n1024_s0
python orbitlab.py train-ae --device mps --model equivariant \
  --size 64 --channels 16 --n 1024 --batch 32 --steps 1000 \
  --seed 0 --data-seed 42 --out runs/eq_n1024_s0
```

The following seed uses `1`, `2`, but `--data-seed 42` is fixed. This is to avoid mixing initialization/randomness with data composition. `plain` is a model without augmentation, and `invariant` is a contrast set that removes information that only leaves the group mean.

**The starter is not a benchmark where the number of parameters or computational cost is precisely matched.** The equivariant model evaluates the shared encoder/decoder from four directions. Please check the parameter count and time in `run.json`, and implement separate parameter-matched and wall-clock-matched comparisons in this experiment. Training saves the final checkpoint for the specified number of steps. Best-validation checkpoint selection and automatic multiple seed aggregation are not implemented.

## 4. Expressional analysis

```bash
python orbitlab.py analyze --device mps --ae runs/eq_n1024_s0/ae.pt \
  --n 256 --batch 32 --out runs/eq_n1024_s0_analysis
python -m pip install scikit-learn
python probe_latents.py --input runs/eq_n1024_s0_analysis \
  --out runs/eq_n1024_s0_analysis/probes.json
```

`analyze` stores the latent for four rotations in train/val/test/ood respectively. `probe_latents.py` only learns the probe and 90 degree linear operation for train, selects regularization for val, and then applies it once to test/ood.

`fixed_rho_equivariance_relative_mse` only has its meaning guaranteed by design in the **C4 model**. plain/aug cannot be concluded that this value is bad solely because the potential coordinate system is arbitrary. Please also compare these models with the learned A90 residuals and closure from `probe_latents.py`.

Fourier band energy is the DFT along the group axis. m=0 is a rotation-invariant component, m=1/3 is a component that changes sign upon 90° rotation, and m=2 is a component that changes sign upon 90° rotation. **This does not automatically mean disentanglement by shape/color/position.** Since the bottleneck capacity of the invariant contrast group is also different, this is not a primary competition model but an information removal experiment.

## 5. Create a new image from noise

```bash
python orbitlab.py train-flow --device mps --ae runs/eq_n1024_s0/ae.pt \
  --n 1024 --batch 128 --steps 2000 --out runs/eq_flow_s0
python orbitlab.py sample --device mps --flow runs/eq_flow_s0/flow.pt \
  --n 48 --ode-steps 64 --out runs/eq_flow_samples_s0
```

`new_samples.png` is an image obtained by integrating latent flow from Gaussian noise, not by simple rotation/interpolation of the learning image. `latent_rotation_*.gif` is a control experiment where the latent is transformed into C4. It does not mean that a GIF rotating will be as good as the original generation quality.

The conditions are 4 shapes / 6 colors, and these conditions labels are provided for **flow learning**. AE only learns reconstruction. You should not claim that you discovered concepts without labels. The diagonal factor pair `(0,0), (1,1), (2,2), (3,3)` is an OOD test for generation and analysis and does not guarantee success.

The conditions for the generation evaluation, including matching rate, nearest neighbor, distribution coverage, seed confidence interval, and 16/32/64 ODE-step comparison, were specified in the plan, but they were not fully automated in this starter. Do not select only a subset of the sample and present the results.

## 6. Choice: Observation of expression of DINOv3

```bash
python -m pip install 'transformers>=4.56' torchvision
# If needed, first accept the terms and obtain access on the official model page.
python extract_dino.py --images ./my_images --device mps \
  --limit 200 --out runs/dino_features.npz
```

The base model is `facebook/dinov3-vits16-pretrain-lvd1689m`. The model weights are downloaded, but the user's images are not uploaded externally. CLS and patch averages are stored per layer, and the register token is excluded from the patch average. It is only extracted if the basic layers 3/6/9/12 exist. Changes to model/config/processor should be treated as separate experiments.

## Included / Not included

Included: generator, 4 AE conditions, group Fourier baseline diagnosis, train/val/test/ood split, latent intervention, conditional flow, PNG/GIF, linear probes, learned A90, checkpoint and environmental records, MPS forward/backward propagation parity test.

Not included: formal JEPA training, dynamics/planning, real RAE replication, proof of meaningful disentanglement of learned latent, all generation quality evaluation, raster duplicate hash removal between train/test, parameter/compute matched experiment automation, Streamlit dashboard. These are follow-up tasks on the plan.

All outputs require a new empty path, so they do not overwrite existing experiments. Use only the local checkpoint created by the package in `torch.load`.
