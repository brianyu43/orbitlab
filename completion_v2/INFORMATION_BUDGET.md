# What each system is given

These are separate diagnostic systems, not one jointly trained end-to-end world model. Generation/perception use polygons; dynamics uses moving disks; SVIB is an external image-to-image task. Their scores cannot be combined into one model accuracy.

| Study | Inference input | Learned part | Built-in or reused information | What the result does not establish |
|---|---|---|---|---|
| Generation | Structured shape/color condition and Gaussian noise | C4 autoencoder, latent flow, decoder | Fixed discrete C4 action, procedural training world, automatic evaluator | Natural-language understanding, arbitrary images, continuous 3D equivariance |
| Perception/edit | RGB image and click/structured edit command | Supervised CNN reads shape, pose, radius, center | Known six-color palette; distinct source-object colors; existing analytic/learned controller; known renderer | Unsupervised object discovery or a newly learned full rendering/command pipeline |
| Detail follow-up | Same RGB/click interface | Binary or intensity crop CNN with full pose labels | Same priors; rasterization details retained by intensity | Abstract orientation understanding independent of renderer |
| Dynamics oracle | True current object state, prescribed action and context | State-transition network | State/radius representation and supplied force context; bounded arms additionally know wall limits and training-derived speed limit | Visual perception or discovery of latent external forces |
| Dynamics RGB measurement | First four RGB frames and prescribed action | Same transition network | Hand-designed known-palette/circle/physics measurement and context inference | End-to-end learned video representation |
| Dynamics force-wall | Same state/context/action as corresponding input arm | None | Analytic force/drag integration and reflecting walls; simulator time discretization; no disk-disk interaction | Learned physical law or equal-prior neural baseline |
| SVIB | RGB source image | Plain or C4 RGB residual predictor | Paired supervised images; fixed architecture and pixel loss | Full benchmark suite performance, exact attribute manipulation success, converged training |

Perception truth masks/states and dynamics future states are training/scoring targets, not inference inputs, except the explicitly named dynamics oracle condition. SVIB metadata are used for data auditing, not as model inputs. Geometric-equivalence pose scores are supplementary and never replace strict scores.
