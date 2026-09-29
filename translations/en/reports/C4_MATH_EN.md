# A page explaining the C4 expression

A 90-degree rotation R is invariant when applied four times. The slot rotation rho is also the same, with rho⁴ = I. The original definition of each slot k is E_k(x) = h(R^(-k)x). After rotating the image by r times, E_k(R^r x) = h(R^(r-k)x) = E_(k-r)(x), so E(R^r x) = rho_r E(x).

The decoder is D(z) = (1/4) Σ_k R^k d(z_k). If the slot of z is changed by r, we can substitute k-r = j, so D(rho_r z) = (1/4) Σ_j R^(j+r)d(z_j) = R^r, which is D(z). It is a structural relationship independent of whether the learning is performed or not. The differences in actual floating-point implementation are verified by numerical testing.

The normalized DFT is zhat_m = (1/2) Σ_k z_k exp(-2πimk/4). The shift of slot r multiplies the m component by exp(-2πimr/4). m=0 is invariant, m=2 reverses the sign by 90 degrees, and m=1/3 is a conjugate pair for real inputs. To restore real values, m=1/3 must be removed together or scaled. This decomposition is a decomposition of group actions, and the meaningful separation of color, shape, and position is not automatically demonstrated.

If only the invariant average is left, the encode will become the same in rotation. If you provide this code to the accurate affine decoder, the decode will also become an invariant image for all C4 rotations, and cannot accurately reconstruct a one-sided object. The restoration loss in B3 can occur even without implementation errors.

There is no reason to enforce the slot rho on arbitrary coordinates of general AE. A90 learned in training is used to test A90 z≈z_rot, A90⁴ z≈z, and D(A90 z)≈R x in independent scenes. The linear A after feature standardization is interpreted as an affine operation that includes inverse standardization in the original latent space.

In Flow, vG(z,t,c) is defined as (1/4)Σ_g rho_g^-1 v(rho_g z,t,c). Since c (shape and color) is invariant under rotation, velocity is equivariant. If the same channel normalization is used in all group slots and coupled noise is also converted to rho, the Euler ODE trajectory exchanges rho with it. This property does not guarantee the shape and color accuracy or diversity of the generated samples.

Verification evidence: `original_smoke_mps.json`, `research_unit_checks.json`. The range is the C4 rotation of the entire scene centered on the image.
