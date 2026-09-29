# If a different future is possible with the same past video

2026-09-23. This is a boundary case configured within the simulator, and it is not the result of measuring the frequency of such cases among actual evaluation data.

When two objects move at the same constant speed v and the external resultant force a follows the relationship a = gamma * v with the resistance coefficient gamma, the forces and resistances cancel each other out. For each substep, the speed update is given by v*exp(-gamma*h) + a*(1-exp(-gamma*h))/gamma = v, so even if gamma is different, the past position and speed may still be the same.

Two environments were created with gamma = 0.004 and 0.009. The gravitational, wind, and resistance values for both environments are within the original variable_force range. The initial state and the four RGB inputs and input behaviors are identical, but when the first object is given the same velocity change k, the subsequent velocity becomes v + k*exp(-gamma), resulting in different values.

Therefore, there is no deterministic video prediction model that can definitively identify which of these two scenarios without additional information. Since the input is the same, the output is also the same. If we treat both cases with the same probability and evaluate the average squared error of the next x/y/vx/vy coordinates of the object, the minimum average error of the common output is the value obtained by dividing the square root of the average of the two difference squares by 4. This conditional lower bound for the two cases does not extend the lower bound of the entire dataset to the overall error bound.

If the speeds of several objects are sufficiently different or if the speed changes during observation, information can be generated that distinguishes resistance and force. This is not an assertion that all general non-contact trajectories are indistinguishable. It is an example used to distinguish boundaries that contain no information in the input, and cases where the model fails to use information properly, even if past videos are perfect.

The actual two trajectories and inputs are in counterexample.npz, the numerical values and verification are in verification.json, and the same four past images are in identical_past.png. They do not replace the observation length and independent repetition entire evaluation of C07.
