# Example where there is no environmental information, the answer is not determined in one way

We kept the exact state and behavior of the two objects identical at the moment and only changed the environmental values hidden in the model. The inputs that blocked force or resistance were the same, but the next correct answer was different. The object sizes, speeds, and environmental values for the three examples were selected within the learning range of variable_force. However, as a configuration example where the same current state is assigned to different environments, it does not represent the frequency of random data.

| Hidden Information | Maximum difference at the next location (px) | Maximum difference at the next speed (px/frame) | Minimum normalized MSE for both cases |
| --- | ---: | ---: | ---: |
| Total | 0.02808520 | 0.04990013 | 2.011615406e-05 |
| Resistance | 0.00669555 | 0.01190450 | 5.370865912e-07 |
| Combined and resistance | 0.02963225 | 0.05263007 | 1.713559457e-05 |

outputs the same value. If the two correct answers are a and b, the squared error calculated with equal weights for both cases is the sum of “the squared distance between the prediction and the median (a+b)/2” and “one-fourth of the squared distance between the two correct answers”. Therefore, even if the minimum value is obtained from the median, an error remains. The table was calculated using the position/31.5 and speed/3 after normalization as the mean of the object and coordinate values.  The

simulator's one-step results were individually tested using closed-form free-motion equations at 8 time points, and wall and object collisions were confirmed to occur. The squared error decomposition was also tested for 32 predictions that differed from the median. This serves as a proof-of-concept test for the three configuration cases and does not indicate the Bayes error, long-term prediction error lower bound, or achievement of learning performance for any arbitrary set of actual evaluation data.

If we provide all environmental information, the two inputs will differ and this ambiguity will disappear. However, since the learned model may be wrong, we evaluate the information shortage and model learning failure separately.
