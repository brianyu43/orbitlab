# Limits of restoring expressions that have lost direction information

Let's say your rotated input is x_g, the average is m, and the output is p that is common to all inputs.

`mean_g ||x_g-p||² = mean_g ||x_g-m||² + ||m-p||²`

If x_g-p is divided by (x_g-m)+(m-p) and squared, the first term's rotation average is zero, so the cross term disappears. Since the second squared term is not negative, the first term is the possible lower bound of the loss.

In simple terms, if you compress different-direction pictures into the same code, the decoder cannot distinguish which direction to choose. Even if you output the average picture, the remaining difference is an unavoidable error.

This is the known minimum-square-error identity. It has been verified numerically with the stored B3 model and the new 512 scenes, and the results apply only to the general pixel MSE.
