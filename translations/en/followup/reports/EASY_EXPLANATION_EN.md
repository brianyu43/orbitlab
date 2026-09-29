# What did you do at OrbitLab and what did you learn?

The goal was not just to “make the drawing a little more realistic,” but to **determine whether we can identify the desired object, change it, and predict up to the next movement**. Right now, we are not yet fully completed that ability; we are in a state of distinguishing which stage is blocked through experiments.

For example, let's consider the order "make a red arrow." If the color is correct, it appears more successful than the actual one when measured. Therefore, we separately checked whether the shape and color were correct, as well as whether the outline formed a distinct geometric shape. A model with added rotation rules was better, but only about 16 out of 100 familiar orders passed a strict automatic criterion. This does not mean that all 16 images were judged as good.

Next, we tested the instruction "only change and rotate this object blue." The model correctly selected the target but incorrectly interpreted the object's shape and direction. Even when executing the command accurately based on the incorrect interpretation, the final image remains incorrect. If we provide the correct state, we can determine that the same manipulator was functioning correctly, so we cannot resolve the current issue by simply modifying the manipulation rules.

In the moving ball experiment, a picture of the past was shown one, two, four, or eight times. Only by looking at one picture it was difficult to determine whether the ball was moving to the left or to the right. When two pictures were shown, the speed estimation of the measurement‑based model improved greatly. However, eight pictures were not always better than four, and when viewed from a distance small errors accumulated, causing the prediction to deviate greatly from the screen.

There are conditions in the rotation rules as well. If you rotate the physics scene, you must also handle the direction of gravity and wind correctly. The model that follows those conditions matched the next moment better. However, it was not better than a simple benchmark that directly calculated the already known physical equations, and the long future was not stable.

Finally, it was tested in other published examples of research. The symmetric model was better than the general model, but the overall error was larger than copying the picture exactly. It left the result that it is difficult to declare that the model learned a nice rule unchanged.

Now it is clear where the next improvement should target. Each needs to be fixed: the ability to accurately read objects, the ability to preserve parts that should not be changed, and the ability to maintain stable movement for an extended period. **We have confirmed that adhering to rotation rules and properly implementing the user's intentions are different tests.**

The result calculation checks are recorded from the end of the process. 240 human judgment sheets are prepared, but there are no actual responses yet, so we do not claim that human evaluation has been completed. [Research Memo containing numbers and evidence](RESEARCH_MEMO_EN.md) · [Detailed conclusions of dynamics](C07_SYNTHESIS_EN.md)
