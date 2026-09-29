# R5 measurement limits observed before primary test scoring

The first frozen attribute reader (dSprites Single Atomic) has completed all
6,000 updates and full clean validation/test replay. It uses the source object's
true center and a fixed112px RGB crop, not masks or true target attributes.

Clean target object all-attribute accuracy is99.82% on validation but78.20% on
its official compositional test. Per-test-attribute accuracy is shape92.65%,
color83.67%,size88.20%; both objects jointly60.775%. These are measurement-reader
scores on ORIGINAL target images, not primary generation scores. The drop is
consistent with a reader sensitive to held-out factor combinations; it is not
proof that a particular primary predictor failed. No primary model test output
has been used to choose or train this reader.

Keep this reader and its frozen protocol. The primary generated-attribute
scorer reports raw attribute accuracy, clean-target calibration accuracy and
agreement with the clean-target readout. Do not subtract the calibration error
from prediction error as an accuracy correction; these are different image
distributions. Do not call these scores perfect semantic ground truth.

A useful additional diagnostic would fix a common subset of targets read
correctly by the clean reader and compare every primary model on that same
subset, alongside ALL scenes and explicit subset denominators. Such a subset
is easier and cannot replace the full test. If an additional factor-balanced
synthetic reader or renderer-template control is built, freeze it before
inspecting its primary-model scores, preserve this original reader, and label
its extra renderer/label information. No such additional reader is implemented
or claimed in this note.

CLEVR/CLEVRTex readers are queued separately. Their clean-test accuracy must
also be reported, whether high or low. Pixel results, readout measurements and
independent human evaluation remain distinct requirements/evidence.
