# Cloudiness Estimation
____
## Dataset description:
The original dataset consists of about 101,000 512x512 labeled images and masks for them.
Each image is a projection of a hemisphere onto the plane taken with spherical cameras taken during an expedition. The labels are ints in [0; 8], the mask is also a 512x512 image hiding the ship's equipment. The dataset was processed - each mask was applied to the image and the resulting mask was resized to 256x256. Also, a small expedition was deleted as there were many misidentifications.
Then the data was split into a training set and a test set, so that one day's images only make up either the training set or the test set.
And some more info about labeling - photos are taken every 10 seconds (approximately), but the specialist only took notes about the weather every hour, so some images are mislabeled (but there is no way to find out - you would have to look through the whole dataset). This fact is the main difficulty of the task.
![Data sample](https://github.com/AnihilatorGun/NN_studying/blob/master/cloudiness_estimation/sample.jpg)
____
## Architecture:
- Task is solved by regression (although classes exist, there are explicit hierarchy)
- MobileNet blocks (1x1 Conv; depthwise separable Conv 3x3; 1x1 Conv) with expansion in the middle (the standard Resnet block has 4 times fewer channels in the middle, where 3x3 Conv is applied).
- Squeeze and Excitation blocks to spread geometric information faster.
- It is necessary to take into account that the images are hemispherical projection, so a RingCutter block is proposed (the captured image or activation with N channels is cut into 4 concentric rings) to provide the geometric information explicitly.
- At the end, after AdaptiveAvgPooling2d, there is a small network that converts timing information into new channel statistics and multiplies the CNN outputs with these new statistics (Squeeze&Excitation).
- A new specific loss function is proposed - it deforms each epoch starting with HuberLoss and then transforms - if the difference is small, nothing changes, but if the difference is large - it saturates to a certain value. The idea is to provide small gradients for samples where the labelling is wrong.
____
## TODO:
- [ ] Figure out how to improve the results and (maybe) find another way to find pure samples (with appropriate labels)
- [x] Find out if the RingCutter is effective (not at all, it seems that the default architecture gathers enough information to show good results. Anyway, there is no way to find out, since both methods achieve about 45% Acc and 85% Leq1, and further improvement is complicated since the dataset is poorly labelled).
- [x] Find out if the new loss is effective (not at all, the reasons are the same as in the previous paragraph).
- [x] Implement pruning (even on the simpliest mnist classification metrics dramatically drops after cutting 20% of the network, maybe I do something wrong)
____
## Conclusions:
- Improved MobileNet blocks and squeeze&excite blocks improve accuracy (see [baseline-article](https://www.mdpi.com/2072-4292/13/2/326#))
- To verify that re-loss and cutting into rings are effective, one must create a custom dataset using a specific method. The training dataset must contain images that have errors with the same error statistics (it is obvious that in the dataset used has difference between the true label and the current label is normal), the test dataset must be pure(!)
