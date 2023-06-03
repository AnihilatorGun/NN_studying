# Cloudiness Estimation
____
## Dataset description:
Original dataset consists of about 101.000 512x512 labeled images and masks for them.
Each iamge is projection of half-sphere on the plane made with spherical cameras taken during ocean expedition, labels are ints in [0; 8], mask is also 512x512 image that hides ship's equipment. Dataset has been processed - every mask has been applied on the image and the resulted mask has been resized to 256x256. Also one small expedition has been deleted becaus there were a lot miss-labeling.
Then data has been splitted into train-test in that way, that images from one's day are only founded either the train set or the test set.
And some info about the labeling - photoes are made every 10 second (approximatly), but the specialist made notes about weather only each hour, thus some images are misslabeled (but there is no way to find that - one should look through the whole dataset). That fact is the main difficulty of the task.

____
## Architecture:
- Task is solved using regression (despite classes exist, there are explicit hierarchy)
- MobileNet blocks (1x1 Conv;depthwise separable  Conv 3x3; 1x1 Conv) with expansion in the middle (defaul resnet block has 4 times less channels in the middle, where 3x3 conv is applied).
- Squeeze and Excitation blocks to spread geometrical information faster.
- One must take some information that images are half-spherical projection, so RingCutter block is made (taken image or activation with N channels is cutted into 4 concentric rings) to provide net geometric information explicitly.
- In the end, after AdaptiveAvgPooling2d there is small net which process time information into new channel statistics and multiply CNN outputs by this new statistics (Works like a Squeeze&Excitation block).
- New specific loss function is proposed - it deformates every epoch starting with HuberLoss and then transforms - when difference is small nothing changes, but when difference big - it saturates to some value. Idea - to provide small gradients for samples where label is missmathed.
----

## TODO:
- [ ] Find out if RingCutter is effective.
- [ ] Find out if new loss is effective.
- [ ] Implement Nvidia DALI (just for experiments)
- [ ] Implement pruning (find out if it is effective)
- [ ] Find out how to improve scores and (may be) find another way how to find pure samples (with appropriate labels)