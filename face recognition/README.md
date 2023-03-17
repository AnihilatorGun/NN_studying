# Face recognition system
Here one can find my realization of face recognition system. It consists of three parts - first part is the network that processes input image(find face, crops it and resizes for further processing), second part is the embedder which maps image to relatively small latent space(in my realization - Cartisial product of three 127 dimensional spheres) and "database" where obtained embedding is compaired with other embeddings are in "database"(it is not a real database, just a imaginary database).
----
## Creation details:
+ Created map-style dataset CelebA, find it's channel-statistics (in my realization I use io from torchvision and after cast obtained tensor from uint8 to float I don't divide it by 256, so one must be carefule).
+ Created simpliest embedder via ResNet (I've implemented 18th, 34th and 50th versions of resnet with different setting if one need not to use last fully-connected layer or want to use relu after the very last layer).
+ Created classifier (it implements margin-losses in logit calculations).
+ Stacked it in pytorch-lightnong module (using adabelief oprimizer, cosine anneling warm restarts as scheduler and MAP@R, P@1 and A@1 as quality metrics).
+ Found some statistics on train dataset and using it stacked three embedders into ensemble.
+ Created module that implements "database" inter logics - crops and resizes input image(here I used pretrained model, see links), looks throuth the data in it and finds proper "user"
+ Tested system on my friend's and familie's picture:)

____
## References:
+ [Article about spherical classifiers](https://arxiv.org/pdf/1801.07698.pdf)
+ [Presentation from Tinkoff's course](https://algocode.ru/files/course_dlfall22/final-lecture-08-dl-basic.pdf)
+ [Adabelief optimizer](https://github.com/juntang-zhuang/Adabelief-Optimizer)
+ [Face alignment model](https://github.com/1adrianb/face-alignment)
