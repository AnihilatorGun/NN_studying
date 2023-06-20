# Face recognition system
Here one can find my implementation of the face recognition system. It consists of three parts - the first part is the network that processes the input image (find face, crops it and resizes for further processing), the second part is the embedder that maps the image to a relatively small latent space (in my implementation - the Cartesian product of three 127 dimensional spheres) and the "database" where the obtained embedding is compared with other embeddings that are in the "database" (it is not a real database, but only an imitation as a plug).
----
## Creation details:
+ Created the map style dataset CelebA, find its channel stats (in my implementation I use io from torchvision and after casting the obtained tensor from uint8 to float I don't divide it by 256, so one have to be careful).
+ Created the simplest embedder using ResNet (I implemented the 18th, 34th and 50th versions of ResNet with different settings if you don't need to use the last fully connected layer or want to use relu after the very last layer).
+ Implemented classifier (it implements margin losses in logit calculations).
+ Added it to the pytorch-lightnong module (with adabelief oprimizer, cosine anneling warm restarts as schedulers and MAP @R, P@1 and A@1 as quality metrics).
+ Found some statistics on the training dataset and used them to combine three embedders into one ensemble.
+ Created module implementing the interlogic of the "database" inter logics- cropping and resizing the input image (here I used pretrained model, see links), browsing the data in it and finding the right "user"
+ Tested the system with the pictures of my friends and family members:)

____
## References:
+ [Article about spherical classifiers](https://arxiv.org/pdf/1801.07698.pdf)
+ [Presentation from Tinkoff's course](https://algocode.ru/files/course_dlfall22/final-lecture-08-dl-basic.pdf)
+ [Adabelief optimizer](https://github.com/juntang-zhuang/Adabelief-Optimizer)
+ [Face alignment model](https://github.com/1adrianb/face-alignment)
