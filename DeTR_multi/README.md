# Detection/Segmentation
____
## Description:
At the beginning this project was planned as a competitive solution, but then suspicions of competition were found and the team disappeared. I started my own project based on the developments made:)\
The previous dataset consists of about 8000 images with size 8000x8000 and bounding boxes as target. So it was just a common detection task. Our team chose DALI because it decodes images with GPU (not CPU) and implements slice decoding (so good for pathches)
____
## Architecture:
____
## TODO:
- [ ] Implement DeTR architecture for detection, segmentation
- [ ] Test fork-based dataloading
- [ ] Choose interesting dataset (maybe COCO, segmentation dataset)
- [x] Spawn-based dataloading (Wow! with 'py_num_workers=6' it is 5 min/epoch)
- [x] DALI tutorial dataloading with 'fn.decoders.image_slice'(still slow, GIL interferse, 20 min/epoch)
- [x] Try Connectome dataloading with patching (too slow, hour/epoch)
