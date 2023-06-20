# BERToDeTR
____
## Dataset description:
The original dataset consists of about 2 million articles retrieved from arxiv.com. I cleaned it up, leaving only 57 classes (there were about 20 classes with fewer than 10 samples), rebalanced it, split it into training and testing, and packaged it into the Huggingface dataset
[Original dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
[Mine](https://www.kaggle.com/datasets/ushakovnikitamipt/arxiv-multilable-classification)
____
## Architecture:
BERToDeTR is my own proposal to create a multilabel classification (a sample can have more than 1 class). The main idea is a transformer-decoder with trainable constant length queries inherited from DeTR. This is a natural solution for object detection and panoptic segmentation, as it does not need NMS, anchor boxes and other tricks (ok, there are trainable queries, but that is the only "trick"). And the main feature of these tasks is the variable number of model outputs, just like in multilabel classification. So I decided to create a combination of BERT and DeTR for this task and test if it is effective.
![BERToDeTR architecture](https://github.com/AnihilatorGun/NN_studying/blob/master/BERToDeTR/BERToDeTR_architecture.jpg)
____
## TODO:
- [ ] Train models, compare them with each other
- [ ] Implement auxiliary losses (get the outputs from each decoder layer, pass them through FFN, calculate the losses and add them to the main loss)
- [ ] FIX "memory_key_padding_mask inside" the decoder - BERT takes it as float [0.0, 1.0], where 1.0 means the element is taken, 0.0 means the element is not taken, but TransformerDecoder as implemented in Torch takes either bool or float, and floar is simply added to the attention.
- [x] Apply some methods from "composer" (there are some API inconsistencies, some methods should not require an optimizer. Also some methods require specific CUDA library which cannot be installed in kaggle)
- [x] Implement hungarian algorithm (meh, made from scratch, MUST be modified)
- [x] Implement DeTR head
- [x] Implement baseline BERT-FC for multilabel classification
- [x] Clear and prepare dataset

