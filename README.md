## Project 2, Week 5: Scene Classification with Transfer Learning

- Dataset: Intel Image Classification, 6 classes, 14,034 train / 3,000 test
- Model: ResNet-18 pretrained on ImageNet
- Feature extraction (3,078 trainable parameters): val accuracy 0.901
- Fine-tuning layer4 (about 8.4M trainable): val accuracy 0.931
- Test accuracy (used once): 0.936; weakest class: glacier (recall 0.870), confused with mountain
- Most instructive error: a photo of a small stone house beside a stream was labelled "buildings" but predicted as "mountain", because the landscape fills most of the image.
- Next: my own domain in Week 6: [your idea]

## Week 6 proposal

[Write one paragraph here: the problem, who benefits, the dataset link, and the number of classes and images.]
