
## Project 2, Week 5: Scene Classification with Transfer Learning

- Dataset: Intel Image Classification, 6 classes, 14,034 train / 3,000 test
- Model: ResNet-18 pretrained on ImageNet
- Feature extraction (3,078 trainable parameters): val accuracy 0.901
- Fine-tuning layer4 (about 8.4M trainable): val accuracy 0.931
- Test accuracy (used once): 0.936; weakest class: glacier (recall 0.870), confused with mountain
- Most instructive error: a photo of a small stone house beside a stream was labelled "buildings" but predicted as "mountain", because the landscape fills most of the image.
- Next: my own domain in Week 6: maize leaf disease classification (4 classes)

## Week 6 proposal


**Problem:** Maize (corn) is a major crop, and leaf diseases such as blight, common rust and gray leaf spot reduce yields when they are noticed late. I will build an image classifier that identifies the disease (or a healthy leaf) from a photo of a maize leaf.

**Who benefits:** Farmers and agriculture extension workers, who could get a quick first diagnosis from a phone photo.

**Dataset:** Corn or Maize Leaf Disease Dataset on Kaggle (smaranjitghose): https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset. It has 4 classes and 4,188 images: Common Rust (1,306), Gray Leaf Spot (574), Blight (1,146) and Healthy (1,162). It is built from the PlantVillage and PlantDoc datasets; license: data files (c) original authors, so I will credit them.

**Approach:** Reuse the Week 5 transfer learning pipeline (ResNet-18, feature extraction then fine-tuning) with my own train/validation/test split, and study which diseases get confused.
