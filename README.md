# Endoscopic OCT Veress Needle

During laparoscopic surgery, precise placement of the Veress needle remains a challenge. In this study, a computer-aided endoscopic optical coherence tomography (OCT) system was developed to effectively and safely guide Veress needle insertion. We used convolutional neural networks (CNNs) in automatic tissue classification and distance estimation. The average testing accuracy in tissue classification was 98.53±0.39%, and the average testing relative error in distance estimation reached 4.42±0.56% (36.09±4.92 μm).

## Dataset

[https://zenodo.org/record/5659573](https://zenodo.org/record/5659573)

The data set is divided into two parts:
- **Classification**
    - The zip file **veress_classification_raw_images.zip** contains 40K images from 8 swine samples where there are 1K images per layer. There are 5 layers of images in the dataset; we excluded the set of images from the skin in this study. 
- **Regression**
    - The zip file **veress_regression_raw_images.zip** contains 8K images of the abdominal space from the same 8 swine samples where there are 1K images, and the ground truth distance labels for each sample are found in the Excel files **S[1-8]_distance_measurement_20210803.xlsx**. 

