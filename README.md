# Flower Classification and Segmentation with Inception Modules (MATLAB)

A small CNN built from three stacked **inception modules**, used for both image classification and segmentation on the Oxford 17 Flowers dataset. I wanted to find out whether inception modules give good accuracy in a small, simple network, instead of relying on a large pretrained one.

![Network architecture](docs/network_architecture.png)

## Results

| Task | Metric | Result |
|---|---|---|
| Classification (17 classes) | Test accuracy | 67.65% |
| | Macro precision / recall | 75.8% / 67.6% |
| Segmentation (flower vs background) | Mean accuracy | 97.48% |
| | Weighted IoU | 95.26% |
| | Mean BF score | 76.3% |

Each result is from a single run. The classification split is random (no fixed seed), so its accuracy may vary by a few points between runs, and its test set is 136 images (8 per class). The segmentation split is fixed with `rng(0)`, but its test set is only 7 images (57 training, 7 validation and 7 test, from 71 Daffodil images), so treat those metrics as indicative.

Both tasks use an 80/10/10 train/validation/test split.

**Classification variants** (test accuracy; each is a small edit to the base script):

| Variant | Accuracy |
|---|---|
| Base model | 67.65% |
| Base + batch norm inside each module | 65.44% |
| Greater channels (no halving between blocks) | 69.12% |
| Smaller mini-batch (64) | 65.44% |

The larger-channel model is slightly better, but with a 136-image test set one image is worth about 0.7 points, so I kept the base model. It also trained noticeably more slowly.

## What I found

- **Segmentation works well.** It finds 97.3% of flower pixels. The low BF score shows that the predicted boundary is often slightly too large or too small, even when the overlap is high.
- **Classification overfits.** Training accuracy reaches about 100% while validation stays near 80%, and test accuracy is lower still. The model can separate the classes it has seen but doesn't generalise well to new images.
- **The errors are between look-alike flowers.** 10 of the 17 classes have at least 6 of 8 test images correct. Class 1 has none, and its images are mostly labelled as class 8, another yellow flower with the same number of petals. Colour and shape are too similar for this small network.
- **Early layers hurt segmentation.** The classification network starts with a 7x7 convolution, a 3x3 convolution, max pooling and batch norm. For segmentation, removing the 3x3 convolution and the pooling and batch norm after it gave tighter boundaries, because the downsampling threw away spatial detail on a small dataset.

![Segmentation with and without the early layers](docs/segmentation_early_layers.png)

*Example output with (left) and without (right) the early 3x3 convolution, pooling and batch norm layers. Flower image from the [Oxford 17 Flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).*

## How it works

**Data.** 17 classes with 80 images each, resized to 256 x 256. For segmentation I use the Daffodil class and merge the leaf and sky labels into a single background class, so the task is flower against background. Pixel counts set the class weights, so the flower class isn't under-represented.

**Network.** A 7x7 convolution, batch norm and a 1x1 convolution (32 channels) prepare the image. Three inception modules follow, each with parallel 1x1, 3x3 and 5x5 branches (1x1 convolutions reduce dimensions before the larger filters), concatenated at the end. Following *Rethinking the Inception Architecture*, I replaced the 5x5 branch with two stacked 3x3 convolutions in the second module for classification, and in the first and second modules for segmentation. Each module ends in max pooling, with 64 channels in the first block and the count doubling in each block after that.

- **Classification head:** batch norm, global average pooling, 50% dropout, a fully connected layer of 17 units and softmax.
- **Segmentation head:** transposed convolutions with stride 2 and ReLU between them, which upsample back to the input size, then softmax and a pixel classification layer.

**Training.** Both tasks use SGD with momentum, because Adam predicted every pixel as background in the segmentation model.

| | Classification | Segmentation |
|---|---|---|
| Epochs | 30 | 20 |
| Mini-batch | 128 | 8 |
| L2 regularisation | 0.0001 | 0.005 |

## Limitations and ideas

- Classification would need more generalisation techniques (augmentation, colour preprocessing) to be useful on unseen images, and printing the model's confidence for each class would show which classes it confuses.
- I chose the segmentation settings and the early-layer change by comparing outputs on a small dataset, so the boundary improvement is a visual observation, not a measured one.
- Each result is a single run, and the segmentation test set is small, so the metrics carry wide uncertainty.

## Running it

Requires MATLAB (developed in R2024b) with the Deep Learning, Image Processing and Computer Vision Toolboxes.

**Classification**

1. Download the [17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) and put the 1,360 images in a `17flowers/` folder next to the script.
2. Run `classification.m`. The first run sorts the images into class folders `1` to `17` (it moves the files), trains the network, saves it as `classnet.mat`, and prints the test accuracy with a confusion chart.

**Segmentation**

1. From the same Oxford dataset page, get the 71 Daffodil images and their segmentation labels. Resize them to 256 x 256 (use nearest-neighbour interpolation for the label masks so the class IDs stay intact) and put them in `daffodilSeg/ImagesRsz256` and `daffodilSeg/LabelsRsz256`.
2. Run `segmentation.m`. It trains the network, saves it as `segmentnet.mat`, and prints the segmentation metrics and confusion matrix.

The datasets aren't included in the repo.

## References

1. Szegedy et al., *Going Deeper with Convolutions*, 2014.
2. Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, 2015.
3. Lin et al., *Network In Network*, 2014 (global average pooling).
4. Hinton et al., *Improving neural networks by preventing co-adaptation of feature detectors*, 2012 (dropout).
