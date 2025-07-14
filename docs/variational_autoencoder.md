# Vartiational AutoEncoder

## Trained Molels

| Dataset      | input size         | embedding size | latent channel | Epochs | MAE     | SSIM    | PSNR     |
| ------------ | ------------------ | -------------- | -------------- | ------ |-------- | ------- | -------- |
| MNIST        |  784 (1 x 28 x 28) |           256  |            128 |    100 | 0.03769 | 0.86650 | 19.69707 |
| FashionMNIST |  784 (1 x 28 x 28) |           512  |            256 |    100 | 0.06388 | 0.69344 | 18.89066 |
| CIFAR10      | 3072 (1 x 32 x 32) |          1024  |            512 |    100 | 0.08997 | 0.08997 | 18.77262 |

## Evaluation

### [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

<img src="../models/variational_autoencoder/mnist_784_256_128/loss_curve.png" width=360px >
<img src="../models/variational_autoencoder/mnist_784_256_128/embedding_vectors.png" width=360px >
<img src="../models/variational_autoencoder/mnist_784_256_128/reconstruction.png" width=720px >


### [FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

<img src="../models/variational_autoencoder/fashion_mnist_784_512_256/loss_curve.png" width=360px >
<img src="../models/variational_autoencoder/fashion_mnist_784_512_256/embedding_vectors.png" width=360px >
<img src="../models/variational_autoencoder/fashion_mnist_784_512_256/reconstruction.png" width=720px >

### [CIFAR10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

<img src="../models/variational_autoencoder/cifar10_3072_1024_512/loss_curve.png" width=360px >
<img src="../models/variational_autoencoder/cifar10_3072_1024_512/embedding_vectors.png" width=360px >
<img src="../models/variational_autoencoder/cifar10_3072_1024_512/reconstruction.png" width=720px >

## How to write a configuration file

```yaml:source/config/variational_autoencoder.yaml
dataset_path: /dataset/directory/path/  # dataset path (see `source/autoencoders/dataloader.py`).
width: 28  # Shape of the input image.
height: 28
channel: 1
embedding_size: 256  # The number of dimensions of the embedded representation.
latent_channel: 128  # The number of dimensions in the latent space.
max_epochs: 100  # Number of epochs.
batch_size: 64  # Batch_size
num_workers: 8  # How many subprocesses to use for data loading (see `torch.utils.data.DataLoader`).
device: cuda  # device name.
output_dir_path: output/dir/path/  #  Directory for storing network training and evaluation results.
model_filename: model_filename.pth  # File of traind model.
```
