# AutoEncoders in PyTorch

## Requirements

- Python 3.12
- torch 2.5.1
- torchvision 0.20.1

## Installation

```
$ git clone https://github.com/yuzunoki-haruno/AutoEncoders.git
$ cd AutoEncoders
$ pip install -r requirements.txt
```

## Usage

### Training

```
$ python run_<MODEL_NAME>.py --mode train --config_file <CONFIG_FILE>.yaml
```

### Evaluation

```
$ python run_<MODEL_NAME>.py --mode test --config_file <CONFIG_FILE>.yaml
```

## Documents

## Results

### [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

| Model name         | MAE     | SSIM    | PSNR     |
| ------------------ |-------- | ------- | -------- |
|  AutoEncoder       | 0.01276 | 0.97470 | 28.19348 |

### [FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

| Model name         | MAE     | SSIM    | PSNR     |
| ------------------ |-------- | ------- | -------- |
|  AutoEncoder       | 0.03296 | 0.89505 | 25.16095 |

### [CIFAR10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)

| Model name         | MAE     | SSIM    | PSNR     |
| ------------------ |-------- | ------- | -------- |
|  AutoEncoder       | 0.07820 | 0.58121 | 20.04579 |

## License

```
"THE BEER-WARE LICENSE" (Revision 42):

<yuzunoki.haruno@gmail.com> wrote this file.  As long as you retain this notice you
can do whatever you want with this stuff. If we meet some day, and you think
this stuff is worth it, you can buy me a beer in return. Haruno Yuzunoki
```
