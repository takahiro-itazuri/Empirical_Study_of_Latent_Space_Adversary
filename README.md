# Empirical Study of Latent-Space Adversary
This repository provides implementations of "Empirical Study of Latent-Space Adversary".

## Setup
### 1. Install Packages
```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets
[MNIST](http://yann.lecun.com/exdb/mnist/), [SVHN](http://ufldl.stanford.edu/housenumbers/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [LSUN](https://www.yf.io/p/lsun) can be downloaded by command.
```bash
cd ${root}
python -c "from misc import *; download_all_dataset();"
```

### 3. Train Classifiers


### 4. Train GANs


### 5. Generate Adversarial Examples


### 6. Calculate Mean Curvatures


### 7. Adversarial Fine-tuning

