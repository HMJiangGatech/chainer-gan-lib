# On Computation and Generalization of Generative Adversarial Networks under Spectrum Control

This repository is forked from https://github.com/pfnet-research/chainer-gan-lib.

For experiments in the [paper](https://openreview.net/forum?id=rJNH6sAqY7), please use the command in [here](paperexp.txt).

# Introduction
This repository collects chainer implementation of state-of-the-art GAN algorithms.  
These codes are evaluated with the _inception score_ on Cifar-10 dataset.  
Note that our codes are not faithful re-implementation of the original paper.

How to use
-------

Install the requirements first:
```
pip install -r requirements.txt
```
This implementation has been tested with the following versions.
```
python 3.5.2
chainer 4.0.0
+ https://github.com/chainer/chainer/pull/3615
+ https://github.com/chainer/chainer/pull/3581
cupy 3.0.0
tensorflow 1.2.0 # only for downloading inception model
numpy 1.11.1
```
Download the inception score module forked from [https://github.com/hvy/chainer-inception-score](https://github.com/hvy/chainer-inception-score).
```
git submodule update -i
```
Download the inception model.
```
cd common/inception
python download.py --outfile inception_score.model
```
You can start training with `train.py`.

`python train.py --gpu 0 --algorithm dcgan --out result_dcgan`

Please see `example.sh` to train other algorithms.

License
-------
MIT License. Please see the LICENSE file for details.
