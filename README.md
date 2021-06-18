
# Fair Attribute Classification through Latent Space De-biasing

### [Project Page](https://princetonvisualai.github.io/gan-debiasing/) | [Paper](https://arxiv.org/abs/2012.01469) | [Colab Notebook](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_GAN_Debiasing_(Optional).ipynb) | [2min Talk](https://youtu.be/j5LwfJQNJYU) | [5min Talk](https://youtu.be/KIJ61YF05Js) | [10min Talk](https://youtu.be/1ebHTCQRNs4)

This repo provides the code for our CVPR 2021 paper "Fair Attribute Classification through Latent Space De-biasing." 
```
@inproceedings{ramaswamy2020debiasing,
author = {Vikram V. Ramaswamy and Sunnie S. Y. Kim and Olga Russakovsky},
title = {Fair Attribute Classification through Latent Space De-biasing},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2021}
}
```
Our work is featured in Coursera's [Generative Adversarial Networks (GANs) Specialization](https://www.coursera.org/specializations/generative-adversarial-networks-gans) course. Check out the colab notebook linked above for details.

## Main experiments

#### Data processing:
- Download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put it in `data/celeba`.
- Run `crop_images.py` to crop the aligned & cropped 178×218 images to 128x128.

#### Baseline:
- Run `main.py --experiment baseline` to train a standard attribute classifier for each target attribute.

#### GAN:
- Option 1: Train a (Progressive) GAN on the CelebA training set (162,770 images).
- Option 2: Set `pretrained=True` in `generate_images.py` to use a GAN trained by [Facebook Research](https://github.com/facebookresearch/pytorch_GAN_zoo).
<!--Train a progressive GAN on celeba (code here: https://github.com/facebookresearch/pytorch_GAN_zoo), save the final model in record/GAN_model/final_model.pt (or set pretrained=True in generate_images.py)-->

#### Our model:
- Run `generate_images.py --experiment orig` to sample random latent vectors z and generated images. 
- Run `get_scores.py` to hallucinate labels for the generated images with the trained baseline models.
- Run `linear.py` to estimate hyperplanes and compute complementary latent vectors z' (our augmentation).
- Run `generate_images.py --experiment pair` to generate images from z'. Set image output directory and latent vector filename.
- Run `main.py --experiment model` to train our models (i.e. target classifiers trained with our augmented data).

## Extensions of our method

#### Using domain-dependent hyperplanes:
- Run `linear_dom_dep.py` to estimate domain-dependent hyperplanes and compute z' with them.
- Run `generate_images.py --experiment pair` to generate images from z' and train a classifier with these images.

#### Augmenting real-images with GAN-inversion:
- Train a GAN with an inversion module. We used the [in-domain GAN inversion method](https://github.com/genforce/idinvert) by Zhu et al.
- Invert CelebA images to latent vectors z_inv.
- Run `linear_inv.py` to estimate hyperplanes and compute complementary latent vectors z_inv' (our augmentation).
- Run `generate_images_inv.py` to generate images from z_inv'. This is the only script that requires TensorFlow as the GAN with an inversion module we've trained was implemented in TensorFlow.
- Run `main.py --experiment model_inv` to train target classifiers trained with data augmented from real images.

#### Augmenting two protected attributes:
- Run `linear_multi_sgd.py` to estimate domain-dependent hyperplanes and compute z' with them.
- Run `generate_images.py --experiment pair` to generate images from z' and train a classifier with these images.


## Additional experiments
- `full_skew_tests.py`: Code for running experiments on the discriminability of attributes.
- `linear_underrep.py`: Code for estimating hyperplanes with different fractions of positive/negative samples.

## Acknowledgements
This work is supported by the National Science Foundation under Grant No. 1763642 and the Princeton First Year Fellowship to SK. We also thank Arvind Narayanan, Deniz Oktay, Angelina Wang, Zeyu Wang, Felix Yu, Sharon Zhang, as well as the Bias in AI reading group for helpful comments and suggestions.
