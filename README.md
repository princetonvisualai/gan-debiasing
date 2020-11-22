
# Fair Attribute Classification through Latent Space De-biasing

We suggest running our code in the following order.

## Main experiments

#### Data processing:
- Download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put it in `data/celeba`.
- Run `crop_images.py` to crop the aligned & cropped 178×218 images to 128x128.

#### Baseline:
- Run `main.py --experiment baseline` to train a standard attribute classifier for each target attribute.

#### GAN:
- Option 1: Train a (Progressive) GAN on CelebA.
- Option 2: Set `pretrained=True` in `generate_images.py` to use a GAN trained by [Facebook Research](https://github.com/facebookresearch/pytorch_GAN_zoo).
<!--Train a progressive GAN on celeba (code here: https://github.com/facebookresearch/pytorch_GAN_zoo), save the final model in record/GAN_model/final_model.pt (or set pretrained=True in generate_images.py)-->

#### Our model:
- Run `generate_images.py --experiment orig` to sample random latent vectors z and generated images. 
- Run `get_scores.py` to hallucinate labels for the generated images with the trained baseline models.
- Run `linear.py` to estimate hyperplanes and compute complementary latent vectors z' (our augmentation).
- Run `generate_images.py --experiment pair` to generate images from z'. 
- Run `main.py --experiment model` to train our models (i.e. target classifiers trained with our augmented data).


## Extensions of our method

#### Domain-dependent hyperplanes:
- Run `domain_dep_linear.py` to estimate domain-dependent hyperplanes and compute z'.
- Run `generate_images.py` to generate images from z. Set output directory and latent vector filenames.

#### Augmenting real-images with GAN-inversion:
- Train a GAN with an inversion module. We used the [in-domain GAN inversion method](https://github.com/genforce/idinvert) of Zhu et al.
- Invert the CelebA training set images to latent vectors z_inv.
- Run `linear_inv.py` to estimate hyperplanes and compute complementary latent vectors z_inv' (our augmentation).
- Run `generate_images.py --experiment pair` to generate images from z_inv'. 
- Run `main.py --experiment model_inv` to train target classifiers trained with data augmented from real images.


