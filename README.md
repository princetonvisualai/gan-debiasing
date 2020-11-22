

Steps to follow:

Preprocessing:
- Download the celeba dataset, put it in a folder named celeba/ in the data/ folder
- Run crop_images.py to crop images, making them square.

Baselines:
- Run main.py (--experiment baseline) to get baseline models for each target attribute

GAN:
- Train a progressive GAN on celeba (code here: https://github.com/facebookresearch/pytorch_GAN_zoo), save the final model in record/GAN_model/final_model.pt (or set pretrained=True in generate_images.py)

Our Model:
- Run generate_images.py (--experiment orig) to get latent vectors and generated images. 
- Run get_scores.py to compute baseline target attribute and protected attribute  scores for the generated images. 
- Run linear.py to compute hyperplanes and z'
- Run generate_images (--experiment pair) to generate paired images 
- Run main.py (--experiment model) to get our models.


Domain dependent hyperplanes:
- Run domain_dep_linear.py to compute hyperplanes and z'
- Run generate_images.py to generate paired images (set output dir, and latent file names)


