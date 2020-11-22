import pickle
import argparse
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from os import path, mkdir

import dnnlib

### Load TF PGAN model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Load pretrained PGAN
    with open('pganinv_celeba_128.pkl', 'rb') as f:
        E, G, D, Gs = pickle.load(f, encoding='latin1')

    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--random_seed', type=int, default=0)
    opt = vars(parser.parse_args())
    attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs',  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
        'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
        'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
        'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
        'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
        'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    opt['attr_name'] = attr_list[opt['attribute']]

    print('\nStart generating images for', opt['attr_name'])
    start_time = time.time()

    # Create directories for saving images
    out_dir = 'fake_images_raninv/{}/prime'.format(opt['attr_name'])
    if not path.isdir('fake_images_raninv/{}'.format(opt['attr_name'])): mkdir('fake_images_raninv/{}'.format(opt['attr_name']))
    if not path.isdir(out_dir): mkdir(out_dir)

    # Load latent vectors
    latent_vectors = pickle.load(open('record/inverted_raninv/prime/latent_vectors_{}.pkl'.format(opt['attr_name']), 'rb'))

    # Generate and save images in batches
    batch_size = 64
    num_images = latent_vectors.shape[0]
    N = int(num_images/batch_size)
    if num_images%batch_size != 0: N += 1

    count = 0
    for B in range(N):
        z = latent_vectors[B*batch_size:(B+1)*batch_size]
        fmt = dict(func=dnnlib.tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        generated_images = Gs.run(z, None, randomize_noise=False, output_transform=fmt)

        for b in range(z.shape[0]):
            img = Image.fromarray(generated_images[b], 'RGB')
            img.save('{}/gen_{}.jpg'.format(out_dir, count))
            count += 1

        if B%100 == 0:
            print(' Batch', B, time.time()-start_time, 'seconds')

    print('Finished generating images for', opt['attr_name'])
    print('Total time', (time.time()-start_time)//60, 'minutes')
