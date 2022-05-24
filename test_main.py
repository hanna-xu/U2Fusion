from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
# from Net import Generator, WeightNet
from scipy.misc import imread, imsave
import scipy.io as scio
from skimage import transform, data
from glob import glob
from model import Model

import matplotlib.image as mpimg


MODEL_SAVE_PATH = './model/model.ckpt'


output_path='./results/vis-ir/TNO/'
path = './test_imgs/vis-ir/TNO/'
path1 = path + 'vis/'
path2 = path + 'ir/'

# output_path='./results/vis-ir/RoadScene/'
# path = './test_imgs/vis-ir/RoadScene/'
# path1 = path + 'vis/'
# path2 = path + 'ir/'

# output_path ='./results/medical/'
# path = './test_imgs/medical/'
# path1 = path + 'pet/'
# path2 = path + 'mri/'

# output_path='./results/multi-exposure/dataset1/'
# path = './test_imgs/multi-exposure/dataset1/'
# path1 = path + 'oe/'
# path2 = path + 'ue/'

# output_path='./results/multi-exposure/dataset2/'
# path = './test_imgs/multi-exposure/dataset2/'
# path1 = path + 'oe/'
# path2 = path + 'ue/'

# output_path='./results/multi-focus/'
# path = './test_imgs/multi-focus/'
# path1 = path + 'far/'
# path2 = path + 'near/'


def listdir(path):
	list_name=[]
	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		if os.path.isdir(file_path):
			os.listdir(file_path, list_name)
		else:
			list_name.append(file_path)
	return list_name

def rgb2ycbcr(img_rgb):
	R = img_rgb[:, :, 0]
	G = img_rgb[:, :, 1]
	B = img_rgb[:, :, 2]
	Y = 0.299 * R + 0.587 * G + 0.114 * B
	Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
	Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
	return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128 / 255.0)
    G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    B = Y + 1.772 * (Cb - 128 / 255.0)
    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)
    return np.concatenate([R, G, B], axis=-1)



def main():
    print('\nBegin to generate pictures ...\n')
    Format = '.png'
    
    files = listdir(path1)
    time_cost = np.ones([len(files)], dtype=float)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pic_num = 0

    with tf.Graph().as_default(), tf.Session() as sess:
        M = Model(BATCH_SIZE=1, INPUT_H=None, INPUT_W=None, is_training=False)
        # restore the trained model and run the style transferring
        t_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list=t_list)
        model_save_path = MODEL_SAVE_PATH
        print(model_save_path)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_path)

        for file in files:
            pic_num += 1
            name = file.split('/')[-1]
            name = name.split('.')[-2]
            print("\033[0;33;40m[" + str(pic_num) + "/" + str(len(files)) + "]: " + name + ".png" + "\033[0m")

            img1 = imread(path1 + file.split('/')[-1], flatten=False) / 255.0
            img2 = imread(path2 + file.split('/')[-1], flatten=False) / 255.0

            Shape1 = img1.shape
            print("shape1:", Shape1)
            if len(Shape1) > 2:
                img1, img1_cb, img1_cr = rgb2ycbcr(img1)
            h1 = Shape1[0]
            w1 = Shape1[1]
            Shape2 = img2.shape
            h2 = Shape2[0]
            w2 = Shape2[1]
            print("shape2:", Shape2)
            assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
            img1 = img1.reshape([1, h1, w1, 1])
            img2 = img2.reshape([1, h1, w1, 1])


            start = time.time()
            outputs = sess.run(M.generated_img, feed_dict={M.SOURCE1: img1, M.SOURCE2: img2})
            output = outputs[0, :, :, 0]
            if len(Shape1)>2:
                output = ycbcr2rgb(output, img1_cb, img1_cr)
            end = time.time()
            time_cost[pic_num-1]=end-start
            print("Testing [%d] success,Testing time is [%f]\n" % (pic_num, end - start))
            imsave(output_path + name + Format, output)

    scio.savemat(output_path + '/time.mat', {'T':time_cost})


if __name__ == '__main__':
	main()
