from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
# from Net import Generator, WeightNet
from scipy.misc import imread, imsave

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




def main():
	print('\nBegin to generate pictures ...\n')
	Format='.png'
	for i in range(10):
		file_name1 = path1 + str(i + 1) + Format
		file_name2 = path2 + str(i + 1) + Format

		img1 = imread(file_name1) / 255.0
		img2 = imread(file_name2) / 255.0
		print('file1:', file_name1)
		print('file2:', file_name2)

		Shape1 = img1.shape
		h1 = Shape1[0]
		w1 = Shape1[1]
		Shape2 = img2.shape
		h2 = Shape2[0]
		w2 = Shape2[1]
		assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
		print('input shape:', img1.shape)
		img1 = img1.reshape([1, h1, w1, 1])
		img2 = img2.reshape([1, h1, w1, 1])

		with tf.Graph().as_default(), tf.Session() as sess:
			M = Model(BATCH_SIZE=1, INPUT_H=h1, INPUT_W=w1, is_training=False)
			# restore the trained model and run the style transferring
			t_list = tf.trainable_variables()
			saver = tf.train.Saver(var_list = t_list)
			model_save_path = MODEL_SAVE_PATH
			print(model_save_path)
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, model_save_path)
			outputs = sess.run(M.generated_img, feed_dict = {M.SOURCE1: img1, M.SOURCE2: img2})
			output = outputs[0, :, :, 0] # 0-1

			fig = plt.figure()
			f1 = fig.add_subplot(311)
			f2 = fig.add_subplot(312)
			f3 = fig.add_subplot(313)
			f1.imshow(img1, cmap = 'gray')
			f2.imshow(img2, cmap = 'gray')
			f3.imshow(output, cmap = 'gray')
			plt.show()

			if not os.path.exists(output_path):
				os.makedirs(output_path)
			imsave(output_path + str(i + 1) + Format, output)



if __name__ == '__main__':
	main()
