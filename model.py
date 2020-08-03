from __future__ import print_function
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

import scipy.io as scio
import time
from datetime import datetime
from scipy.misc import imsave
import scipy.ndimage
from skimage import img_as_ubyte

from Net import Generator
from LOSS import SSIM_LOSS, Fro_LOSS #, L1_LOSS, Fro_LOSS
from collections import Iterable

from VGGnet.vgg16 import Vgg16
WEIGHT_INIT_STDDEV = 0.05

eps = 1e-8


class Model(object):
	def __init__(self, BATCH_SIZE, INPUT_H, INPUT_W, is_training):
		self.batchsize = BATCH_SIZE
		self.G = Generator('Generator')
		self.var_list = []

		self.step = 0
		if not hasattr(self, "ewc_loss"):
			self.Add_loss = 0

		self.SOURCE1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, INPUT_H, INPUT_W, 1), name = 'SOURCE1')
		self.SOURCE2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, INPUT_H, INPUT_W, 1), name = 'SOURCE2')

		self.c = tf.placeholder(tf.float32, shape = (), name = 'c')
		print('source shape:', self.SOURCE1.shape)

		self.generated_img = self.G.transform(I1 = self.SOURCE1, I2 = self.SOURCE2, is_training = is_training, reuse=False)
		self.var_list.extend(tf.trainable_variables())

		# for i in self.var_list:
		# 	print(i.name)

		if is_training:
			''' SSIM loss'''
			SSIM1 = 1 - SSIM_LOSS(self.SOURCE1, self.generated_img)
			SSIM2 = 1 - SSIM_LOSS(self.SOURCE2, self.generated_img)
			mse1 = Fro_LOSS(self.generated_img-self.SOURCE1)
			mse2 = Fro_LOSS(self.generated_img-self.SOURCE2)

			with tf.device('/gpu:1'):
				self.S1_VGG_in = tf.image.resize_nearest_neighbor(self.SOURCE1, size = [224, 224])
				self.S1_VGG_in = tf.concat((self.S1_VGG_in, self.S1_VGG_in, self.S1_VGG_in), axis = -1)
				self.S2_VGG_in = tf.image.resize_nearest_neighbor(self.SOURCE2, size = [224, 224])
				self.S2_VGG_in = tf.concat((self.S2_VGG_in, self.S2_VGG_in, self.S2_VGG_in), axis = -1)

				vgg1 = Vgg16()
				with tf.name_scope("vgg1"):
					self.S1_FEAS = vgg1.build(self.S1_VGG_in)
				vgg2 = Vgg16()
				with tf.name_scope("vgg2"):
					self.S2_FEAS = vgg2.build(self.S2_VGG_in)

				for i in range(len(self.S1_FEAS)):
					self.m1 = tf.reduce_mean(tf.square(features_grad(self.S1_FEAS[i])), axis = [1, 2, 3])
					self.m2 = tf.reduce_mean(tf.square(features_grad(self.S2_FEAS[i])), axis = [1, 2, 3])
					if i == 0:
						self.ws1 = tf.expand_dims(self.m1, axis = -1)
						self.ws2 = tf.expand_dims(self.m2, axis = -1)
					else:
						self.ws1 = tf.concat([self.ws1, tf.expand_dims(self.m1, axis = -1)], axis = -1)
						self.ws2 = tf.concat([self.ws2, tf.expand_dims(self.m2, axis = -1)], axis = -1)

			self.s1 = tf.reduce_mean(self.ws1, axis = -1) / self.c
			self.s2 = tf.reduce_mean(self.ws2, axis = -1) / self.c
			self.s = tf.nn.softmax(
				tf.concat([tf.expand_dims(self.s1, axis = -1), tf.expand_dims(self.s2, axis = -1)], axis = -1))


			self.ssim_loss = tf.reduce_mean(self.s[:, 0] * SSIM1 + self.s[:, 1] * SSIM2)
			self.mse_loss = tf.reduce_mean(self.s[:, 0] * mse1 + self.s[:, 1] * mse2)
			self.content_loss = self.ssim_loss + 20 * self.mse_loss



	def compute_fisher(self, imgset, c, sess, num_samples = 200):
		# computer Fisher information for each parameter
		# initialize Fisher information for most recent task
		self.F_accum = []
		print("val_list length:", len(self.var_list))


		# for i in self.var_list:
		# 	print(i.name)

		for v in range(len(self.var_list)):
			self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

		start_time_cf = datetime.now()

		for i in range(num_samples):
			for k in range(len(imgset)):
			# select random input image
				set=imgset[k]
				im_ind = np.random.randint(set.shape[0] - self.batchsize)
				# compute first-order derivatives
				s1_index = np.random.choice([0, 1], 1)
				s1 = np.expand_dims(set[im_ind:im_ind + self.batchsize, :, :, s1_index[0]], -1)
				s2 = np.expand_dims(set[im_ind:im_ind + self.batchsize, :, :, 1 - s1_index[0]], -1)
				ders = sess.run(tf.gradients(-self.content_loss, self.var_list),
				                feed_dict = {self.SOURCE1: s1, self.SOURCE2: s2, self.c: c[k]})

			# square the derivatives and add to total
				for v in range(len(self.F_accum)):
					self.F_accum[v] += np.square(ders[v])

			elapsed_time_cf = datetime.now() - start_time_cf
			print("compute fisher: %s/%s, elapsed_time: %s" % (i + 1, num_samples, elapsed_time_cf))


		# divide totals by number of samples
		for v in range(len(self.F_accum)):
			self.F_accum[v] /= (num_samples*len(imgset))

		# scio.savemat('F1.mat', {'F': self.F_accum})
		# f = open("F.txt", "w")
		# str = '\n'
		# f.writelines(str.join(self.F_accum))
		# f.close()

	# F_save = flatten(self.F_accum)
		# for v in range(len(F_save)):
		# 	print(F_save,'\n')




	def star(self):
		# used for saving optimal weights after most recent task training
		self.star_vars = []
		for v in range(len(self.var_list)):
			self.star_vars.append(self.var_list[v].eval())

	def restore(self, sess):
		# reassign optimal weights for latest task
		if hasattr(self, "star_vars"):
			for v in range(len(self.var_list)):
				sess.run(self.var_list[v].assign(self.star_vars[v]))

	def update_ewc_loss(self, lam):
		# elastic weight consolidation
		# lam is weighting for previous task(s) constraints
		if not hasattr(self, "ewc_loss"):
			self.ewc_loss = self.content_loss

		for v in range(len(self.var_list)):
			self.Add_loss += tf.reduce_sum(
				tf.multiply(self.F_accum[v].astype(np.float32), tf.square(self.var_list[v] - self.star_vars[v])))
			self.ewc_loss += (lam / 2) * self.Add_loss



# def EN(inputs):
# 	len = inputs.shape[0]
# 	entropies = []  # tf.Variable(tf.zeros(shape = (len, 1)))
# 	grey_level = 256
# 	counter = tf.Variable(tf.zeros(shape = (grey_level, 1), dtype = tf.int32))
# 	one = tf.constant(1)
#
# 	for i in range(len):
# 		input_uint8 = tf.cast(inputs[i, :, :, 0] * 255, dtype = tf.int32)
# 		input_uint8 = input_uint8 + 1
# 		W = inputs.shape[1]
# 		H = inputs.shape[2]
# 		for m in range(int(W)):
# 			for n in range(int(H)):
# 				indexx = input_uint8[m, n]
# 				# print("counter[indexx]", counter[indexx])
# 				counter[indexx] = tf.add(counter[indexx], one)
# 		counter = tf.cast(counter, tf.float32)
# 		total = tf.reduce_sum(counter)
# 		p = counter / total
# 		for k in range(grey_level):
# 			entropies.append(- p[k] * (tf.log(p[k] + eps) / tf.log(2.0)))
# 	return entropies
#
#
# def tf_EN(inputs):
# 	len = inputs.shape[0]
# 	W = int(inputs.shape[1])
# 	H = int(inputs.shape[2])
# 	total = W * H * 1.0
# 	grey_level = 256
#
# 	for i in range(len):
# 		input0 = tf.cast(inputs[i, :, :, 0] * 255, dtype = tf.int32)
# 		input1 = tf.cast(input0, dtype = tf.float32)
# 		input2 = input1 + 1
#
# 		for k in range(grey_level):
# 			if k == 0:
# 				p = tf.expand_dims(num(input2, k) / total, axis = 0)
# 			else:
# 				p = tf.concat([p, tf.expand_dims(num(input2, k) / total, axis = 0)], axis = 0)
# 		ep = - tf.multiply(p, tf.log(p + eps * eps) / tf.Variable(tf.log(2.0)))
#
# 		if i == 0:
# 			entropies = tf.expand_dims(tf.reduce_sum(ep), axis = 0)
# 		else:
# 			entropies = tf.concat([entropies, tf.expand_dims(tf.reduce_sum(ep), axis = 0)], axis = 0)
# 	return entropies


def num(input, k):
	a = binary(input - tf.Variable(k + 0.5))
	b = binary(tf.Variable(k + 1.5) - input)
	return tf.reduce_sum(tf.multiply(a, b))


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g


def features_grad(features):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	_, _, _, c = features.shape
	c = int(c)
	for i in range(c):
		fg = tf.nn.conv2d(tf.expand_dims(features[:, :, :, i], axis = -1), kernel, strides = [1, 1, 1, 1],
		                  padding = 'SAME')
		if i == 0:
			fgs = fg
		else:
			fgs = tf.concat([fgs, fg], axis = -1)
	return fgs


def flatten(input_list):
	output_list = []
	while True:
		if input_list == []:
			break
		for index, i in enumerate(input_list):

			if type(i) == list:
				input_list = i + input_list[index + 1:]
				break
			else:
				output_list.append(i)
				input_list.pop(index)
				break
	return output_list


@tf.RegisterGradient("QuantizeGrad")
def sign_grad(op, grad):
	input = op.inputs[0]
	cond = (input >= -1) & (input <= 1)
	zeros = tf.zeros_like(grad)
	return tf.where(cond, grad, zeros)


def binary(input):
	x = input
	with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
		x = tf.sign(x)
	x = x / 2 + 0.5
	return x
