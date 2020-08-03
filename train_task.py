from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from datetime import datetime
# from scipy.misc import imsave
import scipy.ndimage
from skimage import img_as_ubyte
import os

from model import Model
# from deepIQA_evaluate import IQA

EPSILON = 1e-5

eps = 1e-8

logging_period = 50
patch_size = 64
LEARNING_RATE = 0.0001

def train_task(model, sess, trainset, save_path=None, validset=[], lam = 0, task_ind=1, c=[], merged=None, writer=None, saver=None, EPOCHES=2):
	start_time = datetime.now()
	num_imgs = trainset.shape[0]
	mod = num_imgs % model.batchsize
	n_batches = int(num_imgs // model.batchsize)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
	if mod > 0:
		trainset = trainset[:-mod]

	model.restore(sess)

	if task_ind == 1:
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			model.solver = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, decay = 0.6,
			                                           momentum = 0.15).minimize(model.content_loss, var_list = model.var_list)

	else:
		model.update_ewc_loss(lam = lam)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			model.solver = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, decay = 0.6,
			                                         momentum = 0.15).minimize(model.ewc_loss,
			                                                                   var_list = model.var_list)


	model.clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in model.var_list]

	initialize_uninitialized(sess)

	# ** Start Training **
	step = 0
	for epoch in range(EPOCHES):
		np.random.shuffle(trainset)
		# for batch in range(5):
		for batch in range(n_batches):
			model.step += 1
			step += 1
			# current_iter = step
			s1_index = np.random.choice([0, 1], 1)
			source1_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, s1_index[0]]
			source2_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, 1 - s1_index[0]]
			source1_batch = np.expand_dims(source1_batch, -1)
			source2_batch = np.expand_dims(source2_batch, -1)

			FEED_DICT= {model.SOURCE1: source1_batch, model.SOURCE2: source2_batch, model.c: c[task_ind-1]}

			sess.run([model.solver, model.clip], feed_dict = FEED_DICT)

			result = sess.run(merged, feed_dict = FEED_DICT)
			writer[task_ind - 1].add_summary(result, model.step)
			writer[task_ind - 1].flush()


			## validation
			if len(validset):
				for i in range(len(validset)):
					sub_validset=validset[i]
					batch_ind = np.random.randint(int(sub_validset.shape[0]/model.batchsize))
					s_index = np.random.choice([0, 1], 1)
					valid_source1_batch = sub_validset[batch_ind * model.batchsize:(batch_ind * model.batchsize + model.batchsize), :, :, s_index[0]]
					valid_source2_batch = sub_validset[batch_ind * model.batchsize:(batch_ind * model.batchsize + model.batchsize), :, :, 1 - s_index[0]]
					valid_source1_batch = np.expand_dims(valid_source1_batch, -1)
					valid_source2_batch = np.expand_dims(valid_source2_batch, -1)


					valid_FEED_DICT = {model.SOURCE1: valid_source1_batch, model.SOURCE2: valid_source2_batch, model.c:c[i]}
					valid_result = sess.run(merged, feed_dict = valid_FEED_DICT)
					writer[i].add_summary(valid_result, model.step)
					writer[i].flush()


			is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
			if is_last_step or step % logging_period == 0:
				elapsed_time = datetime.now() - start_time
				sloss = sess.run(model.ssim_loss, feed_dict = FEED_DICT)
				print("c", sess.run(model.c, FEED_DICT))
				print('epoch:%d/%d, step:%d/%d, model step:%d, elapsed_time:%s' % (
					epoch + 1, EPOCHES, step % n_batches, n_batches, model.step, elapsed_time))
				print('ssim loss: %s\n' % sloss)

				if hasattr(model, "ewc_loss"):
					add_loss = sess.run(model.Add_loss, feed_dict = FEED_DICT)
					print("Add_loss:%s\n" % add_loss)


			if is_last_step or step % 100 == 0:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')
	# writer.close()
	# saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def initialize_uninitialized(sess):
	global_vars = tf.global_variables()
	is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
	not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
	# print('not_initialized_vars:')
	# for i in not_initialized_vars:
	# 	print(str(i.name))
	if len(not_initialized_vars):
		sess.run(tf.variables_initializer(not_initialized_vars))



# def EN(inputs):
# 	len = inputs.shape[0]
# 	entropies = np.zeros(shape = (len, 1))
# 	grey_level = 256
# 	counter = np.zeros(shape = (grey_level, 1))
#
# 	for i in range(len):
# 		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
# 		input_uint8 = input_uint8 + 1
# 		for m in range(patch_size):
# 			for n in range(patch_size):
# 				indexx = input_uint8[m, n]
# 				counter[indexx] = counter[indexx] + 1
# 		total = np.sum(counter)
# 		p = counter / total
# 		for k in range(grey_level):
# 			if p[k] != 0:
# 				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
# 	return entropies


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g