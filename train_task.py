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
		# model.G_solver1 = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE*2, decay = 0.6,
		#                                            momentum = 0.15).minimize(model.ssim_loss,
		#                                                                      var_list = model.theta_G)
		# model.G_solver2 = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, decay = 0.6,
		#                                            momentum = 0.15).minimize(model.self_loss,
		#                                                                      var_list = model.theta_G)
		# model.W_solver = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE*2, decay = 0.6,
		#                                            momentum = 0.15).minimize(model.ssim_loss,
		#                                                                      var_list = model.theta_W)

	else:
		model.update_ewc_loss(lam = lam)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			model.solver = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, decay = 0.6,
			                                         momentum = 0.15).minimize(model.ewc_loss,
			                                                                   var_list = model.var_list)


	model.clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in model.var_list]
	# model.clipW = [p.assign(tf.clip_by_value(p, -30, 30)) for p in model.theta_W]

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
			# closs, add_loss = sess.run([model.content_loss, model.Add_loss], feed_dict = FEED_DICT)
			#print("Content loss:%s, Add_loss:%s\n" % (closs, add_loss*lam/2))

			# ms, ws1, ws2 = sess.run([model.s, model.ws1, model.ws2], feed_dict = FEED_DICT)
			# print("s1-s2:", ws1[0]-ws2[0])
			# print("s1-s2:", ws1[1] - ws2[1])
			# print("s1-s2:", ws1[2] - ws2[2])
			# print("w1: %s, w2: %s" % (ms[0, 0], ms[0, 1]))
			# print("w1: %s, w2: %s" % (ms[1, 0], ms[1, 1]))
			# print("w1: %s, w2: %s\n" % (ms[2, 0], ms[2, 1]))
			# fig = plt.figure()
			# f1 = fig.add_subplot(321)
			# f2 = fig.add_subplot(322)
			# f3 = fig.add_subplot(323)
			# f4 = fig.add_subplot(324)
			# f5 = fig.add_subplot(325)
			# f6 = fig.add_subplot(326)
			# f1.imshow(source1_batch[0, :, :, 0], cmap='gray')
			# f2.imshow(source2_batch[0, :, :, 0], cmap='gray')
			# f3.imshow(source1_batch[1, :, :, 0], cmap='gray')
			# f4.imshow(source2_batch[1, :, :, 0], cmap='gray')
			# f5.imshow(source1_batch[2, :, :, 0], cmap='gray')
			# f6.imshow(source2_batch[2, :, :, 0], cmap='gray')
			# plt.show()


			# generated_img = sess.run(model.generated_img, feed_dict=FEED_DICT)
			# with tf.device('/gpu:1'):
			# 	iqa_f=IQA(inputs = generated_img, trained_model_path = IQA_model)
			# 	en_f=EN(generated_img)
			# 	score_f=np.mean(iqa_f + w_en * en_f)

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
					# valid_w1, valid_w2 = W(inputs1 = valid_source1_batch, inputs2 = valid_source2_batch, trained_model_path = IQA_model,w_en=w_en,c=c)

					valid_FEED_DICT = {model.SOURCE1: valid_source1_batch, model.SOURCE2: valid_source2_batch, model.c:c[i]}
					# print("c", sess.run(model.c, feed_dict = valid_FEED_DICT))
					# valid_generated_img = sess.run(model.generated_img, feed_dict = valid_FEED_DICT)

					# with tf.device('/gpu:1'):
					# 	valid_iqa_f = IQA(inputs = valid_generated_img, trained_model_path = IQA_model)
					# 	valid_en_f = EN(valid_generated_img)
					# 	valid_score_f = np.mean(valid_iqa_f + w_en * valid_en_f)

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


# def W(inputs1,inputs2, trained_model_path, w_en, c):
# 	# with tf.device('/gpu:1'):
# 	iqa1 = IQA(inputs = inputs1, trained_model_path = trained_model_path)
# 	iqa2 = IQA(inputs = inputs2, trained_model_path = trained_model_path)
#
# 	with tf.device('/cpu:0'):
# 		en1 = EN(inputs1)
# 		en2 = EN(inputs2)
# 		score1 = iqa1 + w_en * en1
# 		score2 = iqa2 + w_en * en2
# 		w1 = np.exp(score1 / c) / (np.exp(score1 / c) + np.exp(score2 / c))
# 		w2 = np.exp(score2 / c) / (np.exp(score1 / c) + np.exp(score2 / c))

	# print('IQA:   1: %f, 2: %f' % (iqa1[0], iqa2[0]))
	# print('EN:    1: %f, 2: %f' % (en1[0], en2[0]))
	# print('total: 1: %f, 2: %f' % (score1[0], score2[0]))
	# print('w1: %s, w2: %s\n' % (w1[0], w2[0]))
	# print('IQA:   1: %f, 2: %f' % (iqa1[1], iqa2[1]))
	# print('EN:    1: %f, 2: %f' % (en1[1], en2[1]))
	# print('total: 1: %f, 2: %f' % (score1[1], score2[1]))
	# print('w1: %s, w2: %s\n' % (w1[1], w2[1]))
	# fig = plt.figure()
	# fig1 = fig.add_subplot(221)
	# fig2 = fig.add_subplot(222)
	# fig3 = fig.add_subplot(223)
	# fig4 = fig.add_subplot(224)
	# fig1.imshow(inputs1[0, :, :, 0], cmap = 'gray')
	# fig2.imshow(inputs2[0, :, :, 0], cmap = 'gray')
	# fig3.imshow(inputs1[1,:,:,0],cmap='gray')
	# fig4.imshow(inputs2[1,:,:,0],cmap='gray')
	# plt.show()
	# return (w1,w2)



def EN(inputs):
	len = inputs.shape[0]
	entropies = np.zeros(shape = (len, 1))
	grey_level = 256
	counter = np.zeros(shape = (grey_level, 1))

	for i in range(len):
		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
		input_uint8 = input_uint8 + 1
		for m in range(patch_size):
			for n in range(patch_size):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
	return entropies


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g