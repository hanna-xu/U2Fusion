from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf

from train_task import train_task
from model import Model

# IQA_model = './IQA/models/nr_tid_weighted.model'

data1_path = './vis_ir_dataset64.h5'
data2_path = './oe_ue_Y_dataset64.h5'
data3_path = './far_near_Y_dataset64.h5'

patch_size = 64
LAM = 0 #80000
LAM_str = '0'
NUM = 30

EPOCHES = [3, 2, 2]
c = [3200, 3500, 100]

def main():
	with tf.Graph().as_default(), tf.Session() as sess:
		model = Model(BATCH_SIZE = 18, INPUT_W = patch_size, INPUT_H = patch_size, is_training = True)
		# for i in model.var_list:
		# 	print(i.name)

		saver = tf.train.Saver(var_list = model.var_list, max_to_keep = 10)

		tf.summary.scalar('content_Loss', model.content_loss)
		tf.summary.scalar('ssim_Loss', model.ssim_loss)
		tf.summary.scalar('mse_Loss', model.mse_loss)
		# tf.summary.scalar('s1', tf.reduce_mean(model.s1))
		# tf.summary.scalar('s2', tf.reduce_mean(model.s2))
		tf.summary.scalar('ss1', model.s[0, 0])
		tf.summary.scalar('ss2', model.s[0, 1])
		tf.summary.image('source1', model.SOURCE1, max_outputs = 3)
		tf.summary.image('source2', model.SOURCE2, max_outputs = 3)
		tf.summary.image('fused_result', model.generated_img, max_outputs = 3)
		merged = tf.summary.merge_all()

		'''task1'''
		print('Begin to train the network on task1...\n')
		with tf.device('/cpu:0'):
			source_data1 = h5py.File(data1_path, 'r')
			source_data1 = source_data1['data'][:]
			source_data1 = np.transpose(source_data1, (0, 3, 2, 1))
			print("source_data1 shape:", source_data1.shape)
		writer1 = tf.summary.FileWriter("logs/lam" + LAM_str + "/plot_1", sess.graph)
		train_task(model = model, sess = sess, merged = merged, writer = [writer1], saver = saver, c=c,
		           trainset = source_data1, save_path = './models/lam' + LAM_str + '/task1/', lam = LAM, task_ind = 1,
		           EPOCHES = EPOCHES[0])



		'''task2'''
		num_imgs = source_data1.shape[0]
		n_batches1 = int(num_imgs // model.batchsize)
		model.step = n_batches1 * EPOCHES[0]
		print('model step:', model.step)

		print('Begin to train the network on task2...\n')
		saver.restore(sess, './models_save/lam' + LAM_str + '/task1/' + str(n_batches1 * EPOCHES[0]) + '/' + str(
			n_batches1 * EPOCHES[0]) + '.ckpt')
		model.compute_fisher([source_data1], c, sess, num_samples = NUM)

		with tf.device('/cpu:0'):
			source_data2 = h5py.File(data2_path, 'r')
			source_data2 = source_data2['data'][:]
			source_data2 = np.transpose(source_data2, (0, 3, 2, 1))
			print("source_data2 shape:", source_data2.shape)

		writer2 = tf.summary.FileWriter("logs/lam" + LAM_str + "/plot_2", sess.graph)
		model.star()
		train_task(model = model, sess = sess, merged = merged, writer = [writer1, writer2], c=c,
		           validset = [source_data1], saver = saver, trainset = source_data2, save_path = './models/lam' + LAM_str+ '/task2/', lam = LAM, task_ind = 2,
		           EPOCHES = EPOCHES[1])



		'''task3'''
		num_imgs = source_data2.shape[0]
		n_batches2 = int(num_imgs // model.batchsize)
		model.step += n_batches2 * EPOCHES[1]
		print('model step:', model.step)
		print('Begin to train the network on task3...\n')
		saver.restore(sess, './models_save/lam' + LAM_str + '/task2/' + str(n_batches2 * EPOCHES[1]) + '/' + str(
			n_batches2 * EPOCHES[1]) + '.ckpt')
		model.compute_fisher([source_data1, source_data2], c, sess, num_samples = NUM)
		for v in range(len(model.F_accum)):
			print(model.F_accum[v])
		with tf.device('/cpu:0'):
			source_data3 = h5py.File(data3_path, 'r')
			source_data3 = source_data3['data'][:]
			source_data3 = np.transpose(source_data3, (0, 3, 2, 1))
			print("source_data3 shape:", source_data3.shape)

		writer3 = tf.summary.FileWriter("logs/lam" + LAM_str + "/plot_3", sess.graph)
		model.star()
		train_task(model = model, sess = sess, merged = merged, writer = [writer1, writer2, writer3],
		           validset = [source_data1, source_data2], saver = saver, c=c,
		           trainset = source_data3, save_path = './models/lam' + LAM_str + '/task3/', lam = LAM, task_ind = 3,
		           EPOCHES = EPOCHES[2])

		saver.restore(sess, './models_save/lam' + LAM_str + '/task3/852/852.ckpt')
		model.compute_fisher([source_data1, source_data2, source_data3], c, sess, num_samples = NUM)

if __name__ == '__main__':
	main()
