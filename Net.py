import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np


WEIGHT_INIT_STDDEV = 0.05

n = 44

class Generator(object):

	def __init__(self, sco):
		self.encoder = Encoder(sco)
		self.decoder = Decoder(sco)
		self.var_list = []
		self.features = []

	def transform(self, I1, I2, is_training, reuse):
		img = tf.concat([I1, I2], 3)
		code = self.encoder.encode(img, is_training, reuse)
		generated_img = self.decoder.decode(code, is_training, reuse)
		# self.var_list.extend(self.encoder.var_list)
		# self.var_list.extend(self.decoder.var_list)
		# self.var_list.extend(tf.trainable_variables())
		return generated_img


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.var_list = []
		self.weight_vars = []

		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(2, n, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(n, n, 3, scope = 'dense_block_conv1'))
				self.weight_vars.append(self._create_variables(n*2, n, 3, scope = 'dense_block_conv2'))
				self.weight_vars.append(self._create_variables(n*3, n, 3, scope = 'dense_block_conv3'))
				self.weight_vars.append(self._create_variables(n*4, n, 3, scope = 'dense_block_conv4'))
				self.weight_vars.append(self._create_variables(n * 5, n, 3, scope = 'dense_block_conv5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
			self.var_list.append(kernel)
			self.var_list.append(bias)
		return (kernel, bias)

	def encode(self, image, is_training, reuse):
		dense_indices = [1, 2, 3, 4, 5]
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense = True, use_lrelu = True, is_training = is_training, reuse = reuse,
				             Scope = self.scope + '/encoder/b' + str(i))
			else:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = True, is_training = is_training,
				             reuse = reuse, Scope = self.scope + '/encoder/b' + str(i))
		return out


class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.var_list = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(n*6, 128, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(64, 32, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_4'))
				# self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_4'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
			self.var_list.append(kernel)
			self.var_list.append(bias)
		return (kernel, bias)

	def decode(self, image, is_training, reuse):
		final_layer_idx = len(self.weight_vars) - 1

		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = False,
				             Scope = self.scope + '/decoder/b' + str(i),  is_training = is_training, reuse=reuse)
				out = tf.nn.tanh(out) / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_lrelu = True,
				             Scope = self.scope + '/decoder/b' + str(i), is_training = is_training, reuse=reuse)
		return out



# class weightnet(object):
# 	def __init__(self, scope_name):
# 		self.weight_vars = []
# 		self.var_list = []
# 		self.scope = scope_name
# 		with tf.name_scope(scope_name):
# 			with tf.variable_scope('weightnet'):
# 				self.weight_vars.append(self._create_variables(2, 32, 3, scope = 'conv1'))
# 				self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv2'))
# 				self.weight_vars.append(self._create_variables(64, 96, 3, scope = 'conv3'))
# 				self.weight_vars.append(self._create_variables(96, 2, 3, scope = 'conv4'))
#
# 	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
# 		with tf.variable_scope(scope):
# 			shape = [kernel_size, kernel_size, input_filters, output_filters]
# 			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
# 			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
# 			self.var_list.append(kernel)
# 			self.var_list.append(bias)
# 		return (kernel, bias)
#
# 	def generate_weight(self, image, is_training):
# 		final_layer_idx = len(self.weight_vars) - 1
#
# 		out = image
# 		for i in range(len(self.weight_vars)):
# 			kernel, bias = self.weight_vars[i]
# 			if i == final_layer_idx:
# 				out = conv2d(out, kernel, bias, use_relu = False,
# 				             Scope = self.scope + '/generate_weight/b' + str(i), BN = False, stride=2, is_training = is_training, reuse=reuse)
# 				out = tf.reduce_mean(out, [1, 2], name='pool', keep_dims=True)
# 				b, _, _, _ = out.shape
# 				out = tf.reshape(out, [int(b), 2])
# 				out = tf.nn.softmax(out)
# 			else:
# 				out = conv2d(out, kernel, bias, use_relu = True, BN = True, stride=2,
# 				             Scope = self.scope + '/generate_weight/b' + str(i), is_training = is_training, reuse=reuse)
# 			# print("%s: out shape:" % i, out.shape)
# 		return out


def conv2d(x, kernel, bias, use_lrelu = True, dense = False, Scope = None, stride = 1, is_training = False, reuse=False):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(input = x_padded, filter = kernel, strides = [1, stride, stride, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	# if BN:
	# 	with tf.variable_scope(Scope):
	# 		# print("Scope", Scope)
	# 		# print("reuse", not is_training)
	# 		# out = tf.contrib.layers.batch_norm(out, decay = 0.9, updates_collections = None, epsilon = 1e-5, scale = True, reuse = reuse)
	#
	# 		out = tf.layers.batch_normalization(out, training = is_training, reuse= reuse, trainable=is_training)
	if use_lrelu:
		# out = tf.nn.relu(out)
		out = tf.maximum(out, 0.2 * out)
	if dense:
		out = tf.concat([out, x], 3)
	return out
