from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Input
from tensorflow.keras.layers import Dropout, Dense

from tensorflow.keras.models import Model


def discriminator(input_shape=(64, 64, 3), n_classes=2):
	init = TruncatedNormal(mean=0.0, stddev=0.02)
	in_image = Input(shape=input_shape)

	def conv_block(x, filter_size, kernel_size=(3, 3), strides=(2, 2), dropout=None):
		x = Conv2D(
			filters=filter_size,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			kernel_initializer=init)(x)
		x = LeakyReLU(alpha=0.2)(x)
		if dropout:
			x = Dropout(dropout)(x)
		return x

	# Downsample
	x = conv_block(in_image, filter_size=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.5)
	x = conv_block(x, filter_size=128, kernel_size=(3, 3), strides=(1, 1), dropout=0.5)
	x = conv_block(x, filter_size=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.5)
	x = conv_block(x, filter_size=256, kernel_size=(3, 3), strides=(1, 1), dropout=0.5)
	x = conv_block(x, filter_size=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.5)

	# flatten feature maps
	features = Flatten()(x)

	# real/fake output
	real_fake_output = Dense(1, activation='sigmoid', name='generation')(features)

	# class label output
	class_label_output = Dense(n_classes, activation='softmax', name='auxiliary')(features)

	# define model
	model = Model(
		inputs=in_image,
		outputs=[real_fake_output, class_label_output],
		name='discriminator')
	return model
