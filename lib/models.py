import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x, **kwargs):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class ModelsBuilder:
    def __init__(self, output_channels=3, image_height=256, image_width=256, normalized_input=True):
        self.output_channels = output_channels
        self.image_height = image_height
        self.image_width = image_width
        self.normalized_input = normalized_input

    def downsample(self, filters, size, norm_type='instancenorm', apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(tf.keras.layers.BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, norm_type='instancenorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def resize_conv(self, filters, kernel_size, strides=1, norm_type='instancenorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.UpSampling2D(2, interpolation='bilinear'))
        # result.add(tf.keras.layers.Lambda(
        #     lambda x: tf.image.resize(x, resize_to, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
        result.add(
            tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.LeakyReLU())

        return result

    def concatenate_image_and_intensity(self, image_input, intensity_input):
        intensity = tf.keras.layers.RepeatVector(self.image_height * self.image_height)(intensity_input)
        intensity = tf.keras.layers.Reshape((self.image_height, self.image_height, 1))(intensity)
        return tf.keras.layers.Concatenate(axis=-1)([image_input, intensity])

    # TODO: Check which is better, instancenorm or batchnorm
    def build_generator(self, use_transmission_map=False, use_gauss_filter=True, norm_type='instancenorm',
                        use_intensity=True, kernel_size=4,
                        use_resize_conv=False):
        image_input = tf.keras.layers.Input(shape=[self.image_height, self.image_height, self.output_channels])
        inputs = image_input
        x = image_input
        if use_intensity:
            intensity_input = tf.keras.layers.Input(shape=(1,))
            inputs = [image_input, intensity_input]
            x = self.concatenate_image_and_intensity(x, intensity_input)

        down_stack = [
            self.downsample(64, kernel_size, norm_type=norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            self.downsample(128, kernel_size, norm_type=norm_type),  # (bs, 64, 64, 128)
            self.downsample(256, kernel_size, norm_type=norm_type),  # (bs, 32, 32, 256)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 16, 16, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 8, 8, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 4, 4, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 2, 2, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 1, 1, 512)
        ]

        if use_resize_conv:
            up_stack = [
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
                self.resize_conv(512, kernel_size, norm_type=norm_type),  # (bs, 16, 16, 1024)
                self.resize_conv(256, kernel_size, norm_type=norm_type),  # (bs, 32, 32, 512)
                self.resize_conv(128, kernel_size, norm_type=norm_type),  # (bs, 64, 64, 256)
                self.resize_conv(64, kernel_size, norm_type=norm_type),  # (bs, 128, 128, 128)
            ]
        else:
            up_stack = [
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type),  # (bs, 16, 16, 1024)
                self.upsample(256, kernel_size, norm_type=norm_type),  # (bs, 32, 32, 512)
                self.upsample(128, kernel_size, norm_type=norm_type),  # (bs, 64, 64, 256)
                self.upsample(64, kernel_size, norm_type=norm_type),  # (bs, 128, 128, 128)
            ]

        initializer = tf.random_normal_initializer(0., 0.02)

        last = tf.keras.layers.Conv2DTranspose(1 if use_transmission_map else self.output_channels,
                                               kernel_size,
                                               strides=2,
                                               padding='same',
                                               name='transmission_layer' if use_transmission_map else 'output_layer',
                                               kernel_initializer=initializer,
                                               activation='tanh' if self.normalized_input else 'sigmoid')  # (bs, 256, 256, 1)
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        if use_transmission_map:
            transmission = x
            if self.normalized_input:
                transmission = tf.keras.layers.Lambda(lambda t: t * 0.5 + 0.5, name='fix_transmission_range')(
                    transmission)
            if use_gauss_filter:
                from . import gauss
                transmission = gauss.gauss_blur_model([self.image_height, self.image_width, 1], name="gauss_blur")(
                    transmission)

            x = tf.keras.layers.multiply([image_input, transmission])
            one_minus_t = tf.keras.layers.Lambda(lambda t: 1 - t, name='transmission_invert')(transmission)
            x = tf.keras.layers.add([x, one_minus_t])

        return tf.keras.Model(inputs=inputs, outputs=x)

    def build_discriminator(self, norm_type='instancenorm', use_intensity=True, kernel_size=4):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        inputs = inp
        x = inp
        if use_intensity:
            intensity_input = tf.keras.layers.Input(shape=(1,))
            inputs = [inp, intensity_input]
            x = self.concatenate_image_and_intensity(x, intensity_input)

        down1 = self.downsample(64, kernel_size, norm_type=norm_type, apply_norm=False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, kernel_size, norm_type=norm_type)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, kernel_size, norm_type=norm_type)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, kernel_size, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
        if norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)
        else:
            norm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, kernel_size, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
        return tf.keras.Model(inputs=inputs, outputs=last)
        # TODO: should add activation to last layer?
