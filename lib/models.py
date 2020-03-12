import tensorflow as tf


class ModelsBuilder:
    def __init__(self, output_channels=3, image_height=256, image_width=256):
        self.output_channels = output_channels
        self.image_height = image_height
        self.image_width = image_width

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def build_generator(self, clear2fog=False, activation='sigmoid'):
        inputs = tf.keras.layers.Input(shape=[self.image_height, self.image_height, self.output_channels])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
            self.downsample(512, 4),  # (bs, 16, 16, 512)
            self.downsample(512, 4),  # (bs, 8, 8, 512)
            self.downsample(512, 4),  # (bs, 4, 4, 512)
            self.downsample(512, 4),  # (bs, 2, 2, 512)
            self.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4),  # (bs, 16, 16, 1024)
            self.upsample(256, 4),  # (bs, 32, 32, 512)
            self.upsample(128, 4),  # (bs, 64, 64, 256)
            self.upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1 if clear2fog else self.output_channels, 4,
                                               strides=2,
                                               padding='same',
                                               name='transmission_layer' if clear2fog else 'output_layer',
                                               kernel_initializer=initializer,
                                               activation=activation)  # (bs, 256, 256, 1)
        x = inputs

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
        if clear2fog:
            transmission = x
            from . import gauss
            transmission = gauss.gauss_blur_model([self.image_height, self.image_width, 1], name="gauss_blur")(transmission)

            x = tf.keras.layers.multiply([inputs, transmission])
            one_minus_t = tf.keras.layers.Lambda(lambda t: 1 - t, name='transmission_invert')(transmission)
            x = tf.keras.layers.add([x, one_minus_t])

        return tf.keras.Model(inputs=inputs, outputs=x)

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        down1 = self.downsample(64, 4, False)(inp)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
        return tf.keras.Model(inputs=inp, outputs=last)
