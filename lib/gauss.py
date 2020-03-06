def gauss_blur_model(input_shape, kernel_size=19, sigma=5, **kwargs):
    import tensorflow as tf
    import numpy as np
    def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        #https://stackoverflow.com/questions/55643675/how-do-i-implement-gaussian-blurring-layer-in-keras
        #https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python/17201686#17201686
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    class SymmetricPadding2D(tf.keras.layers.Layer):
        # Source: https://stackoverflow.com/a/55210905/11394663
        def __init__(self, output_dim, padding=(1, 1),
                     data_format="channels_last", **kwargs):
            self.output_dim = output_dim
            self.data_format = data_format
            self.padding = padding
            super(SymmetricPadding2D, self).__init__(**kwargs)

        def build(self, input_shape):
            super(SymmetricPadding2D, self).build(input_shape)

        def call(self, inputs, **kwargs):
            if self.data_format is "channels_last":
                # (batch, depth, rows, cols, channels)
                pad = [[0, 0]] + [[i, i] for i in self.padding] + [[0, 0]]
            # elif self.data_format is "channels_first":
            else:
                # (batch, channels, depth, rows, cols)
                pad = [[0, 0], [0, 0]] + [[i, i] for i in self.padding]
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "REFLECT")
            return out

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.output_dim

    if kernel_size % 2 == 0:
        raise Exception("kernel size should be an odd number")
    gauss_inputs = tf.keras.layers.Input(shape=input_shape)

    kernel_weights = matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=sigma)
    in_channels = input_shape[-1]
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)  # apply the same filter on all the input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons
    gauss_layer = tf.keras.layers.DepthwiseConv2D(kernel_size, use_bias=False, padding='valid')
    p = (kernel_size - 1) // 2
    # noinspection PyCallingNonCallable
    x = SymmetricPadding2D(0, padding=[p, p])(gauss_inputs)
    x = gauss_layer(x)
    ########################
    gauss_layer.set_weights([kernel_weights])
    gauss_layer.trainable = False
    return tf.keras.Model(inputs=gauss_inputs, outputs=x, **kwargs)
