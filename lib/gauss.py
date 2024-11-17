def gauss_blur_model(input_shape, kernel_size=19, sigma=5, **kwargs):
    import tensorflow as tf
    import numpy as np

    def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
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
        def __init__(self, padding=(1, 1),
                     data_format="channels_last", **kwargs):
            self.data_format = data_format
            self.padding = padding
            super(SymmetricPadding2D, self).__init__(**kwargs)

        def build(self, input_shape):
            super(SymmetricPadding2D, self).build(input_shape)

        def call(self, inputs, **kwargs):
            if self.data_format == "channels_last":
                # (batch, rows, cols, channels)
                pad = [[0, 0]] + [[p, p] for p in self.padding] + [[0, 0]]
            else:
                # (batch, channels, rows, cols)
                pad = [[0, 0], [0, 0]] + [[p, p] for p in self.padding]
            paddings = tf.constant(pad)
            return tf.pad(inputs, paddings, "REFLECT")

        def compute_output_shape(self, input_shape):
            if self.data_format == "channels_last":
                return (input_shape[0],
                        input_shape[1] + 2 * self.padding[0],
                        input_shape[2] + 2 * self.padding[1],
                        input_shape[3])
            else:
                return (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * self.padding[0],
                        input_shape[3] + 2 * self.padding[1])

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

    # Apply symmetric padding
    x = SymmetricPadding2D(padding=[p, p])(gauss_inputs)

    # Ensure the input to DepthwiseConv2D has the correct shape
    x = gauss_layer(x)

    # Set the weights for the gaussian filter
    gauss_layer.set_weights([kernel_weights])
    gauss_layer.trainable = False

    return tf.keras.Model(inputs=gauss_inputs, outputs=x, **kwargs)
