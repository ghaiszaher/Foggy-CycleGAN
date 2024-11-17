def gauss_blur_model(input_shape, kernel_size=19, sigma=5, **kwargs):
    """
    Creates a Keras model that applies Gaussian blur to the input.

    Parameters:
    - input_shape (tuple): Shape of the input (height, width, channels).
    - kernel_size (int): Size of the Gaussian kernel.
    - sigma (float): Standard deviation of the Gaussian kernel.
    - **kwargs: Additional arguments for the model creation.

    Returns:
    - tf.keras.Model: A Keras model applying Gaussian blur to the input.
    """

    import tensorflow as tf
    from keras_cv.api.layers import RandomGaussianBlur

    inputs = tf.keras.Input(shape=input_shape)

    gaussian_blur = RandomGaussianBlur(
        kernel_size=kernel_size,
        factor=(sigma, sigma)
    )

    # Apply Gaussian blur
    blurred = gaussian_blur(inputs)

    # Create and return the model
    model = tf.keras.Model(inputs=inputs, outputs=blurred, **kwargs)
    return model
