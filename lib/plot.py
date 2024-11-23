def get_images_and_intensities(test_input_clear, test_input_fog, normalized_input=True):
    image_clear = test_input_clear[0][0]
    intensity_clear = test_input_clear[1][0][0]
    image_fog = test_input_fog[0][0]
    intensity_fog = test_input_fog[1][0][0]
    if normalized_input:
        intensity_clear = intensity_clear * 0.5 + 0.5
        intensity_fog = intensity_fog * 0.5 + 0.5
    return image_clear, intensity_clear, image_fog, intensity_fog


def plot_generators_predictions(model_clear2fog, test_input_clear, model_fog2clear, test_input_fog,
                                normalized_input=True, close_fig=False):
    import matplotlib.pyplot as plt
    prediction_clear2fog = model_clear2fog(test_input_clear)
    prediction_fog2clear = model_fog2clear(test_input_fog)

    image_clear, intensity_clear, image_fog, intensity_fog = get_images_and_intensities(test_input_clear,
                                                                                        test_input_fog,
                                                                                        normalized_input)
    fig = plt.figure(figsize=(12, 12))

    display_list = [image_clear, prediction_clear2fog[0], image_fog, prediction_fog2clear[0]]
    title = ['Clear', 'To Fog {:0.2}'.format(intensity_clear), 'Fog {:0.2}'.format(intensity_fog), 'To Clear']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        to_display = display_list[i]
        if normalized_input:
            to_display = to_display * 0.5 + 0.5
        plt.imshow(to_display)
        plt.axis('off')

    if close_fig:
        plt.close(fig)
    return fig


def plot_generators_predictions_v2(test_input_clear, prediction_clear2fog, test_input_fog, prediction_fog2clear,
                                   normalized_input=True, close_fig=False):
    import matplotlib.pyplot as plt

    image_clear, intensity_clear, image_fog, intensity_fog = get_images_and_intensities(test_input_clear,
                                                                                        test_input_fog,
                                                                                        normalized_input)
    fig = plt.figure(figsize=(12, 12))

    display_list = [image_clear, prediction_clear2fog[0], image_fog, prediction_fog2clear[0]]
    title = ['Clear', 'To Fog {:0.2}'.format(intensity_clear), 'Fog {:0.2}'.format(intensity_fog), 'To Clear']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        to_display = display_list[i]
        if normalized_input:
            to_display = to_display * 0.5 + 0.5
        plt.imshow(to_display)
        plt.axis('off')
    if close_fig:
        plt.close(fig)
    return fig


def plot_discriminators_predictions(discriminator_clear, sample_clear, discriminator_fog, sample_fog,
                                    use_intensity_for_fog_discriminator=False, close_fig=False):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))

    sample_fog_input = sample_fog if use_intensity_for_fog_discriminator else sample_fog[0]
    sample_clear_input = sample_clear[0]

    plt.subplot(1, 2, 1)
    plt.title('Is real clear?')
    plt.imshow(discriminator_clear(sample_clear_input)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(1, 2, 2)
    plt.title('Is real fog?')
    plt.imshow(discriminator_fog(sample_fog_input)[0, ..., -1], cmap='RdBu_r')

    if close_fig:
        plt.close(fig)
    return fig


def plot_generators_and_discriminators_predictions(test_input_clear, prediction_clear2fog, test_input_fog,
                                                   prediction_fog2clear,
                                                   discriminator_clear_output, discriminator_fog_output,
                                                   discriminator_fakeclear_output, discriminator_fakefog_output,
                                                   normalized_input=True, close_fig=False):
    import matplotlib.pyplot as plt

    image_clear, intensity_clear, image_fog, intensity_fog = get_images_and_intensities(test_input_clear,
                                                                                        test_input_fog,
                                                                                        normalized_input)

    fig = plt.figure(figsize=(20, 10))

    display_list = [image_clear, discriminator_clear_output[0, ..., -1],
                    prediction_clear2fog[0], discriminator_fakefog_output[0, ..., -1],
                    image_fog, discriminator_fog_output[0, ..., -1],
                    prediction_fog2clear[0], discriminator_fakeclear_output[0, ..., -1]]
    title = ['Real Clear', 'Is real clear? Expected: Yes',
             'To Fog {:0.2}'.format(intensity_clear), 'Is real fog? Expected: No',
             'Fog {:0.2}'.format(intensity_fog), 'Is real fog? Expected: Yes',
             'To Clear', 'Is real clear? Expected: No']
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.title(title[i])
        plt.axis('off')
        to_display = display_list[i]
        if i % 2 == 0:
            if normalized_input:
                to_display = to_display * 0.5 + 0.5
            plt.imshow(to_display)
        else:
            plt.imshow(to_display, cmap='RdBu_r')
            plt.clim(-4, 4)
            plt.colorbar()
    if close_fig:
        plt.close(fig)
    return fig


def get_generator_square_image(test_input_clear, prediction_clear2fog, test_input_fog, prediction_fog2clear,
                               normalized_input=True):
    import tensorflow as tf

    image_clear, intensity_clear, image_fog, intensity_fog = get_images_and_intensities(test_input_clear,
                                                                                        test_input_fog,
                                                                                        normalized_input)

    row1 = tf.concat((image_clear, prediction_clear2fog[0]), axis=1)
    row2 = tf.concat((image_fog, prediction_fog2clear[0]), axis=1)
    img = tf.concat((row1, row2), axis=0)
    if normalized_input:
        img = (img + 1) * 127.5
    else:
        img = img * 255
    img = tf.cast(img, tf.uint8)
    return img


def plot_clear2fog_intensity(model_clear2fog, image_clear, intensity=0.5,
                             normalized_input=True, close_fig=False):
    import matplotlib.pyplot as plt
    import tensorflow as tf

    original_intensity = intensity
    if normalized_input:
        intensity = intensity * 2 - 1
    intensity = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(intensity), 0), 0)
    prediction_clear2fog = model_clear2fog((tf.expand_dims(image_clear, 0), intensity))

    fig = plt.figure(figsize=(12, 6))

    display_list = [image_clear, prediction_clear2fog[0]]
    if normalized_input:
        display_list = [item * 0.5 + 0.5 for item in display_list]
    title = ['Clear', 'To Fog {:0.2}'.format(original_intensity)]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        to_display = display_list[i]
        plt.imshow(to_display)
        plt.axis('off')

    if close_fig:
        plt.close(fig)
    return fig, display_list[1]
