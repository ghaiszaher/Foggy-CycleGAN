def plot_generators_predictions(model_clear2fog, test_input_clear, model_fog2clear, test_input_fog,
                                normalized_input=True):
    import matplotlib.pyplot as plt
    prediction_clear2fog = model_clear2fog(test_input_clear)
    prediction_fog2clear = model_fog2clear(test_input_fog)

    plt.figure(figsize=(12, 12))

    display_list = [test_input_clear[0], prediction_clear2fog[0], test_input_fog[0], prediction_fog2clear[0]]
    title = ['Clear', 'To Fog', 'Fog', 'To Clear']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        to_display = display_list[i]
        if normalized_input:
            to_display = to_display * 0.5 + 0.5
        plt.imshow(to_display)
        plt.axis('off')
    return plt


def plot_generators_predictions_v2(test_input_clear, prediction_clear2fog, test_input_fog, prediction_fog2clear,
                                   normalized_input=True):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))

    display_list = [test_input_clear[0], prediction_clear2fog[0], test_input_fog[0], prediction_fog2clear[0]]
    title = ['Clear', 'To Fog', 'Fog', 'To Clear']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        to_display = display_list[i]
        if normalized_input:
            to_display = to_display * 0.5 + 0.5
        plt.imshow(to_display)
        plt.axis('off')
    return plt


def plot_discriminators_predictions(discriminator_clear, sample_clear, discriminator_fog, sample_fog):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.title('Is real clear?')
    plt.imshow(discriminator_clear(sample_clear)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(1, 2, 2)
    plt.title('Is real fog?')
    plt.imshow(discriminator_fog(sample_fog)[0, ..., -1], cmap='RdBu_r')

    return plt


def plot_generators_and_discriminators_predictions(test_input_clear, prediction_clear2fog, test_input_fog,
                                                   prediction_fog2clear,
                                                   discriminator_clear_output, discriminator_fog_output,
                                                   discriminator_fakeclear_output, discriminator_fakefog_output,
                                                   normalized_input=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))

    display_list = [test_input_clear[0], discriminator_clear_output[0, ..., -1],
                    prediction_clear2fog[0], discriminator_fakefog_output[0, ..., -1],
                    test_input_fog[0], discriminator_fog_output[0, ..., -1],
                    prediction_fog2clear[0], discriminator_fakeclear_output[0, ..., -1]]
    title = ['Real Clear', 'Is real clear? Expected: Yes',
             'To Fog', 'Is real fog? Expected: No',
             'Fog', 'Is real clear? Expected: Yes',
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
            plt.colorbar()
    return plt


def get_generator_square_image(test_input_clear, prediction_clear2fog, test_input_fog, prediction_fog2clear,
                               normalized_input=True):
    import tensorflow as tf
    row1 = tf.concat((test_input_clear[0], prediction_clear2fog[0]), axis=1)
    row2 = tf.concat((test_input_fog[0], prediction_fog2clear[0]), axis=1)
    img = tf.concat((row1, row2), axis=0)
    if normalized_input:
        img = (img + 1) * 127.5
    else:
        img = img * 255
    img = tf.cast(img, tf.uint8)
    return img
