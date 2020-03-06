def plot_generators_predictions(model_clear2fog, test_input_clear, model_fog2clear, test_input_fog):
    import matplotlib.pyplot as plt
    prediction_clear2fog = model_clear2fog(test_input_clear)
    prediction_fog2clear = model_fog2clear(test_input_fog)

    plt.figure(figsize=(12, 12))

    display_list = [test_input_clear[0], prediction_clear2fog[0], test_input_fog[0], prediction_fog2clear[0]]
    title = ['Clear', 'To Fog', 'Fog', 'To Clear']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()



def plot_discriminators_predictions(discriminator_clear, sample_clear, discriminator_fog, sample_fog):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))

    plt.subplot(1,2,1)
    plt.title('Is real clear?')
    plt.imshow(discriminator_clear(sample_clear)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(1,2,2)
    plt.title('Is real fog?')
    plt.imshow(discriminator_fog(sample_fog)[0, ..., -1], cmap='RdBu_r')

    plt.show()