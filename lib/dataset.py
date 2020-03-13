import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetInitializer:
    def __init__(self, image_height=256, image_width=256, normalized_input=True):
        self.image_height = image_height
        self.image_width = image_width
        self.normalized_input = normalized_input

    def random_crop(self, image):
        cropped_image = tf.image.random_crop(
            image, size=[self.image_height, self.image_width, 3])

        return cropped_image

    def random_jitter(self, image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = self.random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def normalize_image(self, image):
        image = tf.cast(image, tf.float32)
        if self.normalized_input:
            return image/127.5-1
        return image / 255.

    def preprocess_image_train(self, image, label):
        image = self.normalize_image(image)
        image = self.random_jitter(image)
        return image

    def preprocess_image_test(self, image, label):
        image = self.normalize_image(image)
        return image

    def prepare_dataset(self, buffer_size, batch_size, AUTOTUNE=tf.data.experimental.AUTOTUNE):
        dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                                      with_info=True, as_supervised=True)

        train_A, train_B = dataset['trainA'], dataset['trainB']
        test_A, test_B = dataset['testA'], dataset['testB']

        train_A = train_A.map(
            self.preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
            buffer_size).batch(batch_size)

        train_B = train_B.map(
            self.preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
            buffer_size).batch(batch_size)

        test_A = test_A.map(
            self.preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            buffer_size).batch(batch_size)

        test_B = test_B.map(
            self.preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            buffer_size).batch(batch_size)

        return (train_A, train_B), (test_A, test_B)
