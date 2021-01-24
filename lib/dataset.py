import tensorflow as tf
import pandas as pd
import os
from .tools import df_length

COLUMN_PATH = 'path'
COLUMN_INTENSITY = 'intensity'
INTENSITY_VALUE_RANDOM = 'random'
INTENSITY_VALUE_SAMPLE = 'sample'


def split_dataframe(df, smaller_split_ratio, random_seed=None):
    import numpy as np
    if random_seed:
        np.random.seed(random_seed)
    random_indices = np.random.permutation(df_length(df))
    split_size = int(df_length(df) * smaller_split_ratio)
    return df.iloc[random_indices[split_size:]], df.iloc[
        random_indices[:split_size]]  # return larger_portion, smaller_portion


def image_names_generator(df, intensity_value=None, random_range=(0.1, 0.95)):
    """
    returns a generator that yields one row of the dataframe at a time.
    :param df:
    :param intensity_value: possible values: [None, 'random', 'sample'].
        None: the intensity value will be returned from the dataframe.
        'random': the intensity will be generated randomly in the range `random_range`
        'sample': generator will yield 9 rows values, each with an intensity from the range [0.1,0.9]
    :param random_range:
    :return:
    """

    def gen():
        for index, row in df.iterrows():
            path = row[COLUMN_PATH]
            if intensity_value == INTENSITY_VALUE_RANDOM:
                intensity = tf.random.uniform((1,), minval=random_range[0], maxval=random_range[1], dtype=tf.float32)
                intensity = tf.round(intensity * 100) / 100
                yield path, intensity
            elif intensity_value == INTENSITY_VALUE_SAMPLE:
                for i in range(1, 10):
                    intensity = tf.expand_dims(tf.cast(i / 10, tf.float32), axis=-1)
                    yield path, intensity
            else:
                intensity = tf.expand_dims(tf.cast(row[COLUMN_INTENSITY], tf.float32), axis=-1)
                yield path, intensity

    return gen


# noinspection PyMethodMayBeStatic
class DatasetInitializer:
    def __init__(self, image_height=256, image_width=256, channels=3, dataset_path='dataset/', normalized_input=True,
                 sample_images_path='sample_images/'):
        self.dataset_path = dataset_path
        self.sample_images_path = sample_images_path
        self.image_height = image_height
        self.image_width = image_width
        self.normalized_input = normalized_input
        self.channels = channels
        self.train_clear_df = None
        self.test_clear_df = None
        self.train_fog_df = None
        self.test_fog_df = None
        self.sample_clear_df = None
        self.sample_fog_df = None

    def preprocess_image_path(self, file_path, intensity):
        if str(file_path).lower().endswith('png'):
            return self.process_png_image_path(file_path, intensity)
        else:
            return self.process_jpeg_image_path(file_path, intensity)

    def process_jpeg_image_path(self, file_path, intensity):
        return tf.io.decode_jpeg(tf.io.read_file(file_path), channels=self.channels), intensity

    def process_png_image_path(self, file_path, intensity):
        return tf.io.decode_png(tf.io.read_file(file_path), channels=self.channels), intensity

    def resize_to_thumbnail(self, image, target_height, target_width, random_crop=False):
        """
        Acts like general thumbnail methods: resizes the image to [target_height,target_width] without
        altering the original images aspect ratio. First it it crops the image either in the center or randomly
        according to `random_crop` parameter, the cropped image will have same aspect ratio as the target image.
        Then the image will be resized to [target_height, target_width] using Nearest Neighbour method
        :param image: The image you want to resize
        :param target_height: The target height
        :param target_width: The target width
        :param random_crop: if True, the image will initially be randomly cropped, otherwise it will be cropped
            from the center
        :return:
        """
        if tf.equal(tf.size(image), 0):
            return image
        target_ratio = tf.cast(tf.divide(target_width, target_height), tf.float32)
        shape = tf.shape(image)
        original_height = tf.cast(shape[0], tf.int32)
        original_width = tf.cast(shape[1], tf.int32)
        original_height_f = tf.cast(original_height, tf.float32)
        original_width_f = tf.cast(original_width, tf.float32)
        # if original_height is None or original_width is None:
        #     return tf.image.resize(image, [target_width, target_height],
        #                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        ratio = tf.divide(original_width_f, original_height_f)
        if ratio > target_ratio:
            crop_width = tf.cast(tf.round(tf.multiply(original_height_f, target_ratio)), tf.int32)
            crop_height = original_height
        else:
            crop_width = original_width
            crop_height = tf.cast(tf.round(tf.divide(original_width_f, target_ratio)), tf.int32)

        if random_crop:
            image = tf.image.random_crop(
                image, size=[crop_height, crop_width, self.channels])
        else:
            # Crop in the center
            center = tf.cast(tf.divide(original_height, 2), tf.int32), tf.cast(tf.math.divide(original_width, 2),
                                                                               tf.int32)
            start = center[0] - tf.cast(tf.math.divide(crop_height, 2), tf.int32), center[1] - tf.cast(
                tf.math.divide(crop_width, 2), tf.int32)

            image = tf.image.crop_to_bounding_box(image, start[0], start[1], crop_height, crop_width)

        return tf.image.resize(image, [target_width, target_height],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def random_jitter(self, image):
        # resizing to 286 x 286 x 3
        jitter_offset = 30
        image = self.resize_to_thumbnail(image, self.image_height + jitter_offset, self.image_width + jitter_offset)

        # randomly cropping to 256 x 256 x 3
        image = tf.image.random_crop(
            image, size=[self.image_height, self.image_width, self.channels])

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def normalize_image_and_intensity(self, image, intensity):
        image = tf.cast(image, tf.float32)
        if self.normalized_input:
            return image / 127.5 - 1, intensity * 2 - 1
        return image / 255., intensity

    def preprocess_image_train(self, image, intensity):
        image, intensity = self.normalize_image_and_intensity(image, intensity)
        image = self.random_jitter(image)
        return image, intensity

    def preprocess_image_test(self, image, intensity):
        image, intensity = self.normalize_image_and_intensity(image, intensity)
        image = self.resize_to_thumbnail(image, self.image_height, self.image_width)
        return image, intensity

    def process_annotations_file(self, file_path):
        df = pd.read_csv(file_path, names=[COLUMN_PATH, COLUMN_INTENSITY])
        parent = os.path.dirname(file_path)
        df[COLUMN_PATH] = df[COLUMN_PATH].apply(lambda p: os.path.join(parent, p))
        return df

    def annotations_to_dataframe(self, path):
        import numpy as np
        images_df = None
        annotation_files = None
        # Because tf.io.matching_files doesn't walk recursively on linux, had to do it manually
        for d in tf.io.gfile.walk(path):
            files = tf.io.matching_files(os.path.join(d[0], "Annotations*.csv")).numpy()
            if annotation_files is not None:
                annotation_files = np.concatenate((annotation_files, files))
            else:
                annotation_files = files

        if annotation_files is None:
            raise Exception("No annotation files found!")

        for s in annotation_files:
            df = self.process_annotations_file(s.decode())
            if images_df is None:
                images_df = df
            else:
                images_df = images_df.append(df)

        if images_df is None:
            raise Exception("No images found!")

        return images_df

    def fill_train_test_dataframes(self, test_split=0.3, random_seed=None):
        images_df = self.annotations_to_dataframe(self.dataset_path)
        clear_df = images_df[images_df[COLUMN_INTENSITY] == 0]
        fog_df = images_df[images_df[COLUMN_INTENSITY] != 0]
        self.train_clear_df, self.test_clear_df = split_dataframe(clear_df, test_split, random_seed)
        self.train_fog_df, self.test_fog_df = split_dataframe(fog_df, test_split, random_seed)
        print("Found {} clear images and {} fog images".format(df_length(clear_df), df_length(fog_df)))
        print("Clear images split to {} train - {} test".format(df_length(self.train_clear_df),
                                                                df_length(self.test_clear_df)))
        print("Fog images split to {} train - {} test".format(df_length(self.train_fog_df),
                                                              df_length(self.test_fog_df)))

    def fill_sample_dataframes(self):
        images_df = self.annotations_to_dataframe(self.sample_images_path)
        self.sample_clear_df = images_df[images_df[COLUMN_INTENSITY] == 0]
        self.sample_fog_df = images_df[images_df[COLUMN_INTENSITY] != 0]
        print("Found {} sample clear image(s) and {} sample fog image(s)".format(df_length(self.sample_clear_df),
                                                                                 df_length(self.sample_fog_df)))

    def prepare_dataset(self, batch_size, buffer_size=1000,
                        test_split=0.3,
                        autotune=tf.data.experimental.AUTOTUNE,
                        return_sample=True, sample_batch_size=1,
                        random_seed=None):
        self.fill_train_test_dataframes(test_split, random_seed=random_seed)
        self.fill_sample_dataframes()

        train_clear_gen = image_names_generator(self.train_clear_df, intensity_value=INTENSITY_VALUE_RANDOM)
        train_fog_gen = image_names_generator(self.train_fog_df)
        test_clear_gen = image_names_generator(self.test_clear_df, intensity_value=INTENSITY_VALUE_RANDOM)
        test_fog_gen = image_names_generator(self.test_fog_df)
        sample_clear_gen = image_names_generator(self.sample_clear_df, intensity_value=INTENSITY_VALUE_SAMPLE)
        sample_fog_gen = image_names_generator(self.sample_fog_df)

        output_types = (tf.string, tf.float32)
        train_clear = tf.data.Dataset.from_generator(train_clear_gen, output_types).shuffle(buffer_size).map(
            self.preprocess_image_path)
        train_fog = tf.data.Dataset.from_generator(train_fog_gen, output_types).shuffle(buffer_size).map(
            self.preprocess_image_path)
        test_clear = tf.data.Dataset.from_generator(test_clear_gen, output_types).shuffle(buffer_size).map(
            self.preprocess_image_path)
        test_fog = tf.data.Dataset.from_generator(test_fog_gen, output_types).shuffle(buffer_size).map(
            self.preprocess_image_path)
        sample_clear = tf.data.Dataset.from_generator(sample_clear_gen, output_types).map(self.preprocess_image_path)
        sample_fog = tf.data.Dataset.from_generator(sample_fog_gen, output_types).map(self.preprocess_image_path)

        train_clear = train_clear.map(
            self.preprocess_image_train, num_parallel_calls=autotune).cache().batch(batch_size)

        train_fog = train_fog.map(
            self.preprocess_image_train, num_parallel_calls=autotune).cache().batch(batch_size)

        test_clear = test_clear.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().batch(batch_size)

        test_fog = test_fog.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().batch(batch_size)

        sample_clear = sample_clear.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().batch(sample_batch_size)

        sample_fog = sample_fog.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().batch(sample_batch_size)

        if return_sample:
            return (train_clear, train_fog), (test_clear, test_fog), (sample_clear, sample_fog)

        return (train_clear, train_fog), (test_clear, test_fog)
