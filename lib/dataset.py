import tensorflow as tf
import pandas as pd
import os
from .tools import df_length

COLUMN_PATH = 'path'
COLUMN_INTENSITY = 'intensity'


# Helper functions:
# TODO: Delete
# def shuffle_dataframe(df):
#     import numpy as np
#     return df.iloc[np.random.permutation(df_length(df))]


def split_dataframe(df, smaller_split_ratio):
    split_size = int(df_length(df) * smaller_split_ratio)
    return df.iloc[split_size:], df.iloc[:split_size]  # return larger_portion, smaller_portion


def image_names_generator(df):
    def gen():
        for index, row in df.iterrows():
            yield row[COLUMN_PATH], row[COLUMN_INTENSITY]

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
        if tf.equal(tf.size(image), 0):
            return image
        target_ratio = tf.divide(target_width, target_height)
        shape = tf.shape(image)
        original_height = tf.cast(shape[0], tf.int32)
        original_width = tf.cast(shape[1], tf.int32)
        original_height_f = tf.cast(original_height, tf.float64)
        original_width_f = tf.cast(original_width, tf.float64)
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

    def normalize_image(self, image):
        image = tf.cast(image, tf.float32)
        if self.normalized_input:
            return image / 127.5 - 1
        return image / 255.

    def preprocess_image_train(self, image, intensity):
        image = self.normalize_image(image)
        image = self.random_jitter(image)
        # TODO: return intensity
        return image

    def preprocess_image_test(self, image, intensity):
        image = self.normalize_image(image)
        image = self.resize_to_thumbnail(image, self.image_height, self.image_width)
        # TODO: return intensity
        return image

    def process_annotations_file(self, file_path):
        df = pd.read_csv(file_path, names=[COLUMN_PATH, COLUMN_INTENSITY])
        parent = os.path.dirname(file_path)
        df[COLUMN_PATH] = df[COLUMN_PATH].apply(lambda p: os.path.join(parent, p))
        return df

    def annotations_to_dataframe(self, path):
        import numpy as np
        images_df = None
        files1 = tf.io.matching_files(os.path.join(path, "**/Annotations*.csv")).numpy()
        files2 = tf.io.matching_files(os.path.join(path, "Annotations*.csv")).numpy()
        files = np.concatenate((files1, files2))
        for s in files:
            df = self.process_annotations_file(s.decode())
            if images_df is None:
                images_df = df
            else:
                images_df = images_df.append(df)
        return images_df

    def fill_train_test_dataframes(self, test_split=0.3):
        images_df = self.annotations_to_dataframe(self.dataset_path)
        clear_df = images_df[images_df[COLUMN_INTENSITY] == 0]
        fog_df = images_df[images_df[COLUMN_INTENSITY] != 0]
        self.train_clear_df, self.test_clear_df = split_dataframe(clear_df, test_split)
        self.train_fog_df, self.test_fog_df = split_dataframe(fog_df, test_split)
        print("Found {} clear images and {} fog images".format(df_length(clear_df), df_length(fog_df)))
        print("Clear images split to {} train - {} test".format(df_length(self.train_clear_df),
                                                                df_length(self.test_clear_df)))
        print("Fog images split to {} train - {} test".format(df_length(self.train_fog_df),
                                                              df_length(self.test_fog_df)))

    def fill_sample_dataframes(self):
        images_df = self.annotations_to_dataframe(self.sample_images_path)
        self.sample_clear_df = images_df[images_df[COLUMN_INTENSITY] == 0]
        self.sample_fog_df = images_df[images_df[COLUMN_INTENSITY] != 0]
        print("Found {} sample clear images and {} sample fog images".format(df_length(self.sample_clear_df),
                                                                             df_length(self.sample_fog_df)))

    def prepare_dataset(self, buffer_size, batch_size,
                        test_split=0.3,
                        autotune=tf.data.experimental.AUTOTUNE,
                        return_sample=True):
        self.fill_train_test_dataframes(test_split)
        self.fill_sample_dataframes()

        train_clear_gen = image_names_generator(self.train_clear_df)
        train_fog_gen = image_names_generator(self.train_fog_df)
        test_clear_gen = image_names_generator(self.test_clear_df)
        test_fog_gen = image_names_generator(self.test_fog_df)
        sample_clear_gen = image_names_generator(self.sample_clear_df)
        sample_fog_gen = image_names_generator(self.sample_fog_df)

        output_types = (tf.string, tf.float64)
        train_clear = tf.data.Dataset.from_generator(train_clear_gen, output_types).map(self.preprocess_image_path)
        train_fog = tf.data.Dataset.from_generator(train_fog_gen, output_types).map(self.preprocess_image_path)
        test_clear = tf.data.Dataset.from_generator(test_clear_gen, output_types).map(self.preprocess_image_path)
        test_fog = tf.data.Dataset.from_generator(test_fog_gen, output_types).map(self.preprocess_image_path)
        sample_clear = tf.data.Dataset.from_generator(sample_clear_gen, output_types).map(self.preprocess_image_path)
        sample_fog = tf.data.Dataset.from_generator(sample_fog_gen, output_types).map(self.preprocess_image_path)

        train_clear = train_clear.map(
            self.preprocess_image_train, num_parallel_calls=autotune).cache().shuffle(
            buffer_size).batch(batch_size)

        train_fog = train_fog.map(
            self.preprocess_image_train, num_parallel_calls=autotune).cache().shuffle(
            buffer_size).batch(batch_size)

        test_clear = test_clear.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().shuffle(
            buffer_size).batch(batch_size)

        test_fog = test_fog.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().shuffle(
            buffer_size).batch(batch_size)

        sample_clear = sample_clear.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().batch(batch_size)

        sample_fog = sample_fog.map(
            self.preprocess_image_test, num_parallel_calls=autotune).cache().batch(batch_size)

        if return_sample:
            return (train_clear, train_fog), (test_clear, test_fog), (sample_clear, sample_fog)

        return (train_clear, train_fog), (test_clear, test_fog)
