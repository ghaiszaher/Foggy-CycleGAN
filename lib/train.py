import tensorflow as tf


class Trainer:
    def __init__(self, generator_clear2fog, generator_fog2clear,
                 discriminator_fog, discriminator_clear, LAMBDA=10,
                 lr=2e-4, beta_1=0.5):
        self.LAMBDA = LAMBDA
        self.generator_clear2fog = generator_clear2fog
        self.generator_fog2clear = generator_fog2clear
        self.discriminator_fog = discriminator_fog
        self.discriminator_clear = discriminator_clear
        # Losses
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_clear2fog_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.generator_fog2clear_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.discriminator_fog_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.discriminator_clear_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        # Checkpoint Manager
        self.weights_path = None
        self.tensorboard_baselogdir = 'tensorboard_logs'

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def get_models_and_paths(self):
        import os
        generator_clear2fog_weights_path = os.path.join(self.weights_path, 'generator_clear2fog.weights')
        generator_fog2clear_weights_path = os.path.join(self.weights_path, 'generator_fog2clear.weights')
        discriminator_clear_weights_path = os.path.join(self.weights_path, 'discriminator_clear.weights')
        discriminator_fog_weights_path = os.path.join(self.weights_path, 'discriminator_fog.weights')
        models = [self.generator_clear2fog,
                  self.generator_fog2clear,
                  self.discriminator_clear,
                  self.discriminator_fog]
        paths = [generator_clear2fog_weights_path,
                 generator_fog2clear_weights_path,
                 discriminator_clear_weights_path,
                 discriminator_fog_weights_path]
        return models, paths

    def configure_checkpoint(self, weights_path, max_to_keep=1):
        import os
        self.weights_path = weights_path
        models, paths = self.get_models_and_paths()
        # TODO: Create the directory recursively if it doesn't exist
        for model, path in zip(models, paths):
            if os.path.isfile(path):
                model.load_weights(path)
                print("Weights loaded: {}".format(path))
            else:
                print("Not found: {}".format(path))

    def save_weights(self):
        models, paths = self.get_models_and_paths()
        for model, path in zip(models, paths):
            model.save_weights(path)

    @tf.function
    def train_step(self, real_clear, real_fog):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            fake_fog = self.generator_clear2fog(real_clear, training=True)
            cycled_clear = self.generator_fog2clear(fake_fog, training=True)

            fake_clear = self.generator_fog2clear(real_fog, training=True)
            cycled_fog = self.generator_clear2fog(fake_clear, training=True)

            # same_x and same_y are used for identity loss.
            same_clear = self.generator_fog2clear(real_clear, training=True)
            same_fog = self.generator_clear2fog(real_fog, training=True)

            disc_real_clear = self.discriminator_clear(real_clear, training=True)
            disc_real_fog = self.discriminator_fog(real_fog, training=True)

            disc_fake_clear = self.discriminator_clear(fake_clear, training=True)
            disc_fake_fog = self.discriminator_fog(fake_fog, training=True)

            # calculate the loss
            gen_clear2fog_loss = self.generator_loss(disc_fake_fog)
            gen_fog2clear_loss = self.generator_loss(disc_fake_clear)

            total_cycle_loss = self.calc_cycle_loss(real_clear, cycled_clear) + self.calc_cycle_loss(real_fog,
                                                                                                     cycled_fog)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_clear2fog_loss = gen_clear2fog_loss + total_cycle_loss + self.identity_loss(real_fog, same_fog)
            total_gen_fog2clear_loss = gen_fog2clear_loss + total_cycle_loss + self.identity_loss(real_clear,
                                                                                                  same_clear)
            disc_clear_loss = self.discriminator_loss(disc_real_clear, disc_fake_clear)
            disc_fog_loss = self.discriminator_loss(disc_real_fog, disc_fake_fog)

        # Calculate the gradients for generator and discriminator
        generator_clear2fog_gradients = tape.gradient(total_gen_clear2fog_loss,
                                                      self.generator_clear2fog.trainable_variables)
        generator_fog2clear_gradients = tape.gradient(total_gen_fog2clear_loss,
                                                      self.generator_fog2clear.trainable_variables)
        discriminator_clear_gradients = tape.gradient(disc_clear_loss,
                                                      self.discriminator_clear.trainable_variables)
        discriminator_fog_gradients = tape.gradient(disc_fog_loss,
                                                    self.discriminator_fog.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_clear2fog_optimizer.apply_gradients(zip(generator_clear2fog_gradients,
                                                               self.generator_clear2fog.trainable_variables))
        self.generator_fog2clear_optimizer.apply_gradients(zip(generator_fog2clear_gradients,
                                                               self.generator_fog2clear.trainable_variables))
        self.discriminator_clear_optimizer.apply_gradients(zip(discriminator_clear_gradients,
                                                               self.discriminator_clear.trainable_variables))
        self.discriminator_fog_optimizer.apply_gradients(zip(discriminator_fog_gradients,
                                                             self.discriminator_fog.trainable_variables))

        return total_gen_clear2fog_loss, total_gen_fog2clear_loss, disc_clear_loss, disc_fog_loss

    def train(self, train_clear, train_fog, epochs=40, epoch_save_rate=1, progress_print_rate=10, epoch_callback=None,
              clear_output_callback=None, use_tensorboard=False):
        from lib.tools import print_with_timestamp
        import time
        import datetime
        import os

        summary_writer = None
        if use_tensorboard:
            tensorboard_logdir = os.path.join(self.tensorboard_baselogdir,
                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            summary_writer = tf.summary.create_file_writer(logdir=tensorboard_logdir)
        length = "Unknown"
        for epoch in range(epochs):
            clear2fog_loss_total = fog2clear_loss_total = disc_clear_loss_total = disc_fog_loss_total = 0
            start = time.time()
            n = 0
            # Using a consistent image (sample_clear) so that the progress of the model
            # is clearly visible.
            if epoch_callback is not None:
                epoch_callback()
            dataset = tf.data.Dataset.zip((train_clear, train_fog))
            for image_clear, image_fog in dataset:
                # Train Step
                clear2fog_loss, fog2clear_loss, disc_clear_loss, disc_fog_loss = self.train_step(image_clear, image_fog)
                # Update Epoch's losses
                clear2fog_loss_total += clear2fog_loss
                fog2clear_loss_total += fog2clear_loss
                disc_clear_loss_total += disc_clear_loss
                disc_fog_loss_total += disc_fog_loss
                # Print Progress
                if n % progress_print_rate == 0:
                    print_with_timestamp('{}/{}'.format(n, length))
                n += 1
            length = n
            if clear_output_callback is not None:
                clear_output_callback()

            # Save weights
            if self.weights_path is not None and epoch_save_rate is not None and (
                    epoch + 1) % epoch_save_rate == 0:
                self.save_weights()
                print_with_timestamp('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                                   self.weights_path))
            print_with_timestamp('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                                            time.time() - start))
            print_with_timestamp('clear2fog loss: {}, fog2clear loss: {}\n\tdisc_clear loss: {}, disc_fog loss: {}'
                                 .format(clear2fog_loss_total, fog2clear_loss_total, disc_clear_loss_total,
                                         disc_fog_loss_total))
            # Tensorflow Board
            if use_tensorboard:
                with summary_writer.as_default():
                    tf.summary.scalar('generator/clear2fog_loss', clear2fog_loss_total, step=epoch + 1)
                    tf.summary.scalar('generator/fog2clear_loss', fog2clear_loss_total, step=epoch + 1)
                    tf.summary.scalar('discriminator/disc_clear_loss', disc_clear_loss_total, step=epoch + 1)
                    tf.summary.scalar('discriminator/disc_fog_loss', disc_fog_loss_total, step=epoch + 1)
                # TODO: Add Graph to tensorboard: https://www.tensorflow.org/tensorboard/graphs


if __name__ == 'main':
    pass
    # TODO: Perform training
