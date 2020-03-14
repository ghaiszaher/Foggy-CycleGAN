import tensorflow as tf


class Trainer:
    def __init__(self, generator_clear2fog, generator_fog2clear,
                 discriminator_fog, discriminator_clear, LAMBDA=10,
                 lr=2e-4, beta_1=0.5, normalized_input=True):
        self.LAMBDA = LAMBDA
        self.generator_clear2fog = generator_clear2fog
        self.generator_fog2clear = generator_fog2clear
        self.discriminator_fog = discriminator_fog
        self.discriminator_clear = discriminator_clear
        self.normalized_input = normalized_input
        self.tensorboard_summary_writer_clear = None
        self.tensorboard_summary_writer_fog = None
        # Losses
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_clear2fog_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.generator_fog2clear_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.discriminator_fog_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.discriminator_clear_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        # Checkpoint Manager
        self.weights_path = None
        self.tensorboard_baselogdir = 'tensorboard_logs'
        self.total_epochs = 0
        self.image_log_path = 'image_logs'

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
        generator_clear2fog_weights_path = os.path.join(self.weights_path, 'generator_clear2fog.h5')
        generator_fog2clear_weights_path = os.path.join(self.weights_path, 'generator_fog2clear.h5')
        discriminator_clear_weights_path = os.path.join(self.weights_path, 'discriminator_clear.h5')
        discriminator_fog_weights_path = os.path.join(self.weights_path, 'discriminator_fog.h5')
        models = [self.generator_clear2fog,
                  self.generator_fog2clear,
                  self.discriminator_clear,
                  self.discriminator_fog]
        paths = [generator_clear2fog_weights_path,
                 generator_fog2clear_weights_path,
                 discriminator_clear_weights_path,
                 discriminator_fog_weights_path]
        return models, paths

    def configure_checkpoint(self, weights_path):
        import os
        from . import tools
        self.weights_path = weights_path
        models, paths = self.get_models_and_paths()
        tools.create_dir(weights_path)
        for model, path in zip(models, paths):
            if os.path.isfile(path):
                model.load_weights(path)
                print("Weights loaded: {}".format(path))
            else:
                print("Not found: {}".format(path))

    def reset_tensorboard_writers(self):
        self.tensorboard_summary_writer_fog = None
        self.tensorboard_summary_writer_clear = None

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

    def epoch_callback(self, sample_test, plot_sample_generator, plot_sample_gen_and_disc,
                       save_sample_generator_output, save_sample_gen_and_disc_output):
        if sample_test is None:
            return
        if type(sample_test) is not list and type(sample_test) is not tuple:
            raise Exception("sample_test should be a list or tuple!")
        if len(sample_test) != 2:
            raise Exception("sample_test should contain 2 elements!")

        sample_clear = sample_test[0]
        sample_fog = sample_test[1]
        prediction_clear2fog = self.generator_clear2fog(sample_clear)
        prediction_fog2clear = self.generator_fog2clear(sample_fog)
        discriminator_clear_output = self.discriminator_clear(sample_clear)
        discriminator_fog_output = self.discriminator_fog(sample_fog)
        discriminator_fakeclear_output = self.discriminator_clear(prediction_fog2clear)
        discriminator_fakefog_output = self.discriminator_fog(prediction_clear2fog)
        from . import plot
        import os
        if plot_sample_gen_and_disc or save_sample_gen_and_disc_output:
            plt = plot.plot_generators_and_discriminators_predictions(sample_clear, prediction_clear2fog,
                                                                      sample_fog,
                                                                      prediction_fog2clear,
                                                                      discriminator_clear_output,
                                                                      discriminator_fog_output,
                                                                      discriminator_fakeclear_output,
                                                                      discriminator_fakefog_output,
                                                                      normalized_input=self.normalized_input)
            if plot_sample_gen_and_disc:
                plt.show()
            if save_sample_gen_and_disc_output:
                plt.savefig(
                    os.path.join(self.image_log_path, "gen_and_disc_output_epoch_{}.jpg".format(self.total_epochs)),
                    bbox_inches='tight', pad_inches=0)

        if plot_sample_generator:
            plot.plot_generators_predictions_v2(sample_clear, prediction_clear2fog, sample_fog,
                                                prediction_fog2clear, normalized_input=self.normalized_input).show()
        if save_sample_generator_output:
            img = plot.get_generator_square_image(sample_clear, prediction_clear2fog, sample_fog,
                                                  prediction_fog2clear,
                                                  normalized_input=self.normalized_input)
            tf.io.write_file(
                os.path.join(self.image_log_path, "gen_output_epoch_{}.jpg".format(self.total_epochs)),
                tf.io.encode_jpeg(img))

    def train(self, train_clear, train_fog, epochs=40, epoch_save_rate=1, progress_print_rate=10,
              clear_output_callback=None, use_tensorboard=False, sample_test=None, plot_sample_generator=False,
              plot_sample_gen_and_disc=True, save_sample_generator_output=True, save_sample_gen_and_disc_output=True):
        from lib.tools import print_with_timestamp
        import time
        import datetime
        import os

        # Create image log path if needed
        if save_sample_generator_output or save_sample_gen_and_disc_output:
            from . import tools
            tools.create_dir(self.image_log_path)

        # Configure tensorboard if not already configured
        if use_tensorboard and (self.tensorboard_summary_writer_clear is None or self.tensorboard_summary_writer_fog is None):
            tensorboard_logdir = os.path.join(self.tensorboard_baselogdir,
                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_logdir_clear = tensorboard_logdir + "-clear"
            tensorboard_logdir_fog = tensorboard_logdir + "-fog"
            if self.tensorboard_summary_writer_clear is None:
                self.tensorboard_summary_writer_clear = tf.summary.create_file_writer(logdir=tensorboard_logdir_clear)
            if self.tensorboard_summary_writer_clear is None:
                self.tensorboard_summary_writer_fog = tf.summary.create_file_writer(logdir=tensorboard_logdir_fog)

        length = "Unknown"
        for epoch in range(epochs):
            clear2fog_loss_total = fog2clear_loss_total = disc_clear_loss_total = disc_fog_loss_total = 0
            self.epoch_callback(sample_test, plot_sample_generator, plot_sample_gen_and_disc,
                                save_sample_generator_output, save_sample_gen_and_disc_output)

            start = time.time()
            dataset = tf.data.Dataset.zip((train_clear, train_fog))
            n = 0
            for image_clear, image_fog in dataset:
                # Train Step
                clear2fog_loss, fog2clear_loss, disc_clear_loss, disc_fog_loss = self.train_step(image_clear,
                                                                                                 image_fog)
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
                print_with_timestamp('Saving checkpoint for epoch {} (total {}) at {}'.format(epoch + 1,
                                                                                              self.total_epochs,
                                                                                              self.weights_path))
            print_with_timestamp('Time taken for epoch {} (total {}) is {} sec'.format(epoch + 1,
                                                                                       self.total_epochs,
                                                                                       time.time() - start))
            print_with_timestamp('clear2fog loss: {}, fog2clear loss: {}\n\tdisc_clear loss: {}, disc_fog loss: {}'
                                 .format(clear2fog_loss_total, fog2clear_loss_total, disc_clear_loss_total,
                                         disc_fog_loss_total))
            # Tensorboard
            if use_tensorboard:
                with self.tensorboard_summary_writer_clear.as_default():
                    tf.summary.scalar('generator', fog2clear_loss_total, step=self.total_epochs + 1,
                                      description='fog2clear loss')
                    tf.summary.scalar('discriminator', disc_clear_loss_total, step=self.total_epochs + 1,
                                      description='discriminator_clear loss')
                with self.tensorboard_summary_writer_fog.as_default():
                    tf.summary.scalar('generator', clear2fog_loss_total, step=self.total_epochs + 1,
                                      description='clear2fog loss')
                    tf.summary.scalar('discriminator', disc_fog_loss_total, step=self.total_epochs + 1,
                                      description='discriminator_fog loss')
                    # TODO: Add Graph to tensorboard: https://www.tensorflow.org/tensorboard/graphs
            self.total_epochs += 1

    if __name__ == 'main':
        pass
        # TODO: Perform training
