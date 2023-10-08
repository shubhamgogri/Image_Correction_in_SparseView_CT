from glob import glob
import tensorflow as tf

import os
import time
import datetime
from matplotlib import pyplot as plt
# from IPython import display
import model
import random

from keras.applications.vgg16 import VGG16

BATCH_SIZE = 4
TEST_BATCH_SIZE = 1 
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256



def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image



def load_data(low_light_image_path, enhanced_image_path):
    low_light_image = model.read_image(low_light_image_path)
    enhanced_image = model.read_image(enhanced_image_path)
    res_low, res_enhanced = resize(low_light_image, enhanced_image, 256,256)
    # low_light_image, enhanced_image = normalize(res_low, res_enhanced)
    return res_low, res_enhanced


def get_dataset(low_light_images, enhanced_images, test):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, enhanced_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    if test:
        return dataset.batch(TEST_BATCH_SIZE, drop_remainder=True)
    return dataset.batch(BATCH_SIZE, drop_remainder=True)
    


"""## Build the generator

The generator of your pix2pix cGAN is a _modified_ [U-Net](https://arxiv.org/abs/1505.04597){:.external}. A U-Net consists of an encoder (downsampler) and decoder (upsampler). (You can find out more about it in the [Image segmentation](../images/segmentation.ipynb) tutorial and on the [U-Net project website](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/){:.external}.)

- Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
- Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
- There are skip connections between the encoder and decoder (as in the U-Net).

Define the downsampler (encoder):
"""

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)

"""Define the upsampler (decoder):"""

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# up_model = upsample(3, 4)
# up_result = up_model(down_result)
# print (up_result.shape)

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


"""### Define the generator loss

GANs learn a loss that adapts to the data, while cGANs learn a structured loss that penalizes a possible structure that differs from the network output and the target image, as described in the [pix2pix paper](https://arxiv.org/abs/1611.07004){:.external}.

- The generator loss is a sigmoid cross-entropy loss of the generated images and an **array of ones**.
- The pix2pix paper also mentions the L1 loss, which is a MAE (mean absolute error) between the generated image and the target image.
- This allows the generated image to become structurally similar to the target image.
- The formula to calculate the total generator loss is `gan_loss + LAMBDA * l1_loss`, where `LAMBDA = 100`. 
This value was decided by the authors of the paper.
"""

def wassertian_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)

def charbonnier_loss(y_true, y_pred, epsilon = 1e-3):
    err = y_true - y_pred
    loss = tf.sqrt( err**2 + epsilon**2 )
    return tf.reduce_mean(loss)

def loss_wass_char(vggModel, disc_real, disc_gen, gen_output_img, target):
    LAMBDA_ADV = 0.5
    LAMBDA_PERC = 0.3
    
    wassertian = wassertian_loss(disc_real, disc_gen)

    charbonnier = charbonnier_loss(target, gen_output_img)

    perceptual = perpetual_loss(target= target, gen_img= gen_output_img, vggModel= vggModel)

    total_generator_loss = LAMBDA_ADV * wassertian + LAMBDA_PERC * perceptual + ( 1 - LAMBDA_ADV - LAMBDA_PERC) * charbonnier
    

    disc_loss = -wassertian

    return disc_loss, total_generator_loss, wassertian, charbonnier, perceptual

def vgg_model():
    vgg_model = VGG16(False, weights= "imagenet", input_shape= (IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS))    
    vgg_model.trainable = False

    model = tf.keras.Model(inputs= vgg_model.input, outputs = vgg_model.get_layer('block4_conv3').output  )
    return model
    
    
def perpetual_loss(gen_img, target, vggModel):
    features_target = vggModel(target)
    features_gen = vggModel(gen_img)
    
    loss = tf.reduce_mean(tf.square(features_target - features_gen))

    return loss


@tf.function
def train_step_wass_char(input_image, target, step, generator, discriminator, summary_writer, vgg_model):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)


        disc_loss, total_generator_loss, wassertian, charbonnier , perpetual_loss = loss_wass_char(vgg_model, disc_real_output,disc_generated_output,gen_output,target)
        
        # gen_total_loss, gen_adv_loss, gen_ssim_loss, gen_psnr_loss = generator_loss(disc_generated_output, gen_output, target)
        # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(total_generator_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

                                           
    psnr_loss = tf.reduce_mean(model.peak_signal_noise_ratio(target , gen_output))
    print(psnr_loss)

    ssim_loss = tf.reduce_mean(model.structural_similarity(target , gen_output))
    print(ssim_loss)

    with summary_writer.as_default():
        tf.summary.scalar('GAN total loss', total_generator_loss, step=step )
        tf.summary.scalar('Generator WASSERSTEIN loss', wassertian, step=step)
        tf.summary.scalar('Generator CHARBONNIER loss', charbonnier, step=step)
        tf.summary.scalar('Generator PERCEPTUAL(VGG16) loss', perpetual_loss, step=step)
        tf.summary.scalar('Generator PSNR loss', psnr_loss, step=step)
        tf.summary.scalar('Generator SSIM loss', ssim_loss, step=step)
        tf.summary.scalar('Discriminator loss', disc_loss, step= step)    

def generator_loss(disc_generated_output, gen_output, target):
    LAMBDA_PSNR = 1
    LAMBDA_SSIM = 0.9

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    psnr_loss = tf.reduce_mean(model.peak_signal_noise_ratio(target , gen_output))
    print(psnr_loss)

    ssim_loss = tf.reduce_mean(model.structural_similarity(target , gen_output))
    print(ssim_loss)
    
    # # Mean absolute error
    # l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA_PSNR * psnr_loss) + (LAMBDA_SSIM * ssim_loss) 

    return total_gen_loss, gan_loss, ssim_loss, psnr_loss

"""The training procedure for the generator is as follows:

![Generator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gen.png?raw=1)

## Build the discriminator

The discriminator in the pix2pix cGAN is a convolutional PatchGAN classifier—it tries to classify if each image _patch_ is real or not real, as described in the [pix2pix paper](https://arxiv.org/abs/1611.07004){:.external}.

- Each block in the discriminator is: Convolution -> Batch normalization -> Leaky ReLU.
- The shape of the output after the last layer is `(batch_size, 30, 30, 1)`.
- Each `30 x 30` image patch of the output classifies a `70 x 70` portion of the input image.
- The discriminator receives 2 inputs:
    - The input image and the target image, which it should classify as real.
    - The input image and the generated image (the output of the generator), which it should classify as fake.
    - Use `tf.concat([inp, tar], axis=-1)` to concatenate these 2 inputs together.

Let's define the discriminator:
"""

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
  
  down2 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down3 = downsample(128, 4)(down2)  # (batch_size, 64, 64, 128)
  down4 = downsample(256, 4)(down3)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

"""Visualize the discriminator model architecture:"""

# discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

"""Test the discriminator:"""

# disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()

"""### Define the discriminator loss

- The `discriminator_loss` function takes 2 inputs: **real images** and **generated images**.
- `real_loss` is a sigmoid cross-entropy loss of the **real images** and an **array of ones(since these are the real images)**.
- `generated_loss` is a sigmoid cross-entropy loss of the **generated images** and an **array of zeros (since these are the fake images)**.
- The `total_loss` is the sum of `real_loss` and `generated_loss`.
"""

def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


    """## Training

- For each example input generates an output.
- The discriminator receives the `input_image` and the generated image as the first input. The second input is the `input_image` and the `target_image`.
- Next, calculate the generator and the discriminator loss.
- Then, calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
- Finally, log the losses to TensorBoard.
"""


@tf.function
def train_step(input_image, target, step, generator, discriminator, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_adv_loss, gen_ssim_loss, gen_psnr_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('GAN total loss', gen_total_loss, step=step )
        tf.summary.scalar('Generator ADVERSARIAL loss', gen_adv_loss, step=step)
        tf.summary.scalar('Generator PSNR loss', gen_psnr_loss, step=step)
        tf.summary.scalar('Generator SSIM loss', gen_ssim_loss, step=step)
        tf.summary.scalar('Discriminator loss', disc_loss, step= step)
# (step+1 )//46690
    # print('gen_total_loss', gen_total_loss, '\n',
    #     'gen_gan_loss', gen_gan_loss,  '\n',
    #     'gen_psnr_loss', gen_psnr_loss, '\n',
    #     'gen_ssim_loss', gen_ssim_loss, '\n',
    #     'disc_loss', disc_loss
    # )    


@tf.function
def val_step(val_dataset, generator, discriminator, summary_writer):
    
    for step , (input_image , target) in val_dataset.enumerate():
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=False)
        disc_generated_output = discriminator([input_image, gen_output], training=False)

        gen_total_loss, gen_gan_loss, gen_ssim_loss, gen_psnr_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        with summary_writer.as_default():
            tf.summary.scalar('val_gen_total_loss', gen_total_loss, step=step)
            tf.summary.scalar('val_gen_gan_loss', gen_gan_loss, step=step)
            tf.summary.scalar('val_gen_psnr_loss', gen_psnr_loss, step=step)
            tf.summary.scalar('val_gen_ssim_loss', gen_ssim_loss, step=step)
            tf.summary.scalar('val_disc_loss', disc_loss, step=step)


"""The actual training loop. Since this tutorial can run of more than one dataset, and the datasets vary greatly in size the training loop is setup to work in steps instead of epochs.

- Iterates over the number of steps.
- Every 10 steps print a dot (`.`).
- Every 1k steps: clear the display and run `generate_images` to show the progress.
- Every 5k steps: save a checkpoint.
"""

def fit(train_ds, val_ds, steps, generator, discriminator, summary_writer, vggModel):
    # example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        print("step", step,"*************************************************************************" )

        train_step_wass_char(input_image, target, step, generator, discriminator, summary_writer , vggModel)
        checkpoint.step.assign_add(1)
    # ckpt.step.assign_add(1)

        # Save (checkpoint) the model every 5 epochs i.e. steps / 18680 --->46690
        epoch_steps = 233450
        # epoch_steps = 15

        num = (step +1) // epoch_steps
        if (step + 1) % epoch_steps == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"checkpoint {num} saved")
            path = f"/user/HS402/sg02064/dissertation/results/{num}.png"

            for example_input, example_target in val_ds.take(1):
                generate_images(generator, example_input, example_target, path, num)    


            # for example_input, example_target in val_ds.take(1):
            #     generate_images(generator, example_input, example_target, path, num )

            # for input, target in val_ds.repeat().take(4).enumerate():
            #     generate_images(generator, input, target,path, random.randint(0,1000))
            print("Image generated")

# def generate_images(model, test_input, tar, step, chk):
#   prediction = model(test_input, training=True)
#   plt.figure(figsize=(15, 15))

#   display_list = [test_input[0], tar[0], prediction[0]]
#   title = ['Input Image', 'Ground Truth', 'Predicted Image']

#   for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.title(title[i])
#     # Getting the pixel values in the [0, 1] range to plot.
#     plt.imshow(display_list[i] * 0.5 + 0.5)
#     plt.axis('off')
#   plt.savefig(f"model_{step}_{chk}.png")
#   plt.show()


def generate_images(model, test_input, tar, path, num):
  prediction = model(test_input, training=False)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.title(f"Image generated at {num*50}th epoch.")  
  plt.savefig(path)

"""Test the function:"""

# for example_input, example_target in test_dataset.take(1):
#   generate_images(generator, example_input, example_target)



if __name__ == '__main__':

    model_path = r'/user/HS402/sg02064/dissertation/results'
    generator = Generator()
    # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file= os.path.join(model_path, "gen.jpeg"))
    discriminator = Discriminator()
    # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file= os.path.join(model_path, "disc.jpeg")) 
    vggModel = vgg_model()
# Initial learning rate
    # initial_learning_rate = 2e-4
# # Create learning rate schedules
#     generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate, decay_steps=46690, decay_rate=0.9
#     )

#     discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate, decay_steps=46690, decay_rate=0.9
#     )
# Create optimizers with the learning rate schedules
    # generator_optimizer = tf.keras.optimizers.Adam(generator_lr_schedule, beta_1=0.5)
    # discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lr_schedule, beta_1=0.5)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


    checkpoint_dir = r'/user/HS402/sg02064/dissertation/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator, step=tf.Variable(1))

        

    log_dir= r"/user/HS402/sg02064/dissertation/logs/"

    summary_writer = tf.summary.create_file_writer(
    # log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir + "fit/gan_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    root = r"/user/HS402/sg02064/dissertation/dataset/"
    # root = r'/user/HS402/sg02064/dissertation/temp/dataset/'
    train_low_light_images = sorted(glob( os.path.join(root,r"train/projections/*")))
    train_enhanced_images = sorted(glob( os.path.join(root,r"train/originals/*")))

    val_low_light_images = sorted(glob( os.path.join(root,r"val/projections/*")))
    val_enhanced_images = sorted(glob( os.path.join(root,r"val/originals/*")))

    test_low_light_images = sorted(glob( os.path.join(root,r"test/projections/*")))
    test_enhanced_images = sorted(glob( os.path.join(root,r"test/originals/*")))

    print('train_low_light_images', len(train_low_light_images))
    print('train_enhanced_images',len(train_enhanced_images))
    print('val_low_light_images', len(val_low_light_images))
    print('val_enhanced_images',len(val_enhanced_images))

    train_low_light_images.extend(val_low_light_images)
    train_enhanced_images.extend(val_enhanced_images)

    print('train_low_light_images', len(train_low_light_images))
    print('train_enhanced_images',len(train_enhanced_images))

    print("____ Train Dataset")
    train_dataset = get_dataset(train_low_light_images, train_enhanced_images, False)
    print(len(train_dataset))


    print("____ Test Dataset")
    val_dataset = get_dataset(test_low_light_images, test_enhanced_images, True)
    print(len(val_dataset))

    print("Fit model")
# 466900
    TOTAL_STEPS = 23345001
    # TOTAL_STEPS = 60
    # epoch_steps ==> iterations at which the 
    epoch_steps = 233450

    if len(os.listdir('/user/HS402/sg02064/dissertation/training_checkpoints')) > 0:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print('Restored from the checkpoint')
        num = 0
        for checkpoint_file in sorted(tf.io.gfile.glob(checkpoint_dir + "/ckpt-*.index")):
            checkpoint_num = int(checkpoint_file[-7:-6])
            if num < checkpoint_num:
                num = checkpoint_num 
        print(num)
        TOTAL_STEPS = TOTAL_STEPS - (num * epoch_steps)

        fit(train_dataset,val_dataset, TOTAL_STEPS , generator, discriminator, summary_writer, vggModel)
    else:
        print("Started training from scratch")
        fit(train_dataset,val_dataset, TOTAL_STEPS, generator, discriminator, summary_writer, vggModel)
    # example_input, example_target = next(iter(train_dataset.take(1)))
    # generate_images(generator, example_input, example_target, 1212121212)