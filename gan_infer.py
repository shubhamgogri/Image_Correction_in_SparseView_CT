import gan, model
import tensorflow as tf
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
import numpy as np
from skimage.transform import resize
from skimage.draw import line
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
import pandas as pd

def get_checkpoints(dir):
    checkpoint_prefix = os.path.join(dir, "ckpt")
    result = []
    for checkpoint_file in sorted(tf.io.gfile.glob(dir + "/ckpt-*.index")):
        checkpoint_prefix = checkpoint_file[:-6]
        print(checkpoint_prefix)
        result.append(checkpoint_prefix)
    return result
    
def generate_image(test_dataset):
    for example_input, example_target in test_dataset.take(1):
        infered = generator(example_input,training = False )
        print('data_type', type(infered))
        # tf_infered = tf.constant(infered[0])
        # final = tf.image.resize(tf_infered,[512,512],method = tf.image.ResizeMethod.BILINEAR)
        #tf.image.resize(infered, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # print('resized', size(final))
        return infered



def get_diff_line(image1, image2):

	# Load your images (image1 and image2)
	# image1 = imread(image1)
	# image2 = imread(image2)
	
	print('image1', image1.shape)
	print('image2', image2.shape)
	# vert 
	x1, x2, y1, y2 = 275, 275, 175 , 310
	# horizontal
	# x1, x2, y1, y2 = 45, 200, 200 , 200

	# Define starting and ending points of the line
	start_point = (y1, x1)  # Replace with your starting point coordinates
	end_point = (y2, x2)    # Replace with your ending point coordinates

	# Get coordinates of the line pixels
	line_coords = line(*start_point, *end_point)

	# Extract intensity values along the line from both images
	intensity_profile_image1 = image1[line_coords]
	intensity_profile_image2 = image2[line_coords]
	return line_coords, intensity_profile_image1, intensity_profile_image2, x1,x2,y1,y2

def opytimizer_with_lr():
# Initial learning rate
	initial_learning_rate = 2e-4
	# # Create learning rate schedules
	generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate, 
	decay_steps=46690, decay_rate=0.9
	)
	discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate, decay_steps=46690, decay_rate=0.9
	)
	# Create optimizers with the learning rate schedules
	generator_optimizer = tf.keras.optimizers.Adam(generator_lr_schedule, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lr_schedule, beta_1=0.5)
	return generator_optimizer, discriminator_optimizer


# prediction and gt

def infer():
    checkpoint_dir = r'/user/HS402/sg02064/dissertation/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	
    checkpoints_list = get_checkpoints(checkpoint_dir)
	

    for chk_path in checkpoints_list:     
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator, step=tf.Variable(1))
        checkpoint.restore(chk_path)
        image = generate_image(val_dataset)
        infer_path = r'/user/HS402/sg02064/dissertation/results/inferences'
        path = os.path.join(infer_path,f"{chk_path[-1]}.jpeg")
        print(path)
        # image = resize(image, (512,512,3), mode = 'reflect', preserve_range= True)
        
        #  -------> save the infered pictures. ------>
        # plt.figure()
        # resized_image1 = resize(image1, desired_shape, mode='constant', preserve_range=True)

        # plt.figure(figsize =(5.12,5.12), dpi = 100)
        plt.imshow(image[0] )
        plt.axis('off')
        # plt.savefig(path)
        plt.savefig(path, bbox_inches='tight',dpi = 100 ,  pad_inches=0)
        # plt.show()
        image = imread(path)
        print('saved image shape', image.shape)
        
def resize_img(input_image):
	input_image = tf.image.resize(input_image, [369, 369], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)        
	return input_image
	
def calculate_loss(pred, og):
	original_image = imread(og)
	pred = imread(pred)
	psnr = peak_signal_noise_ratio(original_image, pred)
	# psnr = model.peak_signal_noise_ratio(original_image, pred)
	ssim = structural_similarity(original_image, pred, win_size=3)
	# ssim = model.structural_similarity(original_image, pred)
	char = np.mean(np.sqrt(original_image - pred) + np.square(1e-3))
	# char = charbonnier_loss(original_image, pred)
	return psnr, ssim, char


def save_qualitative(psnr_list, ssim_list,char_loss_list,  proj , og):
	print('og and proj psnr: ',calculate_loss(proj, og)[0])
	print('psnr_list', psnr_list)
	print('og and proj ssim: ',calculate_loss(proj, og)[1])
	print('ssim_list',ssim_list)
	print('og and proj charbonnier: ', calculate_loss(proj, og)[2])
	print('charbonnier',char_loss_list)
	# proj_r =  f"/user/HS402/sg02064/dissertation/dataset/test1/projections/img_235_48.jpeg"
	# og_r =  f"/user/HS402/sg02064/dissertation/dataset/test1/originals/img_235_48.jpeg"
	# print(calculate_loss(og_r, og_r))
	df = pd.DataFrame({ 'PSNR': psnr_list, 'SSIM':ssim_list, 'CHARBONNIER':char_loss_list})
	out = df.agg({'PSNR':['mean', 'std'],'SSIM':['mean', 'std'],'CHARBONNIER':['mean', 'std'] })
	# final_df = out.append(pd.DataFrame([[calculate_loss(proj, og)[0], calculate_loss(proj, og)[1] , calculate_loss(proj, og)[2]]], columns =out.columns, index=['og_proj']))
	
	out.loc['og_proj'] = [calculate_loss(proj, og)[0], calculate_loss(proj, og)[1] , calculate_loss(proj, og)[2]]
	out.to_csv('/user/HS402/sg02064/dissertation/results/comparision.csv')


def generate_compare(proj_img, og_img, pred_img, pred_path):
	display_list = [proj_img, og_img, pred_img]
	title = ['Input Image', 'Ground Truth', 'Predicted Image']
	epoch = int(pred_path[:1])
	for i in range(3):
		plt.subplot(1, 3, i+1)
		plt.title(title[i])
		# Getting the pixel values in the [0, 1] range to plot.
		plt.imshow(display_list[i])
		plt.axis('off')
		plt.tight_layout()
		if epoch==0:
			plt.title(f"Image generated at {10*10}th epoch.")
		else:
			plt.title(f"Image generated at {epoch*10}th epoch.")
		plt.savefig(f'/user/HS402/sg02064/dissertation/results/{pred_path}')


if __name__ == '__main__':

    # 1. initialize network with the optimizers with dataset
    # 2. list of checkpoints
    # 3. load the checkpoints
    # 4. infer on all the checkpoints
    # 5. save the results (predicted image ) with the name 'image_name_{checkpoint_num}.jpeg'

    # 1. initialize network with the optimizers
    discriminator = gan.Discriminator()
    generator = gan.Generator()
    vggModel = gan.vgg_model()

    # generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    # discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_optimizer, discriminator_optimizer = opytimizer_with_lr()

    root = r"/user/HS402/sg02064/dissertation/dataset/"
    # root = r'/user/HS402/sg02064/dissertation/temp/dataset/'

    test_low_light_images = sorted(glob( os.path.join(root,r"infer/projections/*")))
    test_enhanced_images = sorted(glob( os.path.join(root,r"infer/originals/*")))

    print("____ Test Dataset")
    val_dataset = gan.get_dataset(test_low_light_images, test_enhanced_images, True)
    print(len(val_dataset))

    # 2. list of checkpoints
    infer()
        
      
	
	# 3. -------> LINE PROFILES. ------>
	# pred = '/content/drive/MyDrive/MIRNet/results/enhancements/img_235_48_model_45.jpeg'

    proj = os.path.join(root, 'test/projections/img_235_48.jpeg' )
    og = os.path.join(root, 'test/originals/img_235_48.jpeg' )
    
    # proj_img = resize_img(imread(proj))
    #imsave(proj, proj_img)
    
    # og_img = resize_img(imread(og))
    #imsave(og, og_img)
    
    proj_img = imread(proj)
    og_img = imread(og)
    
    psnr_list = []
    ssim_list = []
    char_loss_list = []
    infer_path = r"/user/HS402/sg02064/dissertation/results/inferences/"
    list_enh = os.listdir(infer_path)
    
    print(list_enh)
	
	for pred_path in list_enh:
		print(pred_path)
		pred = os.path.join(infer_path, pred_path)

		psnr, ssim, charb = calculate_loss(pred, og)
		psnr_list.append(psnr)
		ssim_list.append(ssim)
		char_loss_list.append(charb)

		print('difference between og and pred')
		pred_img = imread(pred)
		pred_line_coords,groundTruth_intensity, intensity_pred, x1,x2,y1,y2 = get_diff_line(og_img,pred_img)


		generate_compare(proj_img, og_img, pred_img, pred_path)


		# proj and Gt
		print('difference between og and proj')
		proj_line_coords,groundTruth_intensity, intensity_proj, x11,x22,y11,y22 = get_diff_line(og_img,proj_img)
		# Create an array of pixel positions along the line
		pred_pixel_positions = np.arange(len(pred_line_coords[0]))
		print(len(pred_pixel_positions))

		proj_pixel_positions = np.arange(len(proj_line_coords[0]))
		# Plot the intensity profiles along with images

		fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
		# Plot Image 1

		ax1.imshow(proj_img, cmap='gray')
		ax1.plot([x1, x2], [y1, y2], color='red', linewidth=2, label='Line')
		ax1.set_title('48 Projections')
		ax1.axis('off')

		# Plot Image 2
		ax2.imshow(og_img, cmap='gray')
		ax2.plot([x1, x2], [y1, y2], color='red', linewidth=2, label='Line')
		ax2.set_title('Ground Truth')
		ax2.axis('off')

		# Plot Image 2
		ax3.imshow(pred_img, cmap='gray')
		ax3.plot([x1, x2], [y1, y2], color='red', linewidth=2, label='Line')
		ax3.set_title('Generated Image')
		ax3.axis('off')

		line1 = ax4.plot(pred_pixel_positions, groundTruth_intensity, label='Ground Truth', color = 'red')
		line2 = ax4.plot(proj_pixel_positions, intensity_proj, label='48 projections', color = 'cyan')
		line3 = ax4.plot(pred_pixel_positions, intensity_pred, label='Generated Image', color = 'green')

		ax4.set_xlabel('Pixel Position')
		ax4.set_ylabel('Intensity')
		ax4.set_title('Intensity Profile Comparison')
		ax4.grid()
		plt.tight_layout()
		save_path = f"/user/HS402/sg02064/dissertation/results/lineprof/vert/{pred_path}"
		plt.savefig(save_path)
		# plt.show()

	save_qualitative(psnr_list, ssim_list,char_loss_list,  proj , og)	








