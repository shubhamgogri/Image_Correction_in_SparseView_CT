import cv2
# import pydicom as dicom
from skimage.transform import radon, iradon
import os
import numpy as np
# import matplotlib.pyplot as plt


def projections(path, theta):
    # path = r"F:\Shubham\surrey\data- ct\p\img_27.jpeg"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    sinogram = img_radon(img, theta)
    projection = img_iradon(sinogram, theta)
    return projection


def save_proj(root, item, destination, theta):
    path = os.path.join(root, item)
    print(path)
    projection = projections(path, theta)
    proj_path_theta = os.path.join(destination, item)
    print(np.shape(projection))

    normalized_image = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert normalized image to BGR color mode
    bgr_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

    # Save the grayscale image
    cv2.imwrite(proj_path_theta, bgr_image)

    # plt.imsave(proj_path_theta, projection, cmap='gray')


def img_iradon(sinogram, angle):
    theta = np.linspace(start=0, stop=180, num=angle, endpoint=False)
    # theta = np.linspace(0., angle, max(sinogram.shape), endpoint=False)
    reconstruction_img = iradon(sinogram, theta=theta, filter_name='ramp')
    return reconstruction_img


def img_radon(image, angle):
    theta = np.linspace(start=0, stop=180, num=angle, endpoint=False)
    # theta = np.linspace(0., angle, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    return sinogram


def process(root, destination):

    # info: root -> originals with 4 copies (48,64,98,192)
    # destination -> the projections of each copies i.e. 48,64,98,192

    comp_list = os.listdir(root)
    i = 1
    total_count = len(comp_list)
    for item in comp_list:
        print(f"{i} out of {total_count} ***************")
        if "192.jpeg" in item:
            save_proj(root, item, destination, 192)
        elif "48.jpeg" in item:
            save_proj(root, item, destination, 48)
        elif "64.jpeg" in item:
            save_proj(root, item, destination, 64)
        elif "96.jpeg" in item:
            save_proj(root, item, destination, 96)
        i += 1
