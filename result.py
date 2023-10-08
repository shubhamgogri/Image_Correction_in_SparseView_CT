import data
import cv2
import matplotlib.pyplot as plt

src = '/user/HS402/sg02064/dissertation/img_235_48.jpeg'

def save(projection, path):
    normalized_image = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert normalized image to BGR color mode
    bgr_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)

        # Save the grayscale image
    cv2.imwrite(path, bgr_image)

img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

sinogram = data.img_radon(img, 360)
projection = data.img_iradon(sinogram, 360)
save(sinogram, 'sinogram.jpeg')
save(projection, 'proj.jpeg')

