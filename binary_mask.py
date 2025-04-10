import os
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import color
from scipy import ndimage, signal


cam_dir = r'results/plots'
image_dir = r"dataset/oxford-iiit-pet\images"
output_dir = r'results/binary_mask'
os.makedirs(output_dir, exist_ok=True)
threshold = 0.9

for filename in os.listdir(cam_dir):
    if not filename.endswith('.jpg') and not filename.endswith('.png'):
        continue

    base_name = os.path.splitext(filename)[0].replace('_cam', '')
    #base_name = os.path.splitext(filename)[0].replace('_blend', '')
    cam_path = os.path.join(cam_dir, filename)
    img_path = os.path.join(image_dir, base_name + '.jpg')
    save_path = os.path.join(output_dir, base_name + '_mask.png')

    if not os.path.exists(img_path):
        continue


    cam = Image.open(cam_path).convert('L').resize((224, 224))
    cam_np = np.array(cam).astype(np.float32) / 255.0
    img_np = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))
    mask = (cam_np > threshold).astype(np.uint8)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    if nb_labels > 0:
        largest = (sizes[1:]).argmax() + 1
        mask = (label_im == largest).astype(np.uint8)
    gray = color.rgb2gray(img_np / 255.0)
    grad_x = signal.convolve2d(gray, [[-1, 1]], mode='same', boundary='symm')
    grad_y = signal.convolve2d(gray, [[-1], [1]], mode='same', boundary='symm')
    edge = np.sqrt(grad_x**2 + grad_y**2)
    edge = (edge > 0.1).astype(np.uint8)
    combined = mask.astype(np.float32) + 0.6 * edge
    combined = (combined > 0.5).astype(np.uint8)
    combined = ndimage.binary_closing(combined, structure=np.ones((3, 3))).astype(np.uint8)
    combined = ndimage.binary_fill_holes(combined).astype(np.uint8)
    binary_mask = combined * 255
    Image.fromarray(binary_mask).save(save_path)
    print(f"Saved mask: {save_path}")


