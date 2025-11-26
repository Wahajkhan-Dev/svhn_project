import numpy as np
import tensorflow as tf
import cv2

def preprocess_img(img, target=(32,32)):
    img = tf.image.resize(img, target)
    img = img/255.0
    return img.numpy()

def add_gaussian_noise(img, sigma=0.1):
    noise = np.random.normal(0, sigma, img.shape)
    out = img + noise
    return np.clip(out, 0, 1)

def blur_img(img, ksize=3):
    arr = (img*255).astype('uint8')
    b = cv2.GaussianBlur(arr, (ksize,ksize), 0)
    return b.astype('float32')/255.0

def save_img(path, arr):
    from PIL import Image
    im = Image.fromarray((arr*255).astype('uint8'))
    im.save(path)
