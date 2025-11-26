"""Generate Grad-CAMs using a Keras model."""
import argparse, os, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model

def make_gradcam(img, model, conv_layer_name=None):
    if conv_layer_name is None:
    # find a conv-like layer robustly
        conv_layer_name = None
        for l in reversed(model.layers):
            try:
                shape = tf.keras.backend.int_shape(l.output)
            except Exception:
                continue
            if shape is None:
                continue
            if len(shape) == 3 or len(shape) == 4:
                conv_layer_name = l.name
                break
        if conv_layer_name is None:
            raise RuntimeError("No conv-like layer found in model for Grad-CAM.")

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(conv_layer_name).output, model.output])
    img_t = tf.expand_dims(img, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_t)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., tf.newaxis], (img.shape[0], img.shape[1]))
    return cam.numpy().squeeze()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='outputs/gradcam')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model)
    import tensorflow_datasets as tfds
    ds_test = tfds.load('svhn_cropped', split='test', as_supervised=True)
    ds_test = ds_test.map(lambda x,y: (tf.image.resize(x, (32,32))/255.0, y)).batch(64)
    saved = 0
    import matplotlib.pyplot as plt
    for imgs, labels in ds_test.take(5):
        for i in range(min(10, imgs.shape[0])):
            img = imgs[i].numpy()
            cam = make_gradcam(img, model)
            plt.imsave(os.path.join(args.out, f'img_{saved}.png'), img)
            plt.imsave(os.path.join(args.out, f'cam_{saved}.png'), cam, cmap='jet')
            saved += 1
            if saved>=20:
                break
        if saved>=20:
            break
    print('Saved sample images and cams to', args.out)
