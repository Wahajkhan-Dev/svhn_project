
import argparse, os, json
import tensorflow as tf
import tensorflow_datasets as tfds

def build_model(input_shape=(32,32,3), num_classes=10):
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights=None)
    x = base.output
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare(ds, batch_size=64, training=True):
    def _prep(img,label):
        img = tf.image.resize(img, (32,32))/255.0
        return img, label
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--save-dir', default='outputs')
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ds_train = tfds.load('svhn_cropped', split='train', as_supervised=True)
    ds_test = tfds.load('svhn_cropped', split='test', as_supervised=True)
    # use small val split
    ds_train = ds_train.take(50000)

    train_ds = prepare(ds_train, args.batch, training=True)
    test_ds = prepare(ds_test, args.batch, training=False)

    model = build_model()
    model.fit(train_ds, epochs=args.epochs, validation_data=test_ds.take(100))
    model.save(os.path.join(args.save_dir, 'model.h5'))

    preds = {'pred': [], 'true': []}
    for imgs, labels in test_ds:
        p = model.predict(imgs)
        preds['pred'].extend(p.argmax(axis=1).tolist())
        preds['true'].extend(labels.numpy().tolist())
    with open(os.path.join(args.save_dir, 'preds.json'),'w') as f:
        json.dump(preds, f)
    print('Saved model and preds to', args.save_dir)
