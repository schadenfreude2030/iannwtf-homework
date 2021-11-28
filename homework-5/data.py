import tensorflow as tf
import tensorflow_datasets as tfds


def load_data() -> (tf.data.Dataset, tf.data.Dataset):
    ds_train, ds_test = tfds.load(
        "fashion_mnist",
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    return prep_data(ds_train), prep_data(ds_test)


def prep_data(ds: tf.data.Dataset) -> tf.data.Dataset:
    # replace uint by float
    ds = ds.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # normalize inputs
    ds = ds.map(lambda img, target: ((img/128.)-1., target))
    # create one-hot targets
    ds = ds.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache
    ds = ds.cache()
    # shuffle, batch and prefetch
    ds = ds.shuffle(1000).batch(50).prefetch(100)
    return ds
