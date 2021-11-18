import pandas as pd
import tensorflow as tf
import numpy as np


def make_binary(item, threshold):
    return (item > threshold) + 0


def prep_data(data):
    return data.cache().batch(8).prefetch(20)


def create_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    full_size = len(df)
    train_size = int(0.7 * full_size)
    valid_size = int(0.15 * full_size)

    target_df = df["quality"]
    features_df = df.drop("quality", axis=1)
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(features_df.values, tf.float32),
            tf.cast(make_binary(target_df.values, np.mean(target_df)), tf.int32)
        )
    )

    train_ds = ds.shuffle(1).take(train_size)
    remaining = ds.skip(train_size)
    valid_ds = remaining.take(valid_size)
    test_ds = remaining.skip(valid_size)
    return prep_data(train_ds), prep_data(valid_ds), prep_data(test_ds)
