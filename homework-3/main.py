import tensorflow as tf
import tensorflow_datasets as tfds

ds_genes, info = tfds.load(
    'genomics_ood',
    split='train',
    shuffle_files=True,
    with_info=True,
    as_supervised=True
)
fig = tfds.show_examples(ds_genes, info)
print(info)


# this did not work on windows, I used google colab:
# https://colab.research.google.com/drive/1nJDrwsh2UyIU5eiTXROaR6HBBXN8fV-g#scrollTo=ufkvs6S7b1Y7
