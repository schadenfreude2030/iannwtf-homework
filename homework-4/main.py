import pandas as pd
import create_data
import training

df = pd.read_csv("./winequality-red.csv", delimiter=";")
# print(df.info())
# print(df.dtypes)
# print(df.head(1))
# quality should be the target. All other keys should be inputs
train_ds, valid_ds, test_ds = create_data.create_data(df)
# print(train_ds)
# print(valid_ds)
# print(test_ds)
training.run(train_ds, test_ds, valid_ds)
