from data import *
from model import MyCnnModel
import matplotlib.pyplot as plt
import numpy as np

ds_train, ds_test = load_data()

tf.keras.backend.clear_session()

# Setting Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.1

# Initialize the loss-function
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer
optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)
# Initialize the model
model = MyCnnModel(cross_entropy_loss, optimizer)

# Initialize lists for tracking loss and accuracy
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Testing models performance before training starts.
# Test-Dataset
test_loss, test_accuracy = model.test(ds_test)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)
# Train-Dataset
train_loss, train_accuracy = model.test(ds_train)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

# Training for EPOCHS.
for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {str(epoch)} starting with test-accuracy of {np.round(test_accuracies[-1], 3)}')
    epoch_loss_agg = []
    epoch_accuracy_agg = []
    for input, target in ds_train:
        train_loss, train_accuracy = model.train_step(input, target)
        epoch_loss_agg.append(train_loss)
        epoch_accuracy_agg.append(train_accuracy)

    # track training loss and accuracy
    train_losses.append(tf.reduce_mean(epoch_loss_agg))
    train_accuracies.append(tf.reduce_mean(epoch_accuracy_agg))
    # track loss and accuracy for test-dataset
    test_loss, test_accuracy = model.test(ds_test)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(20, 6)

fig.suptitle('Training Progress for Genomics Bacteria Classification')
axs[0].plot(train_losses, color='orange', label='train losses')
axs[0].plot(test_losses, color='green', label='test losses')
axs[0].set(ylabel='Losses')
axs[0].legend()
axs[1].plot(train_accuracies, color='orange', label='train accuracies')
axs[1].plot(test_accuracies, color='green', label='test accuracies')

axs[1].set(xlabel='Epochs', ylabel='Accuracies')
axs[1].legend()
plt.show()