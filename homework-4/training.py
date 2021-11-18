import tensorflow as tf
import numpy as np
from custom_model import *
import matplotlib.pyplot as plt


def train_step(model, inp, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(inp)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (inp, target) in test_data:
        prediction = model(inp)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = 0 + (np.round(target, 0) == np.round(prediction, 0))
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def run(train_dataset, test_dataset, validation_dataset):
    tf.keras.backend.clear_session()

    # For showcasing we only use a subset of the training and test data (generally use all of the available data!)

    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.1

    # Initialize the model.
    model = CustomModel()
    # Initialize the loss: binary cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []

    test_losses = []
    test_accuracies = []

    # testing once before we begin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # check how model performs on train data once before we begin
    train_loss, _ = test(model, train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        # training (and checking in with training)
        epoch_loss_agg = []
        for inp, target in train_dataset:
            train_loss = train_step(model, inp, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # Visualize accuracy and loss for training and test data.
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3), ("training", "test", "test accuracy"))
    plt.ylim([0, 1])
    plt.show()
