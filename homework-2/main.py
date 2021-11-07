import mlp
import data_providers
from matplotlib import pyplot as plt

STEP_SIZE = 100

fig, axes = plt.subplots(2, 3)
axes = axes.ravel()
titles = ["AND", "NAND", "OR", "NOR", "XOR"]
generators = [
    data_providers.create_and_generator(),
    data_providers.create_nand_generator(),
    data_providers.create_or_generator(),
    data_providers.create_nor_generator(),
    data_providers.create_xor_generator()
]

for ind_operator in range(5):
    network = mlp.MLP()
    accuracies = []
    losses = []

    for j in range(10):
        sum_results = 0
        sum_loss = 0
        for i in range(STEP_SIZE):
            input, output = next(generators[ind_operator])
            calc_out = network.forward_step(input)
            loss = network.backprop_step(output)
            if calc_out > 0.5:
                result = 1
            else:
                result = 0
            sum_results += 0 + (result == output)
            sum_loss += loss
        accuracies.append(sum_results / STEP_SIZE)
        losses.append(sum_loss / STEP_SIZE)

    axes[ind_operator].set_title(titles[ind_operator])
    axes[ind_operator].plot(accuracies, label="accuracy")
    axes[ind_operator].plot(losses, label="loss")
    if ind_operator == 4:
        axes[ind_operator].legend()

plt.ylim([0, 1])
plt.show()

