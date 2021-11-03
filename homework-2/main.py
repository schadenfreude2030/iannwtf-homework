import mlp
import data_providers
from matplotlib import pyplot as plt

STEP_SIZE = 500
network = mlp.MLP()
data_generator = data_providers.create_and_generator()
accuracies = []
losses = []
for j in range(10):
    sum_results = 0
    sum_loss = 0
    for i in range(STEP_SIZE):
        input, output = next(data_generator)
        calc_out = network.forward_step(input)
        sum_loss += network.backprop_step(output)
        if calc_out > 0.5:
            result = 1
        else:
            result = 0
        sum_results += 0 + (result == output)
    accuracies.append(sum_results / STEP_SIZE)
    losses.append(sum_loss / STEP_SIZE)
print(accuracies)

plt.plot(accuracies, label="accuracy")
plt.plot(losses, label="loss")
plt.ylim([0, 1])
plt.legend()
plt.show()
