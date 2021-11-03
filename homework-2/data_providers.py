import numpy as np


def create_and_generator():
    while True:
        a, b = np.random.randint(2), np.random.randint(2)
        res = a and b
        yield [a, b], res
