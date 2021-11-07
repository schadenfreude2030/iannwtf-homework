import numpy as np


def create_and_generator():
    while True:
        a, b = np.random.randint(2), np.random.randint(2)
        res = a and b
        yield np.array([a, b]), res


def create_nand_generator():
    while True:
        a, b = np.random.randint(2), np.random.randint(2)
        res = not a and b
        yield np.array([a, b]), res


def create_or_generator():
    while True:
        a, b = np.random.randint(2), np.random.randint(2)
        res = a or b
        yield np.array([a, b]), res


def create_nor_generator():
    while True:
        a, b = np.random.randint(2), np.random.randint(2)
        res = not a or b
        yield np.array([a, b]), res


def create_xor_generator():
    while True:
        a, b = np.random.randint(2), np.random.randint(2)
        res = a ^ b
        yield np.array([a, b]), res
