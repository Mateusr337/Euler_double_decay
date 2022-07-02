import numpy as np
import matplotlib.pyplot as plt

# Resolução, estudo e erro do problema de decaimento duplo, usando o algoritmo de Euler.
# O evolução das quantidades dos material radiativos estão regidas pelas seguintes equações:

# definindo lambda1 = 2 e lamda2 = 4

lamb_one = 2
lamb_two = 4


def f_line_one(lamb, fx):
    return - lamb * fx


def f_line_two(lamb_one, lamb_two, fx_one, fx_two):

    return lamb_one * fx_two - fx_one * lamb_two
