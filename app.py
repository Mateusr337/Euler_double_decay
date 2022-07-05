import numpy as np
import matplotlib.pyplot as plt

# Resolução, estudo e erro do problema de decaimento duplo, usando o algoritmo de Euler.
# O evolução das quantidades dos material radiativos estão regidas pelas seguintes equações:

lamb_one = 0.0105
lamb_two = 0.1155

delta_t = 0.001
final_interval_t = 330
f0_one = 1
f0_two = 0

steps = np.arange(0, final_interval_t + delta_t, delta_t)


def f_line_one(fx):
    return - lamb_one * fx


def f_line_two(fx_one, fx_two):
    return lamb_one * fx_one - lamb_two * fx_two


def f_n2(t):
    return (f0_one * lamb_one / (lamb_two - lamb_one)) * (np.exp(-lamb_one * t) - np.exp(-lamb_two * t))


def euler_method(f0_one, f0_two, delta_x, steps, f_line_one, f_line_two):
     # f(x0 + ∆x) = f(x0) + f'(x0)∆x
    results_one = [f0_one]
    results_two = [f0_two]

    fx_one = f0_one
    fx_two = f0_two

    for t in steps:
        next_f_one = fx_one + f_line_one(fx_one) * delta_x
        results_one.append(next_f_one)

        next_f_two = fx_two + f_line_two(fx_one, fx_two) * delta_x
        results_two.append(next_f_two)

        fx_two = next_f_two
        fx_one = next_f_one

    results_one.pop()
    results_two.pop()

    return {'one': results_one, 'two': results_two}


euler = euler_method(f0_one, f0_two, delta_t, steps, f_line_one, f_line_two)

analytical = []
for t in steps:
    analytical.append(f_n2(t))

plt.plot(steps, euler['one'])
plt.plot(steps, (euler['two']))
plt.title('Decaimento Radioativo')
plt.xlabel('Tempo (h)')
plt.ylabel('Massa do elemento (Kg)')
plt.legend(["Função 1 (n1)", "Função 2 (n2)"])
plt.savefig("decaimento-radioativo.png", dpi=100)
plt.show()


plt.plot(steps, np.multiply(euler['two'], 10))
plt.title('Decaimento Radioativo')
plt.xlabel('Tempo (h)')
plt.ylabel('Massa do elemento (Kg)')
plt.legend(["Função 2 (n2 * 10)"])
plt.savefig("decaimento-radioativo-10n2.png", dpi=100)
plt.show()


plt.plot(steps, analytical)
plt.plot(steps, euler['two'], "--")
plt.title('Decaimento Radioativo')
plt.xlabel('Tempo (h)')
plt.ylabel('Massa do elemento (Kg)')
plt.legend(["analitical", "Função 2 (n2)"])
plt.savefig("decaimento-radioativo-comparação-erro.png", dpi=100)
plt.show()


delta_t_2 = delta_t * 1000
steps_2 = np.arange(0, final_interval_t + delta_t_2, delta_t_2)
euler_2 = euler_method(f0_one, f0_two, delta_t_2,
                       steps_2, f_line_one, f_line_two)


plt.plot(steps, analytical)
plt.plot(steps, euler['two'], "--")
plt.plot(steps_2, euler_2['two'])
plt.title('Decaimento Radioativo')
plt.xlabel('Tempo (h)')
plt.ylabel('Massa do elemento (Kg)')
plt.legend(["analitical", "Função 2 (n2) - delta t: {0}".format(
    delta_t), "Função 2 (n2) - delta t: {0}".format(delta_t_2)])
plt.savefig("decaimento-radioativo-comparação-erro-delta_maior.png", dpi=100)
plt.show()
