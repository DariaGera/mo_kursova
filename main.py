import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
number_func = 0
k = 0  # k-тий етап


def func(x):
    global number_func
    number_func = number_func + 1
    return (10 * (x[0] - x[1]) ** 2 + (x[0] - 1) ** 2) ** (1 / 4)


def sven_func(x0, s, func, sven_param=0.0001):
    delta_lambda = sven_param * (math.sqrt(x0[0] ** 2 + x0[1] ** 2)) / math.sqrt(s[0] ** 2 + s[1] ** 2)
    lambd = [0]

    f0 = func(x0)
    if f0 < func([x0[0] + delta_lambda * s[0],
                  x0[1] + delta_lambda * s[1]]):  # визначаємо куди рухаємось ліворуч чи праворуч
        delta_lambda = -delta_lambda
        lambd.append(delta_lambda)

    x1 = lambd[-1] + delta_lambda  # власне рухаємось
    f1 = func([x0[0] + x1 * s[0], x0[1] + x1 * s[1]])
    lmb = 0
    while f1 < f0:  # рухаємось доти, доки не почнемо рости
        delta_lambda *= 2
        lmb = x1
        x1 = lmb + delta_lambda
        f0 = f1
        f1 = func([x0[0] + x1 * s[0], x0[1] + x1 * s[1]])

    a = lmb + delta_lambda / 2  # Знаходимо наш інтервал невизначенності
    b = lmb - delta_lambda / 2
    f0 = func([x0[0] + lmb * s[0], x0[1] + lmb * s[1]])
    f1 = func([x0[0] + b * s[0], x0[1] + b * s[1]])

    if f0 < f1:
        if a < b:
            return [a, b]
        else:
            return [b, a]
    elif f1 < f0:
        if lmb < x1:
            return [lmb, x1]
        else:
            return [x1, lmb]
    else:
        if lmb < b:
            return [lmb, b]
        else:
            return [b, lmb]


def golden_func(x0, s):
    a, b = sven_func(x0, s, func)
    L = b - a
    lam1 = a + 0.382 * L
    lam2 = a + 0.618 * L
    f1 = func([x0[0] + lam1 * s[0], x0[1] + lam1 * s[1]])
    f2 = func([x0[0] + lam2 * s[0], x0[1] + lam2 * s[1]])
    while L > epsilon:
        if f1 > f2:
            a = lam1
            b = b
        elif f1 < f2:
            a = a
            b = lam2
        L = b - a
        lam1 = a + 0.382 * L
        lam2 = a + 0.618 * L
        f1 = func([x0[0] + lam1 * s[0], x0[1] + lam1 * s[1]])
        f2 = func([x0[0] + lam2 * s[0], x0[1] + lam2 * s[1]])
    if f1 < f2:
        lambd = lam1
    else:
        lambd = lam2
    return lambd


def dsc_powell_method(x0, s):
    l1, l3 = sven_func(x0, s, func)
    l2 = (l1 + l3) / 2

    f1 = func([x0[0] + l1*s[0], x0[1] + l1*s[1]])
    f2 = func([x0[0] + l2*s[0], x0[1] + l2*s[1]])
    f3 = func([x0[0] + l3*s[0], x0[1] + l3*s[1]])

    x_approx = l2 + ((l3 - l2) * (f1 - f3)) / (2 * (f1 - 2 * f2 + f3))  # Отримуємо апросимуючий поліном
    while (
        abs(l2 - x_approx) >= epsilon
        or
        abs(
            func([x0[0] + l2*s[0], x0[1] + l2*s[1]]) - func([x0[0] + x_approx*s[0], x0[1] + x_approx*s[1]])
        ) >= epsilon
    ):
        if x_approx < l2:
            l3 = l2
        else:
            l1 = l2
        l2 = x_approx
        funcRes = [
            func([x0[0] + l1*s[0], x0[1] + l1*s[1]]),
            func([x0[0] + l2*s[0], x0[1] + l2*s[1]]),
            func([x0[0] + l3*s[0], x0[1] + l3*s[1]]),
        ]
        a1 = (funcRes[1] - funcRes[0]) / (l2 - l1)
        a2 = ((funcRes[2] - funcRes[0]) / (l3 - l1) - a1) / (l3 - l2)
        x_approx = (l1 + l2) / 2 - a1 / (2 * a2)
    return l2


def Main_conjugate_powell_method(x0):
    global k
    n = 2  # розмірність простору
    k = 0  # k-тий етап
    S = []
    x_road = []
    not_done = True
    while not_done:
        if k == 0:
            s1 = [1, 0]
            s2 = [0, 1]
            S = [s2, s1]

        x_results = [x0]
        x_road.append(x0)
        for i in range(n):
            lambd_i = golden_func(x_results[-1], S[i])
            x_ii = [x_results[-1][0] + lambd_i*S[i][0], x_results[-1][1] + lambd_i*S[i][1]]
            x_results.append(x_ii)
            x_road.append(x_ii)

        x_n = x_results[-1]
        x_nn = [2*x_n[0] - x0[0], 2*x_n[1] - x0[1]]
        x_results.append(x_nn)

        max = 0
        indx_s = 0
        for i in range(1, n + 1):
            delta = func(x_results[i - 1]) - func(x_results[i])
            if delta > max:
                max = delta
                s_m = S[i - 1]
                indx_s = i - 1

        first = func(x_results[-1]) >= func(x0)
        sec = (func(x0) - 2*func(x_results[-2]) + func(x_results[-1]))*(func(x0) - func(x_results[-2]) - max)**2
        ond = 0.5*max*(func(x0) - func(x_results[-1]))
        second = sec >= ond
        if first or second:
            # на к+1 такі ж напрямки, що і на к
            S_kk = S
            S = S_kk
            if func(x_results[-2]) > func(x_results[-1]):
                x0 = x_results[-1]
            else:
                x0 = x_results[-2]

        else:
            # min f(x) по напрямку з х_0 в х_n
            s = [x_results[-2][0] - x_results[0][0], x_results[-2][1] - x_results[0][1]]
            lambd = golden_func(x_results[-2], s)
            x0 = [x_results[-2][0] + lambd*s[0], x_results[-2][1] + lambd*s[1]]
            S_old = S
            S = []
            s_m = s
            for i in range(n):
                if i == indx_s:
                    S.append(s_m)
                    pass
                else:
                    S.append(S_old[i])

        if LA.norm([x_results[-2][0] - x_results[0][0], x_results[-2][1] - x_results[0][1]]) <= powell_method_epsilon:
            not_done = False
            return x_road, x_results[-1]
        k += 1


def simple_conjugate_powell_method(x0):
    global k
    n = 2  # розмірність простору
    k = 0  # k-тий етап
    S = []
    x_road = []
    not_done = True
    while not_done:
        if k == 0:
            s1 = [1, 0]
            s2 = [0, 1]
            S = [s2, s1]

        x_results = [x0]
        x_road.append(x0)
        for i in range(n):
            lambd_i = golden_func(x_results[-1], S[i])
            x_ii = [x_results[-1][0] + lambd_i*S[i][0], x_results[-1][1] + lambd_i*S[i][1]]
            x_results.append(x_ii)
            x_road.append(x_ii)

        x_n = x_results[-1]
        x_nn = [2*x_n[0] - x0[0], 2*x_n[1] - x0[1]]
        x_results.append(x_nn)

        max = 0
        indx_s = 0
        for i in range(1, n + 1):
            delta = func(x_results[i - 1]) - func(x_results[i])
            if delta > max:
                max = delta
                s_m = S[i - 1]
                indx_s = i - 1

        # min f(x) по напрямку з х_0 в х_n
        s = [x_results[-2][0] - x_results[0][0], x_results[-2][1] - x_results[0][1]]
        lambd = golden_func(x_results[-2], s)
        x0 = [x_results[-2][0] + lambd*s[0], x_results[-2][1] + lambd*s[1]]
        S_old = S
        S = []
        s_m = s
        for i in range(n):
            if i == indx_s:
                S.append(s_m)
                pass
            else:
                S.append(S_old[i])

        if LA.norm([x_results[-2][0] - x_results[0][0], x_results[-2][1] - x_results[0][1]]) <= powell_method_epsilon:
            not_done = False
            return x_road, x_results[-1]
        k += 1


x0 = [-1.2, 0]
epsilon = 0.1  # від цього значення дуже залежить чи спроможна програма роз'язати задачу
powell_method_epsilon = 0.001

simple_x_road, simp_x = simple_conjugate_powell_method(x0)
print(simp_x)
print(number_func)
number_func = 0
main_x_road, main_x = Main_conjugate_powell_method(x0)
print(main_x)
print(number_func)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = (10*(x-y)**2+(x-1)**2)**(1/4)


for x_i in simple_x_road:
    if x_i == simple_x_road[0]:
        ax.plot(x_i[0], x_i[1], func(x_i), 'ko', alpha=1)
    elif x_i == simple_x_road[-1]:
        ax.plot(x_i[0], x_i[1], func(x_i), 'go', alpha=0.5)  # note the 'ro' (no '-') and the alpha
    else:
        ax.plot(x_i[0], x_i[1], func(x_i), 'ro', alpha=0.5)  # note the 'ro' (no '-') and the alpha

for x_i in main_x_road:
    if x_i == main_x_road[0]:
        ax.plot(x_i[0], x_i[1], func(x_i), 'ko', alpha=1)
    if x_i == main_x_road[-1]:
        ax.plot(x_i[0], x_i[1], func(x_i), 'bo', alpha=0.5)  # note the 'ro' (no '-') and the alpha
    else:
        ax.plot(x_i[0], x_i[1], func(x_i), 'yo', alpha=0.5)  # note the 'ro' (no '-') and the alpha

ax.plot_surface(x, y, z)
plt.show()

fig, ax = plt.subplots()
print(k)
res_list1 = np.array(simple_x_road).T
ax.plot(res_list1[0], res_list1[1], marker='v', markersize=4)
circle1 = plt.Circle((0, 0), 2, color='g', fill=False)
ax = plt.gca()
ax.add_patch(circle1)
plt.show()







