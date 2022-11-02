import matplotlib.pyplot as plt
import numpy as np


# Rosenbrock function
def f(x1, x2):
    return 100 * ((x2 - x1 ** 2) ** 2) + (1 - x1) ** 2


# gradient vector
def nabla(x1, x2):
    partial_x1 = -400 * x2 * x1 + 400 * (x1 ** 3) - 2 + 2 * x1
    partial_x2 = 200 * x2 - 200 * (x1 ** 2)
    return np.matrix([[partial_x1, ], [partial_x2, ]])


# hessian matrix
def hessian(x1, x2):
    mat = [[0, 0], [0, 0]]
    mat[0][0] = -400 * x2 + 1200 * (x1 ** 2) + 2
    mat[0][1] = -400 * x1
    mat[1][0] = -400 * x1
    mat[1][1] = 200
    return np.matrix(mat)


# backtracking line search
def line_search(x1, x2, type):
    rho = 0.9
    c = 0.6
    alpha = 1
    grad = nabla(x1, x2)
    dir = np.matrix([[0, ], [0, ]])
    if type == "newton":
        dir = -np.matmul(np.linalg.inv(hessian(x1, x2)), grad)
    elif type == "steepest descent":
        dir = -grad

    while True:
        curr_val = f(x1, x2)
        new_val = f(x1 + alpha * dir.item((0, 0)), x2 + alpha * dir.item((1, 0)))
        change = c * alpha * np.matmul(grad.transpose(), dir).item((0, 0))

        if new_val <= curr_val + change:
            break
        alpha *= rho

    return alpha * dir


# Optimisation method
def minimize(x1, x2, type):
    epsilon = 1e-9
    k = 0
    x1_trace = []
    x2_trace = []
    while True:
        x1_trace.append(x1)
        x2_trace.append(x2)
        grad = nabla(x1, x2)
        if grad.item((0, 0)) ** 2 + grad.item((1, 0)) ** 2 < epsilon ** 2:
            return x1, x2, np.array(x1_trace), np.array(x2_trace)
        else:
            k += 1
            delta = line_search(x1, x2, type)
            x1 = x1 + delta.item((0, 0))
            x2 = x2 + delta.item((1, 0))
            if k > 100000:
                return None


print("========== Starting Newton's method...")
x1_newton1, x2_newton1, x1_trace_newton1, x2_trace_newton1 = minimize(1.2, 1.2, "newton")
print(f"Newton approx starting from (1.2, 1.2): ({x1_newton1}, {x2_newton1})")
x1_newton2, x2_newton2, x1_trace_newton2, x2_trace_newton2 = minimize(-1.2, 1.0, "newton")
print(f"Newton approx starting from (-1.2, 1.0): ({x1_newton2}, {x2_newton2})")
print("========== Starting steepest descent method...")
x1_sd1, x2_sd1, x1_trace_sd1, x2_trace_sd1 = minimize(1.2, 1.2, "steepest descent")
print(f"Steepest descent approx starting from (1.2, 1.2): ({x1_sd1}, {x2_sd1})")
x1_sd2, x2_sd2, x1_trace_sd2, x2_trace_sd2 = minimize(-1.2, 1.0, "steepest descent")
print(f"Steepest descent approx starting from (-1.2, 1.0): ({x1_sd2}, {x2_sd2})")
traces = ((x1_trace_newton1, x2_trace_newton1), (x1_trace_newton2, x2_trace_newton2), (x1_trace_sd1, x2_trace_sd1), (x1_trace_sd2, x2_trace_sd2))


# plot
DELTA = 0.025
x1 = np.arange(-1.5, 1.5, DELTA)
x2 = np.arange(-1.0, 2.0, DELTA)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)
levels = np.array([20, 50, 100, 200, 500, 1000])
types = [
    "Newton's Method Starting at (1.2, 1.2)",
    "Newton's Method Starting at (-1.2, 1.0)",
    "Steepest Descent Method Starting at (1.2, 1.2)",
    "Steepest Descent Method Starting at (-1.2, 1.0)"
]

fig, axs = plt.subplots(2, 2, constrained_layout=False)
index = 0
for ax, type in zip(axs.flat, types):
    ax.set_title(type, fontsize=8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$", labelpad=0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 2.0)
    contour = ax.contour(X1, X2, Z, levels)
    ax.clabel(contour, inline=True, fontsize=5)
    ax.plot(traces[index][0], traces[index][1], marker='o', markersize=1, c='g')
    index += 1

plt.gcf().subplots_adjust(right=0.9, hspace=0.5, wspace=0.3)
plt.savefig('hw1.pdf')
