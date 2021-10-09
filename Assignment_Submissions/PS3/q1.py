import numpy as np
import matplotlib.pyplot as plt

# Problem 1


def rk4_step(fun, x, y, h):
    """
    Perform one step of Runge-Kutta 4th order integration
    """
    k1 = h * fun(x, y)
    k2 = h * fun(x + h/2, y + k1/2)
    k3 = h * fun(x + h/2, y + k2/2)
    k4 = h * fun(x + h, y + k3)
    dy = (k1 + 2*k2 + 2*k3 + k4)/6
    return y + dy


def rk4_stepd(fun, x, y, h):
    y1 = rk4_step(fun, x, y, h)
    y21 = rk4_step(fun, x, y, h/2)
    y22 = rk4_step(fun, x + h/2, y21, h/2)

    delta = y22 - y1

    return y22 + delta/15


def rk4_looper(fun, f_true, n_pts, y0, method='Simple'):
    start, stop, npts = -20, 20, n_pts
    x_array = np.linspace(start, stop, npts)

    y_array = np.zeros(n_pts)
    y_array[0] = y0

    hs = np.diff(x_array)

    solver = rk4_step if method == 'Simple' else rk4_stepd

    func.count = 0
    for i, x in enumerate(x_array[:-1]):
        y_array[i+1] = solver(fun, x, y_array[i], hs[i])
    n_calls = func.count

    y_true = f_true(x_array)
    err = y_array - y_true
    print(
        f"""
        {method} Method Results
          Standard error: {np.std(err):.3e}
          Function calls: {n_calls}
          Function calls/step: {n_calls/(n_pts - 1)}
        """)
    return x_array, err, n_calls, y_array


def func(x, y):
    func.count += 1
    return y/(1 + x**2)


def f_true(x, x0=-20):
    return np.exp(np.arctan(x)-np.arctan(x0))


def problem_1():
    print('\tProblem 1')
    y0 = 1
    n_pts = 200
    n_pts_s = 3*(n_pts//3) + 1
    n_pts_m = n_pts//3 + 1

    xs, e_s, n_s, y_s = rk4_looper(func, f_true, n_pts_s, y0, method='Simple')
    xm, e_m, n_m, y_m = rk4_looper(func, f_true, n_pts_m, y0, method='Modified')

    print(
        f"""
        We find that using the same number of function calls,
        the modified rk4 has a standard error {e_m.std()/e_s.std():.3e} smaller
        """)

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].set_title('ODE Solver Solution')
    axs[0].plot(xs, f_true(xs), '-', label='Analytical')
    axs[0].plot(xs, y_s, '--', label='RK4 Simple')
    axs[0].plot(xm, y_m, '-.', label='RK4 Modified')
    axs[0].legend()

    axs[1].set_title('ODE Solver Residuals')
    axs[1].plot(xs, e_s, '--', label='RK4 Simple')
    axs[1].plot(xm, e_m, '-.', label='RK4 Modified')
    axs[1].legend()

    plt.savefig('Results/Problem_1.png')
    plt.clf()
