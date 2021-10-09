import numpy as np
import matplotlib.pyplot as plt


def paraboloid(p, x_arr):
    x, y = x_arr
    grad = np.zeros((len(x), len(p)))
    grad[:, 0] = 1
    grad[:, 1] = (x**2 + y**2)
    grad[:, 2] = -2*x
    grad[:, 3] = -2*y

    return grad@p, grad


def fit_newton(m, fun, x, y, niter=10):
    for i in range(niter):
        model, derivs = fun(m, x)
        r = y - model
        lhs = derivs.T@derivs
        rhs = derivs.T@r
        dm = np.linalg.inv(lhs)@rhs
        m = m + dm
        chisq = np.sum(r**2)
        if i == 0:
            print(f'Initial chisq: {chisq}')
    print(f'Final chisq: {chisq}')
    return m


x, y, z = np.loadtxt('dish_zenith.txt').T


def problem_3():
    print('\tProblem 3')
    print(
        """
        a)
        We rearrange the equation as follows:
        z = z0 + a * ((x - x_0)^2 + (y - y_0)^2)
        z = z0 + a * (x^2 + y^2 -2*x*x_0 - 2*y*y_0 + x_0^2 + y_0^2)
        z = (z0 + a*(x_0^2 + y_0^2)) + a * (x^2 + y^2) - 2*a*x*x_0 - 2*a*y*y_0

        z = p0 + p1*(x^2 + y^2) - 2*p2*x - 2*p3*y
        
        where the new parameters are given by:        
        p0 = (z0 + a*(x_0^2 + y_0^2))
        p1 = a
        p2 = a*x_0
        p3 = a*y_0
        """
        )

    m0 = np.random.rand(4)
    m_fit = fit_newton(m0, paraboloid, (x, y), z)
    z_fit, A = paraboloid(m_fit, [x, y])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, label='data points', alpha=0.5)
    # we offset the fit so that the points dont overlap in the graph
    ax.scatter(x, y, z_fit+100, label='Fit', alpha=0.5)
    plt.legend()
    plt.savefig('Results/problem3.png')
    plt.clf()
    plt.close()
    print(
        f"""
        b)
        We carry out the fit, and find the paramters to be:
        p0 = {m_fit[0]}
        p1 = {m_fit[1]}
        p2 = {m_fit[2]}
        p3 = {m_fit[3]}

        Which, in previous paramters correspond to:
        a = {m_fit[1]} mm
        x0 = {m_fit[2]/m_fit[1]} mm
        y0 = {m_fit[3]/m_fit[1]} mm
        z0 = {m_fit[0] - (m_fit[2]**2 + m_fit[3]**2)/m_fit[1]} mm
        """)

    # c)

    noise = np.std(z - z_fit)
    N = np.identity(len(z)) * noise**2
    covar = np.linalg.inv(A.T@np.linalg.inv(N)@A)

    a_err = np.sqrt(covar[1, 1])
    a_est = m_fit[1]

    f_est = 1/(4*a_est)
    f_err = f_est*a_err/a_est

    print(
        f"""
        c)
        From the noise estimate:
        Parameter a estimate: ({a_est:.4e} +- {a_err:.4e}) mm

        we have that f = 1/(4a)

        From the first order taylor expansion:
        f_err = f * a_error/a

        Focal length estimate: ({f_est*1e-3:.4e} +- {f_err*1e-3:.4e})m
        """)
