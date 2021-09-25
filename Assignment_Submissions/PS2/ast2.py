import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Problem 2
"""
We place Problem 2 before problem one so that we can use the integrator
"""


def simpsons_integrate(dx, y):
    # Write Vectorized version of simpsons
    ans = y[0] + y[-1]
    ans += 4 * y[1:-1:2].sum()
    ans += 2 * y[2:-1:2].sum()
    return ans*dx/3


def integrate_adaptive(fun, a, b, tol, extra=None):
    # print('integrating between ', a, b)
    x = np.linspace(a, b, 5)
    dx = (b-a)/(len(x) - 1)

    if extra is None:
        y = fun(x)
    else:
        y_mid = fun(x[[1, 3]])
        y = extra
        y = np.array([y[0], y_mid[0], y[1], y_mid[1], y[2]])

    area1 = simpsons_integrate(2*dx, y[[0, 2, 4]])
    area2 = simpsons_integrate(dx, y)

    if np.abs(area1-area2) < tol:
        return area2
    else:
        xmid = (b+a)/2
        left = integrate_adaptive(fun, a, xmid, tol/2, extra=y[:3])
        right = integrate_adaptive(fun, xmid, b, tol/2, extra=y[2:])
        return left+right


# Class example to test the integrator
def lorentz(x):
    return 1/(1+x**2)

a = -100
b = 100

ans = integrate_adaptive(lorentz, a, b, 1e-7)
print(ans-(np.arctan(b)-np.arctan(a)))

"""
Assuming 20 iterations, we compare the number of function calls for
lazy vs efficient integrator
"""

n_iterations = 20
N_lazy = [2**(n+1) + 1 for n in range(n_iterations)]
N_efficient = N_lazy[-1]
N_lazy = np.sum(N_lazy)

print("Lazy integrator number of calls:", N_lazy)
print("Efficient integrator number of calls:", N_efficient)
print(f'Efficient vs Lazy N ration: {N_efficient/N_lazy}')


"""
We get about half the number of function calls the efficient way
"""

# Problem 1


def myint_loop(func, zvals):
    """
    We iterate the use of my integrator over z values in a list comprehension
    """
    s = [integrate_adaptive(lambda y: func(y, z=z), a, b, tol) for z in zvals]
    return np.array(s)


def quad_loop(func, zvals):
    """
    We iterate the use of quad integrator over z values in a list comprehension
    """
    s = [quad(lambda y: func(y, z=z), a, b) for z in zvals]

    # We take only the first row which contains the answers, not the errors
    return np.array(s)[:, 0]


# setting all constants to 1
eps_0 = 1
sigma = 1
R = 1

a, b = -1, 1

tol = 1e-3


def func(x, z=1):
    return (z-R*x)*(z**2 + R**2 - 2*z*R*x)**(-3/2)


z_vals = np.linspace(0, 10, 100)
"""
We note that if we include z=R, my integrator crashes because of Runtime error.
Quad does not.
"""


plt.plot(z_vals + 2, myint_loop(func, z_vals), label='My Integrator (z+2)')
plt.plot(z_vals, quad_loop(func, z_vals), label='Quad Integration (z)')
plt.xlabel('Distance from center z')
plt.legend()
plt.savefig('E_field.png')
plt.clf()

# Problem 3

def rescale(x):
    """
    Linearn mapping an x array to [-1,1] range
    We end up not using this function given that numpy's chebyshev fit
    already takes in unscaled inputs

    """
    xmax = x.max()
    xmin = x.min()
    return (x - (xmax + xmin)/2) / ((xmax - xmin)/2)


def log2(x, c, interpolator):
    """
    Returns the log base 2 of x with the chosen interpolator
    """
    mantissa, exps = np.frexp(x)
    if interpolator == 'cheb':
        return np.polynomial.chebyshev.chebval(mantissa, c) + exps
    elif interpolator =='legend':
        return np.polynomial.legendre.legval(mantissa, c) + exps
    elif interpolator == 'poly':
        return np.polynomial.polynomial.polyval(mantissa, c) + exps
    else:
        raise Exception('Interpolator not found')


def mylog2(x, c, interpolator='cheb'):
    """
    Inputs a value, the interpolation coefficients and the chosen interpolator
    
    This computes ln(x) = l_2(x)/l_2(e)

    Note that we write a generalized function to compare all
    interpolations at once
    """
    l2 = log2(x, c, interpolator=interpolator)
    l2e = log2(np.e, c, interpolator=interpolator)
    return l2/l2e


npts = 50
degree = 3

x = np.linspace(0.5, 1, npts)
y = np.log2(x)

x_e = np.linspace(0.5, 10, npts)
y_e = np.log(x_e)


coeffs_c = np.polynomial.chebyshev.chebfit(x, y, deg=degree)
y_c_2 = np.polynomial.chebyshev.chebval(x, coeffs_c)
err_c_2 = y_c_2 - y

y_c_e = mylog2(x_e, coeffs_c, interpolator='cheb')
err_c_e = y_c_e - y_e

coeffs_l = np.polynomial.legendre.legfit(x, y, deg=degree)
y_l_2 = np.polynomial.legendre.legval(x, coeffs_l)
err_l_2 = y_l_2 - y

y_l_e = mylog2(x_e, coeffs_l, interpolator='legend')
err_l_e = y_l_e - y_e


coeffs_p = np.polynomial.polynomial.polyfit(x, y, deg=degree)
y_p_2 = np.polynomial.polynomial.polyval(x, coeffs_p)
err_p_2 = y_p_2 - y

y_p_e = mylog2(x_e, coeffs_p, interpolator='poly')
err_p_e = y_p_e - y_e


fig, axs = plt.subplots(2, 6, sharex=False, figsize=(18, 12))
axs[0, 0].plot(x, y, '--', label='data')
axs[0, 0].plot(x, y_c_2, '-.', label='Chebyshev Fit')
axs[0, 0].set_title(r'$log_2(x)$ Chebyshev Fit')
axs[0, 0].legend()
axs[1, 0].plot(x, err_c_2)
axs[1, 0].set_title(r'$log_2(x)$ Fit Residuals')

axs[0, 1].plot(x_e, y_c_e, '--', label='data')
axs[0, 1].plot(x_e, y_c_e, '-.', label='Mylog2')
axs[0, 1].set_title(r'$ln(x)$ Chebyshev Fit')
axs[0, 1].legend()
axs[1, 1].plot(x_e, err_c_e)
axs[1, 1].set_title(r'$ln(x)$ Fit Residuals')

axs[0, 2].plot(x, y, '--', label='data')
axs[0, 2].plot(x, y_l_2, '-.', label='Legendre Fit')
axs[0, 2].set_title(r'$log_2(x)$ Legendre Fit')
axs[0, 2].legend()
axs[1, 2].plot(x, err_l_2)
axs[1, 2].set_title(r'$log_2(x)$ Fit Residuals')

axs[0, 3].plot(x_e, y_l_e, '--', label='data')
axs[0, 3].plot(x_e, y_l_e, '-.', label='Mylog2')
axs[0, 3].set_title(r'$ln(x)$ Legendre Fit')
axs[0, 3].legend()
axs[1, 3].plot(x_e, err_l_e)
axs[1, 3].set_title(r'$ln(x)$ Fit Residuals')

axs[0, 4].plot(x, y, '--', label='data')
axs[0, 4].plot(x, y_p_2, '-.', label='Poly Fit')
axs[0, 4].set_title(r'$log_2(x)$ Poly Fit')
axs[0, 4].legend()
axs[1, 4].plot(x, err_p_2)
axs[1, 4].set_title(r'$log_2(x)$ Fit Residuals')

axs[0, 5].plot(x_e, y_p_e, '--', label='data')
axs[0, 5].plot(x_e, y_p_e, '-.', label='Mylog2')
axs[0, 5].set_title(r'$ln(x)$ Poly Fit')
axs[0, 5].legend()
axs[1, 5].plot(x_e, err_p_e)
axs[1, 5].set_title(r'$ln(x)$ Fit Residuals')

plt.tight_layout()
plt.savefig('logs_fit_comparison.png')
plt.clf()

"""
We note no visible difference between the proposed fits in the bonus,
hence we compare the predictions of legendre fit and polynomial fit
to chebyshev
"""
plt.title('Difference between Natural log predictions')
plt.plot(x_e, y_c_e - y_l_e, label='Chebyshev vs Legendre')
plt.plot(x_e, y_c_e - y_p_e, label='polynomial vs Legendre')
plt.legend()
plt.savefig('pred_difference_ln.png')
plt.clf()
