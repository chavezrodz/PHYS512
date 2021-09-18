import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# Global Variables
eps = 2**-52


# Problem 1

def four_pt_deriv(x, func, dx):
    """
    Inputs:
        x: point of interest
        func: function to derive
        dx: differential size
    Output:
        Estimated Derivative

    """
    df = 8*(func(x + dx) - func(x - dx))
    df += -func(x + 2*dx) + func(x - 2*dx)
    df /= 12*dx
    return df


def derivative_error(x, func, dfunc, dfunc5):
    """
    Inputs:
        x: point of interest
        func: function to derive
        func5: fifth theoretical derivative
        dfunc: theoeretical derivative
    Output:
        Estimated error
    """
    dx = ((45*eps*func(x)) / (4*dfunc5(x)))**(1/5)

    df = four_pt_deriv(x, func, dx)

    return np.abs(df-dfunc(x))


def problem_1(a=0.01, x=2):

    def func(y):
        return np.exp(a*y)

    def dfunc(y):
        return a*func(y)

    def dfunc5(y):
        return a**5 * func(y)

    print('Problem 1')
    print(f'Four point derivative error exp(x) is: '
          f'{derivative_error(x, np.exp, np.exp, np.exp)}')
    print(f'Four point derivative error exp({a}*x) is: '
          f'{derivative_error(x, func, dfunc, dfunc5)}\n')


problem_1()


# Problem 2

def ndiff(fun, x, full=False):
    # Calculate dx
    dx = eps**(1/3)

    x1 = x + dx 
    # Recalculate dx as seen in class
    dx = x1 - x

    df = (fun(x + dx) - fun(x - dx)) / (2*dx)

    # Roundoff error from class
    err = (eps*fun(x))/dx

    if not full:
        return df
    else:
        return df, dx, err


# Problem 3


dat = np.loadtxt('lakeshore.txt')


def Lakeshore(V, data):

    n_pts = 100
    # Load Data
    T, Volt, dVdT = data[:, 0], data[:, 1], data[:, 2]

    #  Initially V vs T shape, we want T vs V
    T, Volt = np.flip(T), np.flip(Volt)

    """
    We estimate the error by interpolating based on a sample,
    then test on the rest
    """
    idx = sorted(np.random.choice(len(Volt), size=len(Volt)//2, replace=False))
    idx_complement = np.arange(len(Volt))
    idx_complement = idx_complement[~np.isin(idx_complement, idx)]

    spln = interpolate.CubicSpline(Volt[idx], T[idx])

    # Visualizing Interpolation
    V_dummy = np.linspace(Volt.min(), Volt.max(), n_pts)
    T_interp = spln(V_dummy)

    plt.plot(Volt[idx], T[idx], '.', label='data')
    plt.plot(Volt[idx_complement], T[idx_complement], '*', label='Test set')
    plt.plot(V_dummy, T_interp, label='interpolation')
    plt.xlabel("Voltage")
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig('Lakeshore.png')
    plt.clf()
    """
    Calculate the error on the complement of the spline sample, then
    make another spline to estimate the error at a point
    """
    err = np.abs(spln(Volt[idx_complement]) - T[idx_complement])
    spln_err = interpolate.CubicSpline(Volt[idx_complement], err)
    plt.plot(Volt[idx_complement], err, '*')
    plt.plot(Volt[idx_complement], spln_err(Volt[idx_complement]))
    plt.xlabel("Voltage")
    plt.ylabel("Error")
    plt.savefig('Error')

    return spln(V), spln_err(V)


def problem_3():
    x = np.linspace(0.2, 1.5, 5)    
    y, y_err = Lakeshore(x, dat)

    print('Problem 3')
    for i in range(len(y)):
        print('Temperature: '+repr(y[i]) + '+/-' + str(y_err[i]))
    print(f"Average temperature error : {y_err.mean()}\n")


problem_3()

# Problem 4

# funcs from class notes


def rat_eval(p, q, x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot


def rat_fit(x, y, n, m, pinv):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    if pinv:
        pars=np.dot(np.linalg.pinv(mat), y)
    else:
        pars=np.dot(np.linalg.inv(mat), y)
    p=pars[:n]
    q=pars[n:]
    return p, q


def rational(x, y, m, n, pinv):
    p, q = rat_fit(y, x, m, n, pinv)
    pred = rat_eval(p, q, x)
    return pred, np.std(pred-y)


def spline(x, y):
    spln = interpolate.CubicSpline(x, y)
    pred = spln(x)
    return pred, np.std(pred-y)


def poly(x, y, n):
    pf = np.polyfit(x, y, n)
    pred = np.polyval(pf, x)
    return pred, np.std(pred-y)


def triple_fit(x, y, n_poly, m_n_rat, pinv):
    _, e1 = poly(x, y, n_poly)
    _, e2 = spline(x, y)

    mr, nr = m_n_rat
    _, e3 = rational(x, y, mr, nr, pinv=pinv)
    print("polynomial interpolation Standard Error:", e1)
    print("spline interpolation Standard Error:", e2)
    print("rational function Standard Error:", e3)
    print('\n')


print('Problem 4')

n_pts = 8
n_poly = 3
m_n_rat = (5, 4)
pinv = False


print('Using Standard Inverse\n')
x1 = np.linspace(-np.pi/2, np.pi/2, n_pts)
y1 = np.cos(x1)
print('Cosine Results')
triple_fit(x1, y1, n_poly, m_n_rat, pinv=pinv)

x2 = np.linspace(-1, 1, n_pts)
y2 = 1/(1 + x2**2)
print('Lorentzian Results')
triple_fit(x2, y2, n_poly, m_n_rat, pinv=pinv)


pinv = True
print('Using Pseudo-Inverse\n')
x1 = np.linspace(-np.pi/2, np.pi/2, n_pts)
y1 = np.cos(x1)
print('Cosine Results')
triple_fit(x1, y1, n_poly, m_n_rat, pinv=pinv)

x2 = np.linspace(-1, 1, n_pts)
y2 = 1/(1 + x2**2)
print('Lorentzian Results')
triple_fit(x2, y2, n_poly, m_n_rat, pinv=pinv)

"""
The error for the lorentzian should be zero since it is
a rational function.

However, when the order is (4,5) the rational fit is the poorest
by ~16 orders of magnitude.

When using Pinv, the rational interpolation remains the poorest,
but performance is greatly increased.

This is due to the higher order polynomials in the rational function
being set to zero, rather than causing heavy oscillations
"""
