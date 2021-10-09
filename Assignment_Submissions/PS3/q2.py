import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# Problem 2


def days_to_years(x):
    return x/365.25


def hours_to_years(x):
    return days_to_years(x/24)


def minutes_to_years(x):
    return hours_to_years(x/60)


def secs_to_years(x):
    return minutes_to_years(x/60)


# Half lives in years
half_lives = np.array([
    4.468e9,
    days_to_years(24.10),
    hours_to_years(6.7),
    245500,
    75380,
    1600,
    days_to_years(3.8235),
    minutes_to_years(3.10),
    minutes_to_years(26.8),
    minutes_to_years(19.9),
    secs_to_years(164.3e-3),
    22.3,
    5.015,
    days_to_years(138376),
    np.inf
])

n_products = len(half_lives)


def problem_2():
    print('\tProblem 2')
    # write system of ode's in matrix form
    # Negative contribution from diagonal terms
    A = -np.diag(half_lives**-1)
    # off diagonal positive contribution from previous product
    idx = np.where(np.eye(n_products, k=-1) == 1)
    A[idx] = half_lives[:-1]**-1
    A = A*np.log(2)  # Scale half lives by ln(2) to get decay rate

    def decay_step(x, y):
        return A@y

    y0 = np.zeros(n_products)
    y0[0] = 1
    start = 0
    stop = half_lives[0]

    t_eval = np.logspace(start, np.log10(stop))
    scp_ans = scipy.integrate.solve_ivp(
        decay_step, (start, stop), y0, method='Radau', t_eval=t_eval)
    print(
        """
        a)
        We use Radau solver in order to be able to handle a stiff ODE
        """)
    y = scp_ans['y']
    t = scp_ans['t']

    plt.plot(t, y.T)
    plt.title('Decay Products')
    plt.xlabel('Years')
    plt.savefig('Results/problem_2a.png')
    plt.clf()
    plt.close()

    # 2b
    print(
        """
        b)
        We note that the Pb206/U238 ratio approaches 1 after one
        U238 half life, which makes sense if we neglect all other products,
        half of the U238 would be Pb206 then.
        """)
    U238 = y[0]
    Pb206 = y[-1]

    fig, axs = plt.subplots(ncols=2)
    axs[0].set_title('Pb206/U238')
    axs[0].plot(t, Pb206/U238)
    axs[0].set_xlabel('Years')

    print(
        """
        We recompute the ODE to zoom in the Th230/U234 interesting region
        before the ratio stabilizes
        """)

    stop = 15*half_lives[3]
    t_eval = np.logspace(start, np.log10(stop-1))
    scp_ans = scipy.integrate.solve_ivp(
        decay_step, (start, stop), y0, method='Radau', t_eval=t_eval)
    y = scp_ans['y']
    t = scp_ans['t']

    Th230 = y[4]
    U234 = y[3]

    axs[1].set_title('Th230/U234')
    axs[1].plot(t, Th230/U234)
    axs[1].set_xlabel('Years')
    plt.savefig('Results/problem_2b.png')
    plt.clf()
    plt.close()
