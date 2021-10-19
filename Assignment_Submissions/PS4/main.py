from uncertainties import ufloat
import shutil
import os
import pandas as pd
import numpy as np
from planck_likelihood import get_spectrum
from lm import fit_lm
from mcmc import mcmc
import time


outdir = 'Results'
# shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)

data = np.loadtxt('mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt',
                  skiprows=1)
x = data[:, 0]
y = data[:, 1]
y_err = np.mean(data[:, 2:4], axis=1)
n_y = len(y)
Ninv = np.diag(y_err**-2)


n_iter_mcmc = 5000
n_iter_lm = 200


def get_chisq(y_pred, y, y_err):
    resid = y-y_pred
    chisq = np.sum((resid/y_err)**2)
    return chisq


def derivative(func, x, n_y, eps=1e-2, order=4):
    n_x = len(x)
    df = np.zeros((n_y, n_x))
    for i in range(n_x):
        dx = np.zeros(n_x)
        dx[i] = eps*x[i]
        if order == 4:
            df[:, i] = 8*(func(x + dx)[:n_y] - func(x - dx)[:n_y])
            df[:, i] += -func(x + 2*dx)[:n_y] + func(x - 2*dx)[:n_y]
            df[:, i] /= 12*dx[i]
        else:
            df[:, i] = (func(x + dx)[:n_y] - func(x - dx)[:n_y])/(2*dx[i])
    return df


def spectrum_adapted(m, x, inc_derivs=True):
    model = get_spectrum(m)[:n_y]
    if inc_derivs:
        derivs = derivative(get_spectrum, m, n_y) if inc_derivs else None
        return model, derivs
    else:
        return model


def problem_1():
    """
    Compare chisq for test params and given params. Which is closer
    to accepted value? Is it acceptable?
    """
    test_pars = np.asarray([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
    new_pars = np.array([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])

    test_model = get_spectrum(test_pars)[:n_y]
    test_chisq = get_chisq(test_model, y, y_err)

    new_model = get_spectrum(new_pars)[:n_y]
    new_chisq = get_chisq(new_model, y, y_err)
    n_dof = len(data)-len(test_pars)

    print(
        f"""
        Problem 1

        For {n_dof} degrees of freedom,
        We have chi-squared values of {test_chisq} and {new_chisq}
        for the test values and the new given values respectively.

        Given that the mean and variance of chi-squared is 
        {n_dof} and {2*n_dof}, a fit within one standard deviation would be
        {n_dof} +- {np.sqrt(2*n_dof)}. Since both tested parameters have a 
        chi-squared value outside of this range, these are not a good fit.

        The new tested parameters have a chi-squared closer
        to the accepted value.
        """
        )

# problem_1()


def problem_2():
    pars = np.array([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
    fit_lm(pars, spectrum_adapted, x, y,
                 Ninv=Ninv, niter=n_iter_lm)

    print(
        f"""
        Problem 2

        We fit the model using an LM fitter and export the paramaters along with
        the covariance matrix.

        We note that our new chi-squared value 2576.153 is much closer to the accepted value.
        """
        )

# problem_2()


def problem_3():
    pars = np.array([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
    cov = np.loadtxt('Results/covariance.txt')
    mcmc(spectrum_adapted, pars, cov, y,
         Ninv=Ninv, niter=n_iter_mcmc)

    chain = pd.read_csv('Results/planck_chain.csv').iloc[150:]

    chisq = ufloat(chain['chisq'].mean(), chain['chisq'].std())

    Hubble = ufloat(chain['H0'].mean(), chain['H0'].std())
    ohmbh2 = ufloat(chain['Ohmbh2'].mean(), chain['Ohmbh2'].std())
    omch2 = ufloat(chain['Ohmch2'].mean(), chain['Ohmch2'].std())

    omega_dark = 1 - (100/Hubble)**2 * (ohmbh2 + omch2)

    print(
    f"""
    Problem 3

    We fit the model using an MCMC with the covariance matrix from problem 2
    and export the chain to Results/planck_chain.csv

    We believe the chain has converged since it roughly stabilizes after
    about 150 steps, and the parameters plot with respect to steps look like noise
    after removing the burning period. Looking at the FFT of the parameters evolution,
    we also observe a flat region in the low frequencies, indicating convergence.
    These plots are produced in display.py and can be found in Results/chain/.

    Moreover, the average chis-squared value
    after removing the burning period is {chisq}, which is about one standard
    deviation close to the accepted value presented in problem one.

    After removing the burn-in period, we take the average of the values and 
    standard errors for the Hubble constant, the Baryon density and the
    Dark matter density. then solve for the dark energy.

    we find that 

    \Omega_\Lambda = 1 - (100/H_0)^2 (\Omega_C h^2 + \Omega_b h^2)

    which gives us \Omega_\Lambda = {omega_dark}


    """
        )



# problem_3()


def problem_4():
    # Setting constants
    tau = 0.0540
    tau_err = 0.0074

    # Importance sampling from Q3
    # Note that we reject all points in the burn-in region
    chain = pd.read_csv('Results/planck_chain.csv').iloc[150:]
    chain_tau = chain['Tau']

    sq_prior = np.square((chain_tau - tau) / tau_err)
    new_likelihood = np.exp(-0.5 * sq_prior) / np.exp(-0.5 * sq_prior).sum()

    new_means = np.average(chain.values[:, -6:],
                           axis=0, weights=new_likelihood)
    cov_tau = np.cov(chain.values[:, -6:].T, aweights=new_likelihood)
    new_errs = np.sqrt(np.diag(cov_tau))

    # Setting priors for MCMC
    par_priors = np.zeros(6)
    par_errs = np.ones(6) * np.inf
    par_priors[-3], par_errs[-3] = tau, tau_err
    pars = np.array([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])

    mcmc(spectrum_adapted, pars, cov_tau, y,
         Ninv=Ninv, niter=n_iter_mcmc,
         par_priors=par_priors, par_errs=par_errs
        )

    tau_burn = 200
    tau_chain = pd.read_csv('Results/planck_chain_tauprior.csv').iloc[tau_burn:]
    mcmc_tau_values = tau_chain.values[:, -6:]
    mcmc_tau_means = np.mean(mcmc_tau_values, axis=0)
    mcmc_tau_err = np.std(mcmc_tau_values, axis=0)
    print(
        f"""
        Problem 4
        We first compute the results by importance sampling our chain from
        question 3 and we find a value for 
        tau =  {new_means[-3]} +- {new_errs[-3]}
        which is close to the prior value, as expected.

        The results for all other parameters are as follows:
        H0: {new_means[0]} +- {new_errs[0]}
        Ohmbh2: {new_means[1]} +- {new_errs[1]}
        Ohmch2: {new_means[2]} +- {new_errs[2]}
        As: {new_means[4]} +- {new_errs[4]}
        ns: {new_means[5]} +- {new_errs[5]}


        Using the same weights from importance sampling, we compute 
        the covariance matrix which we use in the new mcmc we run with
        tau prior knowledge.

        we then find the following values:

        H0: {mcmc_tau_means[0]} +- {mcmc_tau_err[0]}
        Ohmbh2: {mcmc_tau_means[1]} +- {mcmc_tau_err[1]}
        Ohmch2: {mcmc_tau_means[2]} +- {mcmc_tau_err[2]}
        Tau: {mcmc_tau_means[3]} +- {mcmc_tau_err[3]}
        As: {mcmc_tau_means[4]} +- {mcmc_tau_err[4]}
        ns: {mcmc_tau_means[5]} +- {mcmc_tau_err[5]}

        While adding the tau prior to the mcmc yielded a tau value closer to
        prior, the value we obtained with importance sampling was close to
        the prior already.
        """
        )


problem_4()   