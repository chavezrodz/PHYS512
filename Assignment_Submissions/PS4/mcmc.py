import pandas as pd
import pprint
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
import time


def prior_chisq(pars, par_priors, par_errs):
    if par_priors is None:
        return 0
    idx = np.argwhere(par_errs != np.inf)
    par_shifts = pars[idx]-par_priors[idx]
    return np.sum((par_shifts/par_errs[idx])**2)


def get_chisq(y_pred, y, Ninv):
    resid=y-y_pred
    if Ninv is not None:
        chisq = resid.T@Ninv@resid
    else:
        chisq = np.sum(resid**2)
    return chisq


def mcmc_step(func, x, y, prev_chisq, Ninv, par_priors, par_errs):
    try:
        model = func(x, 0, inc_derivs=False)
    except Exception as e:
        print("Trial failed, skipping step")
        return False, None

    chisq = get_chisq(model, y, Ninv) + prior_chisq(x, par_priors, par_errs)

    accept_prob = np.exp(-0.5*(chisq-prev_chisq))

    return np.random.rand(1) < accept_prob, chisq


def mcmc(func, x, x_cov, y, niter=1000,
         Ninv=None, par_priors=None, par_errs=None
         ):
    outfile = 'planck_chain' if par_priors is None else 'planck_chain_tauprior'
    outfile = 'Results/' + outfile + '.csv'
    fieldnames = ['step', 'chisq', 'time', 'accepted', 'H0', 'Ohmbh2', 'Ohmch2', 'Tau', 'As', 'ns']

    if os.path.exists(outfile):
        df = pd.read_csv(outfile)
        step = df['step'].iloc[-1] + 1
        prev_chisq = df['chisq'].iloc[-1]
        x = df.values[-1][4:]
    else:
        step = 0
        model = func(x, 0, inc_derivs=False)
        prev_chisq = get_chisq(model, y, Ninv)
        prev_chisq += prior_chisq(x, par_priors, par_errs)

    with open(outfile, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if step == 0:
            writer.writeheader()

        while step < niter:
            start = time.time()
            x_trial = np.random.multivariate_normal(x, x_cov)

            accepted, chisq = mcmc_step(
                func, x_trial, y, prev_chisq,
                Ninv, par_priors, par_errs
                )
            if accepted:
                prev_chisq = chisq
                x = x_trial
            row = {
                'step': step,
                'chisq': prev_chisq,
                'time': time.time() - start,
                'accepted': int(accepted),
                'H0': x[0],
                'Ohmbh2': x[1],
                'Ohmch2': x[2],
                'Tau': x[3],
                'As': x[4],
                'ns': x[5]
            }
            writer.writerow(row)
            pprint.pprint(row)
            print('\n')
            step += 1
