import os
import numpy as np
import csv
from matplotlib import pyplot as plt


def prior_chisq(pars, par_priors, par_errs):
    if par_priors is None:
        return 0
    idx = np.argwhere(par_errs != np.inf)
    par_shifts = pars[idx]-par_priors[idx]
    return np.sum((par_shifts/par_errs[idx])**2)


def update_lamda(lamda, success):
    if success:
        lamda = lamda/1.5
        if lamda < 0.5:
            lamda = 0
    else:
        if lamda == 0:
            lamda = 1
        else:
            lamda = lamda*1.5**2
    return lamda


def get_matrices(m, fun, x, y, Ninv=None):
    model, derivs = fun(m, x)
    r=y-model
    if Ninv is None:
        lhs=derivs.T@derivs
        rhs=derivs.T@r
        chisq=np.sum(r**2)
    else:
        lhs=derivs.T@Ninv@derivs
        rhs=derivs.T@(Ninv@r)
        chisq=r.T@Ninv@r
    return chisq, lhs, rhs


def linv(mat,lamda):
    mat=mat+lamda*np.diag(np.diag(mat))
    return np.linalg.inv(mat)


def fit_lm(m, fun, x, y, Ninv=None, niter=10, chitol=0.01,
           par_priors=None, par_errs=None):
    outfile = 'Results/planck_fit_params'
    log_file = 'Results/LM_logs'
    params_file = 'Results/best_params'
    cov_file = 'Results/covariance'

    if par_priors is not None:
        outfile += '_tauprior'
        log_file += '_tauprior'
        params_file += '_tauprior'
        cov_file += '_tauprior'

    outfile += '.txt'
    log_file += '.txt'
    params_file += '.txt'
    cov_file += '.txt'

    lamda = 0
    chisq, lhs, rhs = get_matrices(m, fun, x, y, Ninv)
    chisq += prior_chisq(m, par_priors, par_errs)
    with open(log_file, 'w') as f:
        for i in range(niter):
            lhs_inv=linv(lhs, lamda)
            dm=lhs_inv@rhs
            chisq_new, lhs_new, rhs_new=get_matrices(m+dm,fun,x,y,Ninv)
            chisq_new += prior_chisq(m+dm, par_priors, par_errs)
            accept = chisq_new < chisq
            rel_diff = np.abs(chisq-chisq_new)/chisq
            if accept:
                if lamda==0:
                    if rel_diff < chitol:
                        f.write(f'Converged after {i} iterations of LM')
                        break
                chisq = chisq_new
                lhs = lhs_new
                rhs = rhs_new
                m = m+dm

            row = f'Iteration: {i} | chisq: {chisq:.3f} | Change: {100*rel_diff:.3f}% | step: {dm.round(3)} | lamda: {lamda}\n'
            print(row)
            lamda = update_lamda(lamda, accept)
            f.write(row)

    cov = np.linalg.inv(lhs)
    errs = np.sqrt(np.diag(cov))
    with open(outfile, 'w') as f:
        f.write(
            f'Best fit parameters:\n'
            f'{m}\n'
            f'Errors:\n'
            f'{errs}'
            )
    np.savetxt(params_file, m)
    np.savetxt(cov_file, cov)

    return m, cov
