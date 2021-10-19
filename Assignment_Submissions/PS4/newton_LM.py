import pandas as pd
import numpy as np
from planck_likelihood import get_spectrum, get_chisq
import time


def derivative(func, x, idx, eps=1e-2, order=4, ):
    n_x = len(x)
    df = np.zeros((len(idx), n_x))

    for i in range(n_x):
        dx = np.zeros(n_x)
        dx[i] = eps*x[i]
        if order == 4:
            df[:, i] = 8*(func(x + dx)[idx] - func(x - dx)[idx])
            df[:, i] += -func(x + 2*dx)[idx] + func(x - 2*dx)[idx]
            df[:, i] /= 12*dx[i]
        else:
            df[:, i] = (func(x + dx)[idx] - func(x - dx)[idx])/(2*dx[i])
    return df


def fit_newton(m, y, err_y, func,
               niter=10, pc_sample=1, lambd=1, step_tol=0.01):
    n_y = len(y)
    n_sample = int(pc_sample*n_y)
 
    """
    We select the pc_sample % number of points that are responsible
    for the larger part of the chisq. To select this sample, we
    initialize it by taking all points at first. Unfortunately, this
    doesnt result being very helpful
    """
    idx_top = np.arange(n_y)
    prev_chisq = np.inf

    for i in range(niter):
        t1 = time.time()
        N_inv = np.diag(err_y[idx_top]**-2)

        model = func(m)
        derivs = derivative(func, m, idx_top, order=2)

        r = y[idx_top] - model[idx_top]

        lhs_0 = derivs.T@N_inv@derivs
        rhs = derivs.T@N_inv@r

        lhs = lhs_0 + lambd*np.diag(np.diag(lhs_0))
        dm = np.linalg.inv(lhs)@rhs
        m_tmp = m + dm
        tmp_chisq = np.square((func(m_tmp)[:n_y] - y)/err_y).sum()
        print(f'tmp chisq:{tmp_chisq:.2f}')

        chi_rel_diff = (tmp_chisq - prev_chisq)/prev_chisq

        if np.abs(chi_rel_diff) > step_tol:
            print(f'Relative Chisq diff:{chi_rel_diff:.3f}, taking step')
            lambd /= np.sqrt(2)
            m = m_tmp
        elif np.abs(chi_rel_diff) < step_tol:
            while np.abs(chi_rel_diff) < step_tol:
                print(f'Relative Chisq diff:{chi_rel_diff:.3f}, increasing lamda')
                lambd *= 2
                lhs = lhs_0 + lambd*np.diag(np.diag(lhs_0))
                dm = np.linalg.inv(lhs)@rhs
                m_tmp = m + dm
                tmp_chisq = np.square((func(m_tmp)[:n_y] - y)/err_y).sum()
                chi_rel_diff = (tmp_chisq - prev_chisq)/prev_chisq
                print(f'tmp chisq:{tmp_chisq:.2f}')
                if chi_rel_diff < 0.01 and lambd < 5:
                    break

        m = m_tmp

        chisq = np.square((func(m)[:n_y] - y)/err_y)
        idx_top = np.argpartition(chisq, -n_sample)[-n_sample:]
        chisq_top = chisq[idx_top].sum()

        prev_chisq = chisq.sum()
        top_chisq_pc = 100*chisq_top/prev_chisq

        dt = time.time() - t1
        print(
            f'Step: {i} Chisq: {prev_chisq:.2f} '
            # f'Top {100*pc_sample}% pts Chisq: {top_chisq_pc:.2f}% '
            f'lambd: {lambd:.3f} '
            f'Time: {dt:.2f}s\n')

    return m, chisq
