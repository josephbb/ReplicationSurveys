import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
import polars as pl
import seaborn as sns
from statsmodels.stats.power import tt_ind_solve_power
import pyarrow
import scipy as sp



def simulate_reproducibility(row, i_data, pubs=1000, prob_sig=96.0 / 99.0):
    count = row[0]
    N = row[1]
    J = row[2]
    alpha = row[3]
    t_c = (1 / np.sqrt(N)) * sp.stats.norm.ppf(1 - alpha/ 2)

    out_p_success = np.zeros(J)
    out_p_rep_success = np.zeros(J)
    out_type_s = np.zeros(J)
    out_observed_es = np.zeros(J)
    out_true = np.zeros(J)
    out_reversal = np.zeros(J)
    out_d_reps = np.zeros(J)

    for jidx in range(J):
        chain = np.random.choice(4, 1)
        sample = np.random.choice(1000, 1)

        tau = i_data['posterior']['tau'].values[chain, sample]
        signals = np.abs(np.random.normal(0,tau,size=1000)).ravel()
        sigma = i_data['posterior']['sigma'].values[chain, sample]
        d_orig = np.random.normal(signals, sigma)

        significant = np.random.binomial(1,
                                        tt_ind_solve_power(
                                        effect_size=d_orig,
                                        alpha=alpha,
                                        nobs1=np.int(N/2.0)))
        p_success = np.mean(significant)
        published = np.random.binomial(1,(prob_sig * significant) + (1 - prob_sig) * \
                                       (1 - significant))

        observed_es = d_orig[published == 1]
        actual_es = signals[published == 1]

        type_s = np.sign(signals) * d_orig < -t_c


        d_rep = np.random.normal(signals[published==1], sigma)


        p_rep_success = np.mean(np.random.binomial(1,
                                        tt_ind_solve_power(
                                        effect_size=d_rep,
                                        alpha=alpha/2.0,
                                        nobs1=np.int(N/2.0),
                                        alternative='larger')))

        reversal = np.random.binomial(1,
                                        tt_ind_solve_power(
                                        effect_size=d_rep,
                                        alpha=alpha/2.0,
                                        nobs1=np.int(N/2.0),
                                        alternative='smaller'))
        out_p_success[jidx] = p_success
        out_p_rep_success[jidx] = p_rep_success
        out_type_s[jidx] = np.mean(type_s)
        out_observed_es[jidx] = np.mean(np.abs(observed_es))
        out_true[jidx] = np.mean(signals[published == 1])
        out_reversal[jidx] = np.mean(reversal)
        out_d_reps[jidx] = np.mean(d_rep)
    return    (np.mean(out_p_success),
                np.mean(out_p_rep_success),
                np.mean(out_type_s),
                np.mean(out_observed_es),
                np.mean(out_true),
                np.mean(out_reversal),
                np.mean(out_d_reps))

def generate_simulation_function(i_data, prob_sig=96.0 / 99.0):
    return lambda row: simulate_reproducibility(row, i_data, prob_sig)


def simulate(i_data,
             N=np.linspace(10, 500, 40, dtype=int),
             j=500,
             alpha=np.array([.005, .01, .05, .1, .2]),
             save_loc=False, prob_sig=96.0 / 99.0):
    replication_simulation_dictionaries = []
    for n in N:
        for a in alpha:
            replication_simulation_dictionaries.append(
                {'N': n, 'j': j, 'alpha': a})

    out = pl.DataFrame(replication_simulation_dictionaries).with_row_count().apply(
        generate_simulation_function(i_data)).rename(
        dict(zip(['column_' + str(idx) for idx in range(7)],
                 ['publication_rate',
                  'replication_rate',
                  'type_s_error',
                  'published_es',
                  'actual_es',
                  'reversals',
                  'replication_es']))
    ).with_row_count().join(
            pl.DataFrame(replication_simulation_dictionaries).with_row_count(),
            on='row_nr'
    ).drop('row_nr')
    if save_loc:
        out.to_csv(save_loc)

    return out
