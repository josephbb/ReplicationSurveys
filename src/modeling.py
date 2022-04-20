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
def sample_publishing_model(df, prior={'s':1,
                                        't':1,
                                        'b':1}):

    with pm.Model() as m_1:

        #Hyperpriors
        sigma = pm.Exponential("sigma", prior['s'])#
        tau = pm.Exponential("tau", prior['t'])

        #Priors
        signal = pm.Normal('signal', sigma=tau, shape=df.shape[0])

        #Epsilon
        s_o=pm.math.sqrt(pm.math.sqr(sigma) + pm.math.sqr(1/pm.math.sqrt(df.n_o.values)))
        s_r=pm.math.sqrt(pm.math.sqr(sigma) + pm.math.sqr(1/pm.math.sqrt(df.n_r.values)))

        #Model
        d_o = pm.TruncatedNormal("d_o",  mu=signal,
                                 sigma=s_o,
                                 lower=df.lower.values,
                                 upper=df.upper.values,
                                 observed=df.d_o*df.direction)
        d_r = pm.Normal("d_r", mu=signal,
                        sigma=s_r, observed=df.d_r*df.direction)
    with m_1:
        dims={
                "d_r": ["study"],
                "d_o": ["study"],
        }
        #Prior checks and sampling
        prior_checks = pm.sample_prior_predictive(
            samples=50,
            random_seed=42,
            idata_kwargs={'dims':dims},

        )

        #Sampling
        idata_1 = pm.sample(
            idata_kwargs={'dims':dims},
            cores=4,
            nuts={"max_treedepth":12,
                  "target_accept":0.95},
            random_seed=42,

        )

        #Posterior Predictive Checks
        ppc = pm.sample_posterior_predictive(
            idata_1,
            var_names=["d_o", "d_r"],
            random_seed=42,
            idata_kwargs={'dims':dims}

        )

        idata_1.extend(ppc)
        idata_1.extend(prior_checks)

    return idata_1, m_1

def make_summary_table(i_data,save_loc=False):
    table = az.summary(i_data,var_names=['sigma','tau'])
    table['index'] = (['sigma', 'tau',])
    table = table.set_index('index').round(2)
    if save_loc:
        table.to_latex(save_loc)
    return table
