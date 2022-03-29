import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import polars as pl
import theano
import seaborn as sns
from statsmodels.stats.power import tt_ind_solve_power
import pyarrow

def sample_publishing_model(df, prior={'s':.5,
                                        't':.5,
                                        'b':.5}):
    with pm.Model() as m_1:
        #Hyperpriors
        sigma = pm.Exponential("sigma", prior['s'])
        tau = pm.Exponential("tau", prior['t'])
        
        bias_sigma =pm.Exponential("bias_sigma", prior['b'])

        #Priors
        noise_r = pm.Normal('noise_o', 0, sigma=sigma,shape=df.shape[0])
        noise_o = pm.Normal('noise_r',0, sigma=sigma,shape=df.shape[0])
        signal = pm.Normal('signal',0,tau,shape=df.shape[0])
        bias = pm.HalfNormal('bias',bias_sigma)

        #Epsilon
        s_o=1/pm.math.sqrt(df.n_o)
        s_r=1/pm.math.sqrt(df.n_r)

        #Model
        d_o = pm.Normal("d_o", mu=noise_o + signal + bias * df.direction, 
                        sigma=s_o, observed=df.direction*df.d_o)
        d_r = pm.Normal("d_r", mu=noise_r + signal, 
                        sigma=s_r, observed=df.direction*df.d_r) 
    with m_1:
        #Prior checks and sampling
        prior_checks = pm.sample_prior_predictive(
            samples=50, 
            random_seed=42
        )

        #Sampling
        trace_1 = pm.sample(
            cores=4,
            target_accept=0.95, 
            random_seed=42
        )

        #Posterior Predictive Checks
        ppc = pm.sample_posterior_predictive(
            trace_1, 
            var_names=["d_o", "d_r", "signal","bias"], 
            random_seed=42
        )

        #Convert to ArViZ
        idata_1 = az.from_pymc3(
            trace_1, 
            prior=prior_checks, 
            posterior_predictive=ppc
        )
        
    return idata_1,trace_1, prior_checks, ppc

def make_summary_table(i_data,save_loc=False):
    table = az.summary(i_data,var_names=['sigma','tau'])
    if save_loc:
        table.to_latex(save_loc)
    return table