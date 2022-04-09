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
import scipy as sp
def sample_publishing_model(df, prior={'s':1,
                                        't':1, 
                                       'b':1}):

    with pm.Model() as m_1:
        
        #Hyperpriors
        sigma = pm.Exponential("sigma", prior['s'])#
        tau = pm.Exponential("tau", prior['t'])
        beta = pm.Exponential("beta", prior['b'])

        #Priors
        signal = pm.Normal('signal', sigma=tau, shape=df.shape[0])
        noise_o = pm.Normal('noise_o', 0, sigma=sigma, shape=df.shape[0])
        noise_r = pm.Normal('noise_r', 0, sigma=sigma, shape=df.shape[0])
        bias = pm.HalfNormal('bias', sigma=beta)#, shape=df.shape[0])

        #Epsilon
        s_o=1/pm.math.sqrt(df.n_o)
        s_r=1/pm.math.sqrt(df.n_r)

        #Model
        d_o = pm.Normal("d_o",  mu=signal*bias + noise_o,
                                 sigma=s_o,
                                 observed=df.d_o*df.direction)
        d_r = pm.Normal("d_r", mu=signal+noise_r, 
                        sigma=s_r, observed=df.d_r*df.direction) 
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
            max_treedepth=12,
            random_seed=42
        )

        #Posterior Predictive Checks
        ppc = pm.sample_posterior_predictive(
            trace_1, 
            var_names=["d_o", "d_r"], 
            random_seed=42
        )

        #Convert to ArViZ
        dims = {
            "d_o": ["effect_size"],
            "d_r": ["effect_size"],
        }
        idata_1 = az.from_pymc3(
            trace_1, 
            prior=prior_checks, 
            dims=dims,
            posterior_predictive=ppc
        )
        
    idata_1.log_likelihood["overall"] = (
    idata_1.log_likelihood.d_o + idata_1.log_likelihood.d_r
    )
    return idata_1,trace_1, prior_checks, ppc

def make_summary_table(i_data,save_loc=False):
    table = az.summary(i_data,var_names=['sigma','tau'])
    table = table.append(table.iloc[1] * np.sqrt(2/np.pi),).reset_index()
    table['index'] = (['sigma', 'tau', 'Avg. effect'])
    table = table.set_index('index').round(2)
    if save_loc:
        table.to_latex(save_loc)
    return table

def alternate_publishing_model(df, prior={'s':.5,
                                        't':.5, 
                                       'b':.5}):

    with pm.Model() as m_1:
        a_sigma = pm.Exponential('a_sigma', 2)
        b_sigma = pm.Gamma('b_sigma',mu=1, sigma=.5)
        
        #Hyperpriors
        tau = pm.Exponential("tau", prior['t'])
        beta = pm.Exponential("beta", prior['b'])

        #Priors
        signal = pm.Normal('signal', sigma=tau, shape=df.shape[0])
        
        s = a_sigma + b_sigma * pm.math.abs_(signal)

        noise_o = pm.Normal('noise_o', 0, sigma=s, shape=df.shape[0])
        noise_r = pm.Normal('noise_r', 0, sigma=s, shape=df.shape[0])
        bias = pm.HalfNormal('bias', sigma=beta)#, shape=df.shape[0])

        #Epsilon
        #s_o=pm.math.sqrt(pm.math.sqr(sigma) +pm.math.sqr(1/pm.math.sqrt(df.n_o)))
        #s_r=pm.math.sqrt(pm.math.sqr(sigma) +pm.math.sqr(1/pm.math.sqrt(df.n_r)))
        s_o=1/pm.math.sqrt(df.n_o)
        s_r=1/pm.math.sqrt(df.n_r)


        #Model
        d_o = pm.Normal("d_o", mu=bias*(signal + noise_o),
                                 sigma=s_o,
                                 observed=df.d_o*df.direction)
        d_r = pm.Normal("d_r", mu=signal+noise_r, 
                        sigma=s_r, observed=df.d_r*df.direction) 
        
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
            max_treedepth=12,
            random_seed=42
        )

        #Posterior Predictive Checks
        ppc = pm.sample_posterior_predictive(
            trace_1, 
            var_names=["d_o", "d_r"], 
            random_seed=42
        )
        #Convert to ArViZ
        dims = {
            "d_o": ["effect_size"],
            "d_r": ["effect_size"],
        }
        #Convert to ArViZ
        idata_1 = az.from_pymc3(
            trace_1, 
            prior=prior_checks, 
            dims=dims,
            posterior_predictive=ppc
        )
        
    idata_1.log_likelihood["overall"] = (
    idata_1.log_likelihood.d_o + idata_1.log_likelihood.d_r
    )
        
    return idata_1,trace_1, prior_checks, ppc