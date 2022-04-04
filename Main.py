
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import polars as pl
import theano
import seaborn as sns
from statsmodels.stats.power import tt_ind_solve_power
from src.simulations import *
from src.plotting import *
from src.modeling import *
import os
from src.plotting import *
# Convert effect sizes from r to Cohen's D
def r_to_d(r): return r / np.sqrt(1 - r**2)


# Make directories
print('Making directories')
for directory in ['./output/',
                  './output/figures/',
                  './output/tables',
                  './output/sims']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Plot theory related figures

print('plotting figure 1....')
fig_1()

print('plotting figure 2....')
fig_2()

print('plotting figure 3....')
fig_3()

print('plotting SI theory figures...')
SI_Theory_Fig1()
SI_Theory_Fig1(
    N=1000, output='./output/figures/SIFigurePTrue10percentN1000.png')

print('Fitting psychology data...')
# Load RPP Data, randomize direction.
df_psych = pl.read_csv('./data/rpp_data.csv', ignore_errors=True).filter(
    (pl.col('T_pval_USE..R.').is_not_null()) &
    (pl.col('T_pval_USE..O.').is_not_null()) &
    (pl.col('T_r..R.').is_not_null()) &
    (pl.col('T_r..O.').is_not_null())
).with_columns([
    (r_to_d(pl.col('T_r..R.').cast(float))).alias("d_r"),
    (r_to_d(pl.col('T_r..O.').cast(float))).alias("d_o"),
    pl.col('N (R)').cast(int).alias("n_r"),
    pl.col('N (O)').cast(int).alias("n_o"),
]).select([
    'd_o', 'd_r', 'n_o', 'n_r'
])
df_psych.head(5)
df_psych['direction'] = np.random.choice(np.array([-1, 1]), df_psych.shape[0])

idata_psych, trace_psych, prior_psych, ppc_psych = sample_publishing_model(df_psych,
                                                                           prior={'s': 1,
                                                                                  't': 1,
                                                                                  'b': 1})
out_psych = simulate(idata_psych,
                     save_loc='./output/psych_sims.csv')


print('Making plots and tables...')
plot_prior_checks(prior_psych, './output/figures/SIPriorRPP.png')
make_summary_table(idata_psych, './output/tables/psych_posterior.tex')
plot_posterior_distribution(
    idata_psych, save_loc='./output/figures/PosteriorDistributionRPP.png')
plot_posterior_predictive(
    df_psych, ppc_psych, './output/figures/PosteriorPredictiveRPP.png')
plot_sim_figure(idata_psych, out_psych.to_pandas(), df_psych,
                save_loc='./output/figures/Figure4.png')


# plot effect sizes
def plot_es(row):
    sign = np.sign(row['mean'])
    plt.scatter(row['idx'], sign * row['mean'], color='k', alpha=.5)
    plt.plot([row['idx'], row['idx']],
             [sign * row['hdi_3%'], sign * row['hdi_97%']], color='k', alpha=.5)


stemp = az.summary(idata_psych['posterior']['signal']).sort_values(
    by='mean', key=abs, ascending=False).reset_index()
stemp['idx'] = np.arange(stemp.shape[0])
stemp.apply(plot_es, axis=1)

plt.ylim(-.5, 2)
plt.plot([-1, 98], [0, 0], ls='--')
plt.xlim([-1, 98])
plt.ylabel('Effect size')
plt.savefig('./output/figures/SIEstimatedEffects.png', dpi=300)


# Fit econ data
print("fitting econ data....")
df_econ = pl.read_csv(('./data/Camerer2016_raw.csv')).filter(
    (pl.col('eorig').is_not_null()) &
    (pl.col('erep').is_not_null()) &
    (pl.col('nrep_act').is_not_null()) &
    (pl.col('norig').is_not_null())
).with_columns([
    (r_to_d(pl.col('erep').cast(float))).alias("d_r"),
    (r_to_d(pl.col('eorig').cast(float))).alias("d_o"),
    pl.col('nrep_act').cast(int).alias("n_r"),
    pl.col('norig').cast(int).alias("n_o"),
]).select([
    'd_o', 'd_r', 'n_o', 'n_r'
])
df_econ['direction'] = np.random.choice([-1, 1], size=df_econ.shape[0])

idata_econ, trace_econ, prior_econ, ppc_econ = sample_publishing_model(df_econ, prior={'s': .5,
                                                                                       't': .5,
                                                                                       'b': .5})

print("Plotting econ figures....")

plot_prior_checks(prior_econ,
                  './output/figures/SIPriorEcon.png')

make_summary_table(idata_econ,
                   './output/tables/econ_posterior.tex')

plot_posterior_distribution(idata_econ,
                            save_loc='./output/figures/PosteriorDistributionEcon.png')

plot_posterior_predictive(df_econ,
                          ppc_econ,
                          './output/figures/PosteriorPredictiveEcon.png')

prob_sig = np.mean(np.array(pl.read_csv(
    ('./data/Camerer2016_raw.csv'))['porig']) < .05)

out_econ = simulate(idata_econ,
                    save_loc='./output/econ_sims.csv',
                    prob_sig=prob_sig)

plot_sim_figure(idata_econ,
                out_econ.to_pandas(),
                df_econ,
                plot_medians=True,
                rep=.61,
                save_loc='./output/figures/Figure5.png')

print("Fitting preclinical cancer biology model...")
# Load RPP Data, randomize direction.
df_biol = pl.read_csv('./data/RP_CB Final Analysis - Effect level data.csv', ignore_errors=True).with_columns([
    (pl.col('Replication effect size (SMD)').cast(
        float, strict=False)).alias("d_r"),
    (pl.col('Replication sample size').cast(int)).alias("n_r"),
    pl.col('Original effect size (SMD)').cast(
        float, strict=False).alias("d_o"),
    pl.col('Original sample size').cast(int, strict=False).alias("n_o"),
]).select([
    'd_o', 'd_r', 'n_o', 'n_r'
]).filter(
    (pl.col('d_o').is_not_null()) &
    (pl.col('d_r').is_not_null()) &
    (pl.col('n_o').is_not_null()) &
    (pl.col('n_r').is_not_null())
)
df_biol.head(5)
df_biol['direction'] = np.random.choice(np.array([-1, 1]), df_biol.shape[0])
df_biol.head()

idata_biol, trace_biol, prior_biol, ppc_biol = sample_publishing_model(df_biol, prior={'s': .5,
                                                                                       't': .5,
                                                                                       'b': .5})
print("Plotting biol figures....")
plot_prior_checks(prior_biol, './output/figures/SIPriorBiol.png')
make_summary_table(idata_biol, './output/tables/biol_posterior.tex')
plot_posterior_distribution(idata_biol,
                            save_loc='./output/figures/PosteriorDistributionBiol.png')

plot_posterior_predictive(df_biol,
                          ppc_biol,
                          './output/figures/PosteriorPredictiveBiol.png')

out_biol = simulate(
    idata_biol, save_loc='./output/biol_sims.csv', prob_sig=136 / 158)
out_biol.head()

axs = plot_sim_figure(idata_biol, out_biol.to_pandas(), df_biol,
                      save_loc='./output/figures/Figure6.png', plot_medians=True, rep=.43, max_effect=10)
