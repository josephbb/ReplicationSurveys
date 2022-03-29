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


def plot_prior_checks(prior,save_location=False):
    
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    az.plot_dist(prior['d_o'],color='orange',label='original prior sim')
    az.plot_dist(prior['d_r'],label='replication prior sim')
    plt.xlabel('Prior predictive effect size')
    plt.ylabel('Density')

    plt.subplot(122)
    plt.scatter(prior['d_o'][45,:],prior['d_r'][45,:])
    plt.xlabel('Prior Sim Original Effect Size')
    plt.ylabel('Prior Sim Replication Effect Size')
    plt.tight_layout()
    if save_location:
        plt.savefig(save_location, dpi=300) 


def plot_posterior_distribution(i_data, save_loc=False,dpi=300):
    sns.set_palette(sns.color_palette("colorblind", n_colors=6))
    az.plot_ppc(i_data)
    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc, dpi=dpi) 

def plot_posterior_predictive(df, ppc, save_loc=False, dpi=300):
    _, ax = plt.subplots()

    ax.plot(df['d_o'].to_numpy(), df['d_r'].to_numpy(),'o',color='orange',alpha=.5,label='Data')
    ax.plot(df['direction'].to_numpy()*ppc['d_o'].mean(axis=0), 
           df['direction'].to_numpy()*ppc['d_r'].mean(axis=0),'o',c='k',alpha=.5,label='Posterior Predictive')
    plt.xlabel('Original Effect Size')
    plt.ylabel('Replication Effect Size')
    plt.legend()
    if save_loc:
        plt.savefig(save_loc, dpi=300) 

        
    

def plot_fig4a(samples_full,axs,max_effect=2):
    pal = sns.color_palette("colorblind", n_colors=6)

    sns.kdeplot(np.abs(np.array(samples_full['tau']).ravel()),label='True Effect Size',clip=[0,10],
                shade=True,color=pal[0],ax=axs)
    sns.kdeplot(np.abs(np.array(samples_full['sigma']).ravel()),label='Mediation',clip=[0,10],
                shade=True,color=pal[1],ax=axs)
    sns.kdeplot(np.abs(np.array(samples_full['bias']).ravel()),label='Inflation',clip=[0,10],
            shade=True,color='grey',ax=axs)
    axs.legend()
    axs.set_xlim([0,max_effect])
    axs.set_xlabel('Effect size')
    
def plot_sims(data, y_var,axs, avg_samp,x_var="N",hue='alpha',pal="mako"):
    sns.lineplot(x=x_var, y=y_var,
             hue="alpha", 
             data=data,ax=axs,palette= sns.color_palette(pal,n_colors=5)) 

    axs.set_ylim(0,1)
    axs.plot([avg_samp,avg_samp],[0,1],ls='--')
    axs.set_xlim(np.min(data[x_var]),np.max(data[x_var]))
    axs.set_xlabel('Sample size')
    return axs

def plot_sim_figure(i_data, sim_data, df,
                    plot_medians=True, 
                    rep=.39, 
                    save_loc=False,
                    dpi=300,max_effect=2,pal="mako"):
    sns.set_context('paper',font_scale=1)
    sns.set_palette(sns.color_palette(pal))

    fig, axs = plt.subplots(3,2,figsize=(8,12))

    res = 100


    plot_fig4a(i_data['posterior'],axs[0][0])
    
    axs[0][0].set_xlim(0, max_effect)

    plot_sims(sim_data, "publication_rate",axs[0][1], avg_samp = df['n_o'].median(),pal="mako")
    axs[0][1].set_ylabel("P(Publish)")


    plot_sims(sim_data, "published_es",axs[1][0], avg_samp = df['n_o'].median(),pal="ch:start=2.8,rot=.3")
    axs[1][0].set_ylabel("Published effect size")
    
    if plot_medians:
        axs[1][0].scatter(df['n_o'].median(), df['d_o'].mean())
    
    axs[1][0].set_ylim(0, max_effect/2)


    plot_sims(sim_data, "replication_rate",axs[1][1], avg_samp = df['n_o'].median(),pal="rocket")
    axs[1][1].set_ylabel("P(replicate)")
    
    if plot_medians:
        axs[1][1].scatter(df['n_o'].median(), rep)



    plot_sims(sim_data, "type_s_error",axs[2][0], avg_samp = df['n_o'].median(),pal="ch:start=1.3,rot=-.1")
    axs[2][0].set_ylabel("Type-S error")


    plot_sims(sim_data, "reversals",axs[2][1], avg_samp = df['n_o'].median(),pal="Greys")
    axs[2][1].set_ylabel("P(reversal)")



    axs = axs.flat
    import string
    for n, ax in enumerate(axs):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=20, weight='bold')
    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc,dpi=dpi)
    return axs