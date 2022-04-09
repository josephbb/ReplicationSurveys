import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm
import arviz as az
import matplotlib as mpl
sns.set_style('white')


@np.vectorize
def replicate_f_vec(rho, tc):
    def temp(x): return stats.norm.pdf(x) * \
        stats.norm.cdf((rho * x - tc) / np.sqrt(1 - rho**2))
    return sp.integrate.quad(temp, a=tc, b=np.inf)


@np.vectorize
def replicate_vec(tau=.1, sigma=0, epsilon=1, n=30, alpha=0.05):
    tc = sp.stats.norm.ppf(1 - alpha / 2)
    tc = (epsilon / np.sqrt(n)) * tc / \
        np.sqrt((epsilon**2) / n + tau**2 + sigma**2)
    rho = tau**2 / (epsilon**2 / n + tau**2 + sigma**2)
    return(replicate_f_vec(rho, tc) / stats.norm.cdf(-tc))[0]


@np.vectorize
def publish_vec(tau=.1, sigma=0, epsilon=1, n=30, alpha=0.05):
    tc = sp.stats.norm.ppf(1 - alpha / 2)
    tc = (epsilon / np.sqrt(n)) * tc / \
        np.sqrt((epsilon**2) / n + tau**2 + sigma**2)
    return(2 * sp.stats.norm.cdf(-tc))


@np.vectorize
def type_m_vec(tau=.4, sigma=.2, epsilon=1, n=100, alpha=0.05):
    tc = sp.stats.norm.ppf(1 - alpha / 2)
    tc = (epsilon / np.sqrt(n)) * tc

    # lower bound adjusted as per
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    def temp(x): return (truncnorm((tc - x) / np.sqrt((epsilon**2) / n + sigma**2),
                                   np.inf,
                                   x,
                                   scale=np.sqrt((epsilon**2) / n + sigma**2)).stats()[0]) * \
        truncnorm(0, np.inf, scale=tau).pdf(x)
    typem = sp.integrate.quad(temp, a=0, b=np.inf)[
        0] / (tau * np.sqrt(2 / np.pi))

    return typem


@np.vectorize
def type_s_vec(tau=.4, sigma=.2, epsilon=1, n=100, alpha=0.05):
    tc = sp.stats.norm.ppf(1 - alpha / 2)
    tc = (epsilon / np.sqrt(n)) * tc

    def func(x): return stats.norm(loc=x,
                                   scale=np.sqrt((epsilon**2) / n + sigma**2)).cdf(-tc) * \
        stats.norm(0, scale=tau).pdf(x)

    t_lessthan_tc = stats.norm(0, scale=np.sqrt(
        (epsilon**2) / n + sigma**2 + tau**2)).cdf(-tc)
    d_greater_0 = stats.norm(0, scale=np.sqrt(
        (epsilon**2) / n + sigma**2 + tau**2)).cdf(0)
    integ = 2 * (sp.integrate.quad(func, a=0, b=np.inf)[0])
    return d_greater_0 * integ / t_lessthan_tc
