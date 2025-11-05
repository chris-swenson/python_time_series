import matplotlib.pyplot as plt

# specific plots from other packages
from seaborn import histplot
from scipy.signal import periodogram
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# functions to support plots
from statsmodels.tsa.api import add_lag
from statsmodels.tsa.api import acf
from statsmodels.tsa.statespace.tools import diff
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.stattools import jarque_bera


# simple time series plot
def tsplot(x, xlab='Time', ylab='Value', figsize=(9, 4), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, **kwargs)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    plt.tight_layout()


# plot to inspect relationship with a specific lag
def lagplot(x, lag=1, figsize=(9,5), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    x_with_lag = add_lag(x, lags=lag)
    lw1 = lowess(x_with_lag[:, 0], x_with_lag[:, 1])
    plt.scatter(x_with_lag[:,0], x_with_lag[:,lag], **kwargs)
    ax.set_xlabel("Value")
    ax.set_ylabel("Lag " + str(lag))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.plot(lw1[:, 0], lw1[:, 1], c="k", linestyle='--')
    plt.tight_layout()


# slightly better acf / pacf plot than default
# combines acf and pacf plots as well with partial flag
def plot_acf2(dataframe, lags=20, partial=False, figsize=(9, 3), bartlett_confint=True, **kwargs):
    if partial:
        a1 = plot_pacf(dataframe, lags=lags, bartlett_confint=bartlett_confint)
    else:
        a1 = plot_acf(dataframe, lags=lags, bartlett_confint=bartlett_confint)
    a1.set(size_inches=figsize)
    plt.tight_layout()
    return a1


# plot the acf and pacf in one figure with two subplots
def plot_acf_pacf(dataframe, lags=20, figsize=(9, 6), bartlett_confint=True, **kwargs):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    fig = plot_acf(dataframe, lags=lags, markersize=3, ax=ax1, bartlett_confint=bartlett_confint, **kwargs)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(dataframe, method='ywm', lags=lags, markersize=3, ax=ax2, **kwargs)
    plt.tight_layout()
    return fig


# plot results of fast fourier transform
# optionally detrend or display in log terms
def plot_periodogram(
    dataframe, ylim=(0, 0), xlab='Frequency', ylab='Spectrum', 
    log=False, detrend=False, figsize=(9, 4)
):
    f, Pxx_den = periodogram(dataframe.squeeze(), detrend=detrend)
    fig = plt.figure(figsize=figsize)
    if log:
        plt.semilogy(f, Pxx_den)
    else:
        plt.plot(f, Pxx_den)
    if ylim != (0, 0):
        plt.ylim([1e-7, 1e2])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    return fig


# plot diagnostics for time series models
def plot_astsa(residuals, lags=20, model_df=None, title=None, figsize=(9, 6), markersize=3):

    assert model_df != None, "Specify model degrees of freedom (model_df=)."
    
    fig = plt.figure(figsize=figsize)

    # Time series plot
    ax1 = fig.add_subplot(311, title='Residuals')
    plt.plot(residuals)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    if title != None:
        plt.title(title, loc='right', fontdict={'fontsize':10})

    # ACF plot
    ax3 = fig.add_subplot(323)
    fig = plot_acf(residuals, lags=lags, markersize=markersize, ax=ax3)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    #ax4 = fig.add_subplot(324)
    #fig = plot_pacf(residuals, lags=lags, markersize=markersize, ax=ax4)

    # Q-Q plot
    ax4 = fig.add_subplot(324, title='Q-Q Plot')
    qqplot(residuals, line='s', markersize=markersize, ax=ax4)

    # Ljung-Box Statistic p values
    ax5 = fig.add_subplot(313, title='Ljung-Box P-Values')
    q_p_values = acorr_ljungbox(residuals, lags=lags, model_df=model_df)['lb_pvalue']
    plt.plot(q_p_values, linestyle='none', marker='o', markersize=markersize)
    plt.xticks(range((model_df+1), (lags+1)))
    plt.axhline(y=0.05, color='r', linestyle='--')
    ax5.set_ylim([-0.14, 1])
    ax5.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig


# extended time series diagnostic plots
def plot_ts_resid(residuals, lags=20, model_df=None, title=None, figsize=(9, 9), markersize=3):
    
    assert model_df != None, "Specify model degrees of freedom (model_df=)."
   
    fig = plt.figure(figsize=figsize)

    # Time series plot
    ax1 = fig.add_subplot(411, title='Residuals')
    plt.plot(residuals, markersize=markersize)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    if title != None:
        plt.title(title, loc='right', fontdict={'fontsize':10})
    
    # ACF plot
    ax3 = fig.add_subplot(423)
    fig = plot_acf(residuals, lags=lags, markersize=markersize, ax=ax3)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax5 = fig.add_subplot(425)
    fig = plot_pacf(residuals, method='ywm', lags=lags, markersize=markersize, ax=ax5)
    ax5.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Q-Q plot
    ax4 = fig.add_subplot(424, title='Q-Q Plot')
    qqplot(residuals, line='s', markersize=markersize, ax=ax4)

    # Histogram
    # Jarque-Bera test results: 'JB Statistic', 'P Value', 'Skew Estimate', 'Kurtosis Estimate'
    jb_pvalue = round(jarque_bera(residuals, axis=0)[1], 4)
    jb_pvalue = '<0.001' if jb_pvalue == 0 else '=' + str(jb_pvalue)
    jb_pvalue = 'Jarque-Bera p' + jb_pvalue
    ax6 = fig.add_subplot(426, title='Histogram (' + jb_pvalue + ')')
    histplot(residuals, stat='density', kde=True, ax=ax6)
    
    # Ljung-Box Statistic p values
    ax7 = fig.add_subplot(414, title='Ljung-Box P-Values')
    q_p_values = acorr_ljungbox(residuals, lags=lags, model_df=model_df)['lb_pvalue']
    plt.plot(q_p_values, linestyle='none', marker='o', markersize=markersize)
    plt.xticks(range((model_df+1), (lags+1)))
    plt.axhline(y=0.05, color='r', linestyle='--')
    ax7.set_ylim([-0.14, 1])
    ax7.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig

