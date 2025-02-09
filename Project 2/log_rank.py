#######################################################################################
# Author: Franny Dean
# Script: log_rank.py
# Function: scratch implementation of log rank test for comparing survival curves
#######################################################################################

import numpy as np
import pandas as pd

from scipy.stats import norm


#######################################################################################

## LOG RANK TEST: ##

# notation:
# groups - k
# n = total patients in risk group
# d = patients with the event
# time - t

## Test statistic is the Fisher test statistic under null 
# d_{t,1} \sim  Hypergeo(n_t, n_{t,k=1}, d_t) ...  where
# Hypergeo(full finite population, number draws with replacement, "successes")
# for EACH time t
# mean: n_{t,1}/n_t * d_t
# variance: n_{t,0}*n_{t,1} * d_t (n_t - d_t)/ n_t^2 (n_t - 1)
# to calculate over all times, use Mantel-Haenszel statistic:
# Z = \sum_{times t} (d_{t,1} - E_t) / \sqrt(\sum_{times} V_t) which is normal with 0,1 under null


#######################################################################################



def log_rank_test(df, 
                  group_variable, 
                  duration_col='Survival Time', 
                  censor_col='Censor (Censor = 1)'):
    """
    Implementation of log rank test to compare two groups survival curve.

    Parameters:
        df = dataframe pandas
        group_variable = treatment column name, 0, 1
        duration_col = time column
        censor_col = 0, 1 for censored

    """
    # Control, treatment groups
    df = df[[duration_col, censor_col, group_variable]].sort_values(by=duration_col)

   
    # Define helpers
    def hypergeo_mean(N, n, k):
        return n/N * k

    def hypergeo_var(N, n, k):
        return ((N - n) * n * k * (N - k)) / (N**2 * (N - 1))
    
    top, bottom = 0, 0
    for t in df[duration_col].unique(): 
        data = df[df[duration_col]>=t] # Subset to those remaining in risk set
        group0, group1 = data[data[group_variable]==0], data[data[group_variable]==1]

        # group d_{t,1} whose event happened at time t in treatment group
        d = len(group1[(group1[duration_col]==t) & (group1[censor_col]==0)])
        # d_t group whose event happened at time t in population
        k = len(data[data[censor_col]==0])
        # total population
        N = len(group0) + len(group1)
        # at risk population in treatment gorup
        n = len(group1)
        if N<2:
            continue
        top += d - hypergeo_mean(N, n, k)
        bottom += hypergeo_var(N, n, k)

    # Mantel-Haenszel is normally distributed
    Z = top/np.sqrt(bottom)
    # Two sided test
    p_value = 2 * norm.sf(abs(Z))

    return Z, p_value 



def weighted_log_rank_test(df, 
                           group_variable, 
                           duration_col='Survival Time', 
                           censor_col='Censor (Censor = 1)',
                           propensity_col='propensity'):
    """
    Implementation of log rank test to compare two groups survival curve with propensity weighting.

    Parameters:
        df = dataframe pandas
        group_variable = treatment column name, 0, 1
        duration_col = time column
        censor_col = 0, 1 for censored
        propensity_col = propensity score column
    """
    # Control, treatment groups
    df = df[[duration_col, censor_col, group_variable,propensity_col]].sort_values(by=duration_col)
    df['weight'] = np.where(df[group_variable] == 1, 1/df[propensity_col], 1/(1-df[propensity_col]))
   
    # Define helpers
    def hypergeo_mean(N, n, k):
        return n/N * k

    def hypergeo_var(N, n, k):
        return ((N - n) * n * k * (N - k)) / (N**2 * (N - 1))
    
    top, bottom = 0, 0
    for t in df[duration_col].unique(): 
        data = df[df[duration_col]>=t] # Subset to those remaining in risk set
        _, group1 = data[data[group_variable]==0], data[data[group_variable]==1]

        # Calculate all sums weighted by propensity score weights?
        # group d_{t,1} whose event happened at time t in treatment group
        d = group1.loc[(group1[duration_col] == t) & (group1[censor_col] == 0), 'weight'].sum()

        # d_t group whose event happened at time t in population
        k = data.loc[data[censor_col] == 0, 'weight'].sum()

        # total population
        N = data['weight'].sum()
        
        # at risk population in treatment gorup
        n = group1['weight'].sum()
        
        if N<2:
            continue
        top += d - hypergeo_mean(N, n, k)
        bottom += hypergeo_var(N, n, k)

    # Mantel-Haenszel is normally distributed
    Z = top/np.sqrt(bottom)
    # Two sided test
    p_value = 2 * norm.sf(abs(Z))

    return Z, p_value 
