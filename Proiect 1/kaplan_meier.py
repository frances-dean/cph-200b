#######################################################################################
# Author: Franny Dean
# Script: kaplan_meier.py
# Function: scratch implementation of kaplan-meier survival estimation
#######################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines.utils import concordance_index

#######################################################################################

def fit_km(T, E):
  """
  Fit Kaplan-Meier suvival curve estimator.

  Parameters:
    T: array-like, time to event
    E: array-like, event indicator, assumes 0 means censored, 1 means event
  """
  # Sort data by time to event, add survival column
  df = pd.DataFrame({'T': T, 'E': E}).sort_values(by='T')
  df['S'] = np.nan
  S = 1.0

  for t in df['T'].unique():
    #print(f'Time: {t}')

    # Compute denominator: number of people at risk
    d = df[(df['T'] >= t)].shape[0]

    # Compute numerator: number of individuals with event at t
    n = df[(df['T'] == t) & (df['E'] == 1)].shape[0]

    # Compute survival probability
    S *= (1.0 - n/d)
    #print(f'Survival at: {S}')

    df.loc[df['T'] == t, 'S'] = S

  return df

def plot_km(df, time_col='T'):
  """
  Plot Kaplan-Meier survival curve.

  Parameters:
    df: DataFrame with survival, after T, E, C passed to fit_km
  """
  plt.plot(df[time_col], df['S'])
  plt.xlabel('Time')
  plt.ylabel('Survival Probability')  
  plt.title('Kaplan-Meier Survival Curve')
  plt.show()

#######################################################################################

def fit_k_nn_km(k, df, idx, covariates,time_col='time',event_col='DEATH_EVENT'):
  """
  Fit Kaplan-Meier suvival curve estimator with k-nearest neighbors.

  Parameters:
    k: int, number of nearest neighbors to build curve from
    df: DataFrame with T, E, C and covariates, ideally normalized
    idx: which individual to calculate curve for
    covariates: array-like, list of covariates which are df col names
  """
  # Find the k nearest neighbors for input individual
  df['dist'] = df[covariates].apply(lambda row: np.sqrt(np.sum((row - df.loc[idx, covariates])**2)), axis=1)
  df_k_nn = df.sort_values(by='dist').iloc[1:].head(k)
  #print(df_k_nn)

  return fit_km(df_k_nn[time_col], df_k_nn[event_col])

def normalize_features(df, covariates):
  """
  Normalize features in DataFrame.
  """
  for covariate in covariates:
    df[covariate] = (df[covariate] - df[covariate].mean()) / df[covariate].std()

  return df
