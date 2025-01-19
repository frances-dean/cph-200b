#######################################################################################
# Author: Franny Dean
# Script: exponential-survival.py
# Function: write a parametric exponential survival model and fit it
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaplan_meier import *

#######################################################################################

class exp_survival_model:
  """
  Parametric exponential survival model. Form is S(t) = exp( - rate * time) which is 
  equivalent to 1 - CDF of the exponential distribution.

  Parameters:
    df: DataFrame with time to event, event indicator, censoring indicator
    event_col: column name for event indicator
    time_col: column name for time to event
  """
  def __init__(self, df, event_col='DEATH_EVENT', time_col='time'):
    self.rate = (1 - df[event_col]).sum() / df[time_col].sum()
    self.df = df
    self.event_col = event_col
    self.time_col = time_col

  def get_estimate(self, t):
    """
    Get survival probability estimate at time t.
    """
    return np.exp(-self.rate * t)
  
  def plot_survival_curve(self, include_km=True):
    """
    Plot survival curve. 

    Parameters:
      include_km: boolean, whether to include Kaplan-Meier curve for comparison
    """

    self.df['survival'] = self.df[self.time_col].apply(self.get_estimate)
    if include_km:
      km_df = fit_km(self.df[self.time_col], self.df[self.event_col])
      plt.plot(km_df['T'], km_df['S'], c='green', label='Kaplan-Meier')
    plt.plot(self.df[self.time_col], self.df['survival'], label='Exponential Model')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Exponential Survival Curve')
    plt.legend()
    plt.show()

    
