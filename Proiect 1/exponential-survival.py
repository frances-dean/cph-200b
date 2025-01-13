#######################################################################################
# Author: Franny Dean
# Script: exponential-survival.py
# Function: write a parametric exponential survival model and fit it
#######################################################################################

# Task: Instead of the nonparametric Kaplan-Meier estimator, one can estimate a survival curve
# using a parametric model that makes assumptions about the distribution of survival times. Assume that
# the survival time in this population follows an exponential distribution. Propose an algorithm for fitting the
# parameters of this parametric model. Compare the results of the fitted exponential distribution with the
# Kaplan-Meier estimate and comment on the limitations of the parametric model

#######################################################################################

class exp_survival_model(rate, time)
  """
  y = a(1 - r)^t, a = 1, r = rate of decay with continuous time t in survival
  """
  
  return (1-rate)^time

  def fit(self)
  
