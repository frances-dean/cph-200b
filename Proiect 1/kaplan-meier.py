#######################################################################################
# Author: Franny Dean
# Script: kaplan-meier.py
# Function: scratch implementation of kaplan-meier survival estimation
#######################################################################################

import pandas as pd
import numpy as np

#######################################################################################

def fit_km(T, E, C):
  while t < T.max():
    

# Task 3: The Kaplan-Meier estimator generates population-level survival curves. Imagine modi-
# fying it to produce patient-specific survival curves instead. Propose a variant of the Kaplan-Meier estimator
# that incorporates a nearest-neighbor approach to estimate patient-level survival probabilities. Implement
# the proposed procedure and evaluate its performance using the Concordance Index (C-index)
