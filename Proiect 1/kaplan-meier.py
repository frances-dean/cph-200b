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
    
