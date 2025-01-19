#######################################################################################
# Author: Franny Dean
# Script: cox_ph_model.py
# Function: write cox ph model from scratch too
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################################################################

class custom_cox_model:
    """
    Cox Proportional Hazards model from scratch.

    Parameters:
        df: DataFrame with time to event, event indicator, covariates
        event_col: column name for event indicator
        time_col: column name for time to event
        covariates: list of column names for covariates
    """
    def __init__(self, df, covariates, event_col='DEATH_EVENT', time_col='time'):
        self.df = df
        self.event_col = event_col
        self.time_col = time_col
        self.covariates = covariates
        self.betas = np.zeros(len(covariates))
        self.hazard_ratios = np.exp(self.betas)
        self.baseline_hazard = 1.0
    
    def fit(self, epochs=1000, lr=0.01):
        """
        Fit Cox PH model.
        """
        for epoch in range(epochs):
            # Compute partial log liklihood loss
            loss = self.partial_log_liklihood_loss()

            # Compute gradient


            # Update betas
            self.betas -= lr * gradient
            
            # Print every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        pass

    def partial_log_liklihood_loss(self):
        """
        Compute loss function.
        """
        loss = 0
        for i in range(len(self.df)):
            loss_top = np.exp(np.dot(self.df[self.covariates].iloc[i], self.betas))
            loss_bottom = sum([np.exp(np.dot(self.df[self.covariates].iloc[j], self.betas)) for j in range(len(self.df)) if self.df[self.time_col].iloc[j] >= self.df[self.time_col].iloc[i]])
            loss += np.log(loss_top / loss_bottom)
        return loss
    
    def get_hazard_ratios(self):
        """
        Get hazard ratios for each covariate.
        """
        self.cov_matrix = np.linalg.inv(np.dot(self.df[self.covariates].T, self.df[self.covariates]))
        
        return pd.DataFrame({'covariate': self.covariates, 'hazard_ratio': self.hazard_ratios,
                             'beta': self.betas, 'CI_low': self.betas - 1.96 * np.sqrt(np.diag(self.cov_matrix)),})


    def concordance_index(self):
        """
        Compute concordance index.
        """
        pass
#######################################################################################
