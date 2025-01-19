#######################################################################################
# Author: Franny Dean
# Script: deep_survival.py
# Function: implementation of proposed deep survival model class
#######################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

#######################################################################################

class DeepSurvival(nn.Module):
    """
    Deep survival model class.
    This class implements a deep learning model for survival analysis modeled after DeepSurv.

    Parameters:
        df: DataFrame with time to event, event indicator, covariates
        event_col: column name for event indicator
        time_col: column name for time to event
        covariates: list of column names for covariates
    """
    def __init__(self, df, covariates, 
                hidden_dim = 128,
                num_layers = 3,
                num_time_predictions = 20,
                use_bn: bool = True,
                censor_col='Censor (Censor = 1)', time_col='Survival Time'):
        
        print('DeepSurv model initializing...')

        # Data parameters
        self.df = df
        self.censor_col = censor_col
        self.time_col = time_col
        self.covariates = covariates
        self.input_dim = len(covariates)
        
        print('Assume baseline hazard is constant at 1.0')
        # TODO - implement baseline hazard as time dependent
        self.baseline_hazard = 1.0

        print('Predicting survival probabilities at 20 time points')
        self.num_time_predictions = num_time_predictions

        # Initialize NN model (fully connected MLP)
        bias = not use_bn

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, self.num_time_predictions))
        elif num_layers > 1:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, self.num_time_predictions))

        self.model = nn.Sequential(*layers)

        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print('DeepSurv model initialized.')

    def forward(self, X):
        """
        Forward pass through DeepSurv model. This gives h_beta(X).
        """
        return self.model(X)
        
    
    def fit(self, X, epochs=1000, lr=0.01):
        """
        Fit DeepSurv model.
        """
        print('Fitting DeepSurv model...')
        for epoch in range(epochs):
            # Forward pass
            h_betas = self.forward(X)

            # Compute partial log liklihood loss
            loss = self.log_liklihood_loss(h_betas)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print('Fitting done.')

    def predict(self, X):
        """
        Predict survival probability from a covariate set.
        """
        h_betas = self.forward(X)
        return self.baseline_hazard * np.exp(h_betas)

    def log_liklihood_loss(self, h_betas):
        """
        Compute loss function.
        """
        loss = 0
        for i in range(len(self.df)):
            loss += ??
        return loss