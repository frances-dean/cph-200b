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


class SurvivalModel(nn.Module):
    """
    Deep learning survival model class. Probably closest resembeles CoxTime.

    Replaces both baseline hazard and hazard ratio with a neural network.

    Parameters:
        df: DataFrame with time to event, event indicator, covariates
        event_col: column name for event indicator
        time_col: column name for time to event
        covariates: list of column names for covariates
    """
    def __init__(self, df, covariates, 
                hidden_dims = [128, 128],
                num_layers = [3, 3],
                #num_time_predictions = 20, # Try implementing with time as input instead
                use_bn = True,
                learning_rate = 0.001,
                censor_col='Censor (Censor = 1)', time_col='Survival Time'):
        
        print('Hazard model initializing...')

        # Data parameters
        self.df = df
        self.censor_col = censor_col
        self.time_col = time_col
        self.covariates = covariates
        self.input_dim_hazard = len(covariates)+1 # +1 for time

        print('Predicting survival probabilities at 20 time points')
        self.num_time_predictions = num_time_predictions

        # Initialize NN model (fully connected MLP) for hazard
        bias = not use_bn

        layers = []
        if num_layers[0] == 1:
            layers.append(nn.Linear(self.input_dim_hazard, 1)) # Try implementing as one output use time as input?
        elif num_layers[0] > 1:
            layers.append(nn.Linear(self.input_dim_hazard, hidden_dims[0], bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dims[0], hidden_dims[0], bias=bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dims[0]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[0],1))

        self.hazard_model = nn.Sequential(*layers)

        # Initialize NN model (fully connected MLP) for baseline hazard
        layers = []
        if num_layers[1] == 1:
            layers.append(nn.Linear(self.input_dim_hazard, 1))
        elif num_layers[1] > 1:
            layers.append(nn.Linear(self.input_dim_hazard, hidden_dims[1], bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dims[1]))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dims[1], hidden_dims[1], bias=bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dims[1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[1], 1))

        self.baseline_model = nn.Sequential(*layers)

        # Training parameters
        self.optimizer = optim.Adam(
            list(self.hazard_model.parameters()) + list(self.baseline_model.parameters()), 
            lr=learning_rate)
        print('DeepSurv model initialized.')

    def forward(self, X, t):
        """
        Forward pass gives the survival probabilities.
        """
        h_betas = self.hazard_model(X, t)
        baseline_hazard = self.baseline_model(X, t)
        return baseline_hazard * torch.exp(h_betas)
        
    
    def fit(self, X, t, epochs=1000):
        """
        Fit DeepSurv model.
        """
        print('Fitting DeepSurv model...')
        self.hazard_model.train()
        self.baseline_model.train()

        # Data
        X = torch.tensor(self.df[self.covariates].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.time_col].values, dtype=torch.float32)

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X, t)

            # Compute partial log liklihood loss
            loss = -self.log_likelihood_loss(predictions)

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
        self.hazard_model.eval()
        self.baseline_model.eval()
        return self.forward(X)

    def log_likelihood_loss(self, predictions):
        """
        Compute loss function as partial log liklihood loss. 

        Parameters:
            predictions is the output of the forward pass, i.e. the hazard at time T_i given X_i, beta

        """
        order = torch.argsort(self.df[self.time_col], descending=False)
        predictions = predictions[order]
        print(predictions)
        E = 1-self.df[self.censor_col][order]
        hazard_term = torch.sum(E * torch.log(predictions)) 
        print(torch.cumsum(predictions, dim=1))
        survival_term = torch.sum(torch.log(torch.cumsum(predictions, dim=1)))
        log_likelihood = hazard_term - survival_term
        return log_likelihood
    
    def concordance_index(self):
        """
        Compute concordance index.

        TODO - needs updated from Cox PH model.
        """
        time_values = self.df[self.time_col]
        events = self.df[self.censor_col]
        concordant = 0
        discordant = 0
        tied = 0

        X = torch.tensor(self.df[self.covariates].values, dtype=torch.float32)

        # First get gradients and Hessian
        self.model.eval()  # Ensure the model is in evaluation mode
    
        # Forward pass to get the log-hazard predictions
        predictions = self.model(X)

        # Loop through all time comparisons
        for i in range(len(time_values)):
            for j in range(i + 1, len(time_values)):  # Ensure i != j
                T_i = time_values.iloc[i]
                T_j = time_values.iloc[j]
                
                # If T_i < T_j and individual i survived shorter
                if (T_i < T_j ) & (events[i]==0):
                    if predictions[i] > predictions[j]: # Exp is monotonically increasing so beta * X is fine to compare
                        concordant+=1
                    else:
                        discordant+=1
                if (T_i > T_j ) & (events[j]==0):
                    if predictions[i] < predictions[j]: 
                        concordant+=1
                    else:
                        discordant+=1
                if (T_i == T_j ) & ((events[i]==0) & (events[j]==0)): 
                    tied+=1

        total_pairs = concordant + discordant + tied
        
        # Catch for no comparable
        if total_pairs == 0:
            return np.nan  # If no pairs were comparable

        c_index = concordant / total_pairs
        return c_index
#######################################################################################
