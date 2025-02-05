#######################################################################################
# Author: Franny Dean
# Script: cox_ph_model.py
# Function: write cox ph model from scratch too
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

#######################################################################################

class custom_cox(torch.nn.Module):
    """
    Cox Proportional Hazards module from scratch.

    Ignores the baseline hazard, only stores ratio of hazards.

    Parameters:
        input_dim: int, number of covariates
    """
    def __init__(self,  input_dim):
        super(custom_cox, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        """ 
        Outputs beta^T * X. Exponentiate to get hazards ratio.
        """
        return self.linear(x)

class custom_cox_model:
    """
    Cox Proportional Hazards model from scratch.
    
    Parameters:
        df: DataFrame with time to event, event indicator, covariates
        event_col: column name for event indicator
        time_col: column name for time to event
        covariates: list of column names for covariates
    """
    def __init__(self, df, covariates, event_col='DEATH_EVENT', time_col='time',
                 learning_rate=0.001, num_epochs=200):
        
        # Data parameters
        self.df = df
        self.covariates = covariates
        self.event_col = event_col
        self.time_col = time_col

        # Initialize the model and optimizer
        self.model = custom_cox(len(self.covariates))
        print(f'Initialized with {len(self.covariates)} covariates')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        # After fitting to define/save
        self.betas = None

    
    def fit(self):
        """
        Fit Cox PH model.
        """
        self.model.train()  # Set model to training mode
        X = torch.tensor(self.df[self.covariates].values, dtype=torch.float32)
        
        for epoch in range(self.num_epochs):
        
            # Forward pass to get predictions
            predictions = self.model(X)
            
            # Compute the log-likelihood (negative for minimization)
            log_likelihood = -self.partial_log_likelihood_loss(predictions)
            
            # Zero the gradients before the backward pass
            self.optimizer.zero_grad()
            
            # Backward pass: Compute the gradient of the loss w.r.t. parameters
            log_likelihood.backward()
            
            # Update the model parameters
            self.optimizer.step()
            
            # Print the log-likelihood every n epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Negative Log-Likelihood: {log_likelihood.item()}")
        
        print(f'Fitting done. Final loss: {log_likelihood.item()}')
        self.betas = self.model.linear.weight.detach().numpy().flatten()  # Use the weights (betas)

    def partial_log_likelihood_loss(self, predictions):
        """
        Compute loss function as partial log liklihood loss.
        """
        
        time_tensor = torch.tensor(self.df[self.time_col].values, dtype=torch.float32)
    
        _, sorted_indices = torch.sort(time_tensor, descending=True)
        sorted_predictions = torch.exp(predictions[sorted_indices])  # Exponentiate to get the hazard ratio
    
        # Prevent dividing by zero:
        sorted_predictions = torch.maximum(sorted_predictions, torch.tensor(1e-8, dtype=torch.float32))
        
        sorted_event_observed = self.df[self.event_col].iloc[sorted_indices].values 

        log_likelihood = 0
        for i in range(len(sorted_predictions)):
            
            # Calculate the log-likelihood contribution for each observation
            # Assuming sorted_event_observed[i] is 1 for events and 0 for censored
            if sorted_event_observed[i] == 1:
                log_likelihood += torch.log(sorted_predictions[i] / torch.sum(sorted_predictions[:i+1]))
        return log_likelihood

    
    def variance_covariance_matrix(self):
        """
        Calculate the variance-covariance matrix of the model's coefficients.
        This can be approximated from the inverse of the Hessian matrix.
        """
        X = torch.tensor(self.df[self.covariates].values, dtype=torch.float32)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Forward pass to get the log-hazard predictions
        predictions = self.model(X)
        
        # Compute the log-likelihood
        log_likelihood = -self.partial_log_likelihood_loss(predictions)
        
        # Compute gradients of the log-likelihood with respect to the parameters
        self.model.zero_grad()
        gradients = torch.autograd.grad(log_likelihood, self.model.parameters(), create_graph=True)
        
        print(self.model.parameters())
        # Initialize Hessian matrix
        hessian = torch.zeros(sum(p.numel() for p in self.model.parameters()), sum(p.numel() for p in self.model.parameters()))
        print(hessian.shape)

        # Loop over the gradients to compute second derivatives (Hessian)
        idx = 0
        for param_i, grad_i in zip(self.model.parameters(), gradients):
            grad_i = grad_i.reshape(-1)  # Flatten the gradient
            
            for j, param_j in enumerate(self.model.parameters()):
                # Compute the second derivative (Hessian) with respect to params i and j
                hessian_ij = torch.autograd.grad(grad_i.sum(), param_j, retain_graph=True)[0].reshape(-1)
                hessian[idx:idx + grad_i.numel(), j:j + hessian_ij.numel()] = hessian_ij
                
            idx += grad_i.numel()

        # Add a small regularization term to the diagonal to help with inversion
        regularization_term = torch.eye(hessian.size(0)) * 0.1 # Gives a highly singular matrix for some reason, even this gives large CI
        hessian += regularization_term

        # Compute the variance-covariance matrix (inverse of Hessian)
        cov_matrix = torch.inverse(hessian).detach().numpy()

        return cov_matrix

    
    def get_hazard_ratios(self):
        """
        Get hazard ratios for each covariate.
        """
        self.cov_matrix = self.variance_covariance_matrix()
        #print(np.diag(self.cov_matrix))

        #condition_number = np.linalg.cond(self.cov_matrix)
        #print(f"Condition number: {condition_number}")

        return pd.DataFrame({'covariate': self.covariates, 
                             'hazard_ratio': np.exp(self.betas),
                             'beta': self.betas}) 
                             
                             #'CI_low': self.betas - 1.96 * np.sqrt(np.diag(self.cov_matrix)),
                             #'CI_high': self.betas + 1.96 * np.sqrt(np.diag(self.cov_matrix))})


    def concordance_index(self):
        """
        Compute concordance index.
        """
        time_values = self.df[self.time_col]
        events = self.df[self.event_col]
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
                if (T_i < T_j ) & (events[i]==1):
                    if predictions[i] > predictions[j]: # Exp is monotonically increasing so beta * X is fine to compare
                        concordant+=1
                    else:
                        discordant+=1
                if (T_i > T_j ) & (events[j]==1):
                    if predictions[i] < predictions[j]: 
                        concordant+=1
                    else:
                        discordant+=1
                if (T_i == T_j ) & ((events[i]==1) & (events[j]==1)): 
                    tied+=1

        total_pairs = concordant + discordant + tied
        
        # Catch for no comparable
        if total_pairs == 0:
            return np.nan  # If no pairs were comparable

        c_index = concordant / total_pairs
        return c_index
#######################################################################################
