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

from pycox.models.loss import CoxPHLoss

#######################################################################################


class SurvivalModel(nn.Module):
    """
    Deep learning survival model class. Closest resembeles CoxTime.

    Replaces both baseline hazard and hazard ratio with a neural network.

    Parameters:
        df: Train DataFrame with time to event, event indicator, covariates.
        event_col: column name for event indicator. Currently its a censoring indicator.
        time_col: column name for time to event.
        covariates: list of column names for covariates to use.
    """
    def __init__(self, df, covariates, 
                hidden_dims = [128, 128, 64],
                num_layers = [5, 5, 5],
                use_bn = True,
                learning_rate = 0.001,
                censor_col='Censor (Censor = 1)', time_col='Survival Time',
                domain_adapatation_model = None):
        super(SurvivalModel, self).__init__()
        print('Hazard model initializing...')

        # Data parameters
        self.df = df
        self.censor_col = censor_col
        self.time_col = time_col
        self.max_time = df[time_col].max()
        self.df[self.time_col] /= self.max_time  # Normalize time to [0, 1]...
        self.covariates = covariates
        self.input_dim_hazard = len(covariates)+1 # +1 for time

        # Initialize NN model (fully connected MLP) for HAZARD RATIO
        bias = not use_bn

        ratio_layers = []
        if num_layers[0] == 1:
            ratio_layers.append(nn.Linear(self.input_dim_hazard, 1)) # Try implementing as one output use time as input?
        elif num_layers[0] > 1:
            ratio_layers.append(nn.Linear(self.input_dim_hazard, hidden_dims[0], bias=bias))
            if use_bn:
                ratio_layers.append(nn.BatchNorm1d(hidden_dims[0]))
            ratio_layers.append(nn.ReLU())
            for _ in range(num_layers[0] - 2):
                ratio_layers.append(nn.Linear(hidden_dims[0], hidden_dims[0], bias=bias))
                if use_bn:
                    ratio_layers.append(nn.BatchNorm1d(hidden_dims[0]))
                ratio_layers.append(nn.ReLU())
            ratio_layers.append(nn.Linear(hidden_dims[0],1))
            
            # # Need hazards to be positive
            # hazard_layers.append(nn.Softplus())

        self.hazard_model = nn.Sequential(*ratio_layers)

        # Initialize NN model (fully connected MLP) for BASELINE hazard
        baseline_layers = []
        if num_layers[1] == 1:
            baseline_layers.append(nn.Linear(self.input_dim_hazard, 1))
        elif num_layers[1] > 1:
            baseline_layers.append(nn.Linear(self.input_dim_hazard, hidden_dims[1], bias=bias))
            if use_bn:
                baseline_layers.append(nn.BatchNorm1d(hidden_dims[1]))
            baseline_layers.append(nn.ReLU())
            for _ in range(num_layers[1] - 2):
                baseline_layers.append(nn.Linear(hidden_dims[1], hidden_dims[1], bias=bias))
                if use_bn:
                    baseline_layers.append(nn.BatchNorm1d(hidden_dims[1]))
                baseline_layers.append(nn.ReLU())
            baseline_layers.append(nn.Linear(hidden_dims[1], 1))
            
            # Need baseline hazards to be positive too
            baseline_layers.append(nn.Softplus())

        self.baseline_model = nn.Sequential(*baseline_layers)

        # Training parameters
        self.optimizer = optim.Adam(
            list(self.hazard_model.parameters()) + list(self.baseline_model.parameters()), 
            lr=learning_rate)
        print('SurvivalModel initialized.')

        # Domain adaptation model - None if not used
        self.domain_adapatation_model = domain_adapatation_model

    def forward(self, combined_tensor):
        """
        Forward pass gives the hazard function at time t given X.

        combined_tensor: tensor of shape (N, |X| + 1), where N is the number of data points and 
                        |X| is the number of covariates plus 1 for the time variable.
        """
        h_betas = self.hazard_model(combined_tensor)
        baseline_hazard = self.baseline_model(combined_tensor)
        return baseline_hazard * torch.exp(h_betas)
        
    
    def fit(self, epochs=50, batch_size=32):
        """
        Fit survival model. Implements batch training in fit... for "simplicity."

        Parameters:
            epochs: number of training epochs.
            batch_size: size of training batches.
        """
        print('Fitting DeepSurv model...')
        self.hazard_model.train()
        self.baseline_model.train()

        # All training data
        X = torch.tensor(self.df[self.covariates].values, dtype=torch.float32) # shape (N, |X|)
        t = torch.tensor(self.df[self.time_col].values, dtype=torch.float32) # shape (N,)
        combined_tensor = torch.cat((X, t.unsqueeze(1)), dim=1) # shape (N, |X| + 1)

        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(X))
            combined_tensor = combined_tensor[indices]

            # Batch data
            for i in range(0, len(X), batch_size):
                combined_tensor_batch = combined_tensor[i:i+batch_size] # shape (batch_size, |X| + 1)
                
                # Forward pass
                predictions = self.forward(combined_tensor_batch)
                predictions = torch.clamp(predictions, min=1e-6)  # Prevent log(0) error

                # Compute cox proportional hazard loss
                events_observed = 1 - torch.tensor(self.df[self.censor_col].iloc[indices][i:i+batch_size].values, dtype=torch.float)
                durations = torch.tensor(self.df[self.time_col].iloc[indices][i:i+batch_size].values, dtype=torch.float)
                loss = -self.log_likelihood_loss(events_observed, durations, predictions) #+ self.concordance_index()
                #loss = CoxPHLoss()(predictions, durations, events_observed)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # Print every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        print('Fitting done.')

    def predict(self, X, times=None):
        """
        Predict survival probability from a covariate set and input times.

        Parameters:
            X: NEW covariate tensor of shape (N, |X|), where N is the number of data points
                and |X| is the number of covariates.
            times: time tensor list of the times to predict the survival curve for,
                defaults to 20 evenly spaced points in the range of training data.

        Returns:
            Predicted survival probabilities for each data point and time point.
            Shape: (N, T), where T is the number of time points.
        """
        if times is None:
            times = torch.linspace(0, 1, steps=20) # Times are scaled to be between 0 and 1.

        print('Predicting survival probabilities...')

        # Ensure `times` has shape (T, 1), where T is the number of time points
        times_tensor = times.unsqueeze(1)  # Shape (T, 1)

        # Repeat `times_tensor` for each data point in X
        times_tensor = times_tensor.repeat(1, X.shape[0]).T  # Shape (N, T)

        # Combine covariates and repeated times along the feature dimension
        # Expand times along the last dimension to match the covariate dimension
        combined_tensor = torch.cat((X.unsqueeze(1).repeat(1, times_tensor.shape[1], 1), times_tensor.unsqueeze(2)), dim=2)

        # Flatten combined tensor for model input: (N * T, |X| + 1)
        combined_tensor = combined_tensor.view(-1, combined_tensor.shape[2])

        # Set models to evaluation mode
        self.hazard_model.eval()
        self.baseline_model.eval()

        # Forward pass through the model
        with torch.no_grad():
            predictions = self.forward(combined_tensor)  # Model expects input of shape (N * T, |X| + 1)

        # Reshape predictions to (N, T)
        predictions = predictions.view(X.shape[0], times_tensor.shape[1])

        return predictions

    
    def compute_survival_from_hazard(self, hazards):
        """
        Compute the survival function from predicted hazards. 

        Parameters:
            hazards: Tensor of predicted hazards of shape (N, T),
                    where N is the number of data points and T is the number of time points.
        """
        cumulative_hazard = torch.cumsum(torch.clamp(hazards, min=1e-6), dim=1)  # Shape: (N, T)
        survival = torch.exp(-cumulative_hazard)  # Shape: (N, T)

        return survival
    
    def plot_average_survival(self, test_df=None):
        """
        Plot average survival curve.

        Parameters:
            test_df: DataFrame to predict survival curve for if not training data.
        """
        if test_df is not None:
            df = test_df
        else:
            df = self.df

        X = torch.tensor(df[self.covariates].values, dtype=torch.float32)
        times = np.linspace(0, self.max_time, num=20)
        predictions = self.compute_survival_from_hazard(self.predict(X)).detach().numpy() # Shape (N, 20) for default 20 time points
        plt.figure(figsize=(10, 6))
        # Plot average survival curve (compute mean across all data points)
        print(np.mean(predictions, axis=0))
        plt.plot(times, np.mean(predictions, axis=0))
        plt.title('Average Survival Curve')
        plt.xlabel('Time') # Converted back to real scale not 0 to 1
        plt.ylabel('Survival Probability')
        plt.show()

    def log_likelihood_loss(self, events_observed, times_tensor, predictions, X=None):
        """
        Compute loss function as log liklihood loss. Ref slide 29 of Lecture 3.

        Parameters:
            events: Tensor of event indicators of shape (N, 1), implementent as censoring indicator.
            times: Tensor of observed times of shape (N, 1)
            predictions: the output of the forward pass, i.e. the hazard at time T_i given X_i, beta
            X: needed only if domain adaptation model is used.
        """
        order = torch.argsort(times_tensor, descending=False)#.numpy() # Sort times in ascending order
        E = 1 - events_observed[order] # because the model has events as censored
        E = E.unsqueeze(1) # Shape (N, 1)
        predictions = predictions[order] # shape (N, 1)

        # Weights for domain adaptation
        weights = torch.ones_like(E) # Default to 1
        if self.domain_adapatation_model is not None:
            # Domain adaptation model predicts probability X is is from uncensored data
            g = self.domain_adapatation_model(X) # X is shape (N, |X|) so g is shape (N, 1)

            # We implement weight which adds higher weight to uncensored data, per slide 32 of Lecture 4
            weights = torch.tensor((1-g)/g, dtype=torch.float32) # Shape (N, 1)
        
        survival_predictions = self.compute_survival_from_hazard(predictions)  # Cumulative sum of predicted hazards
        #print(survival_predictions)
        
        hazard_term = torch.sum(weights*E * torch.log(torch.clamp(predictions, min=1e-6))) # Prevent log(0)
        survival_term = torch.sum(weights*E * torch.log(survival_predictions + 1e-6)) # Prevent log(0)

        log_likelihood = hazard_term + survival_term
        return log_likelihood
    
    def concordance_index(self, test_df=None):
        """
        Compute concordance index.
        """
        if test_df is not None:
            df = test_df
        else:
            df = self.df
        
        time_values = df[self.time_col]
        events = df[self.censor_col] # censoring indicator
        concordant = 0
        discordant = 0
        tied = 0

        X = torch.tensor(df[self.covariates].values, dtype=torch.float32)
        t = torch.tensor(df[self.time_col].values, dtype=torch.float32)
        combined_tensor = torch.cat((X, t.unsqueeze(1)), dim=1)

        # First get gradients and Hessian
        self.hazard_model.eval()
        self.baseline_model.eval()  # Ensure the models are in evaluation mode
    
        # Forward pass to get the log-hazard predictions
        with torch.no_grad():
            predictions = self.forward(combined_tensor)
            #print(predictions)
            #print(predictions.shape)
            predictions = predictions.detach().numpy().flatten()
        

        # Loop through all time comparisons
        for i in range(len(time_values)):
            for j in range(i + 1, len(time_values)):  # Ensure i != j
                T_i = time_values.iloc[i]
                T_j = time_values.iloc[j]
                
                # If T_i < T_j and individual i survived shorter
                if (T_i < T_j ) & (events.iloc[i]==0):
                    if predictions[i] > predictions[j]: # hazard of X, t
                        concordant+=1
                    else:
                        discordant+=1
                if (T_i > T_j ) & (events.iloc[j]==0):
                    if predictions[i] < predictions[j]: 
                        concordant+=1
                    else:
                        discordant+=1
                if (T_i == T_j ) & ((events.iloc[i]==0) & (events.iloc[j]==0)): 
                    tied+=1

        total_pairs = concordant + discordant + tied
        
        # Catch for no comparable
        if total_pairs == 0:
            return np.nan  # If no pairs were comparable

        c_index = concordant / total_pairs
        return c_index
#######################################################################################
