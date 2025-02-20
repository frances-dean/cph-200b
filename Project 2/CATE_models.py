#######################################################################################
# Author: Franny Dean
# Script: CATE_models.py
# Function: pytorch implementation of TARNet, CFRMMD from Shalit et al. 2017
#           then implement Dragonnet and Targeted regularization from Shi et al. 2019
#######################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

# for visualizing the representation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#######################################################################################

## TARNet or CFRMMD depending on alpha ##
class TARNet():
    def __init__(self, df, covariate_cols=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',
                                           'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20',
                                           'X21','X22','X23','X24','X25'],
                                           treatment_col='T', label_col='Y', alpha=0, lambda_=0):
        # Data parameters
        self.df = df
        self.covariate_cols = covariate_cols
        self.treatment_col = treatment_col
        self.label_col = label_col
        
        # Loss hyperparameters
        self.lambda_ = lambda_
        self.alpha = alpha
        
        self.u = sum(self.df[self.treatment_col].values)/len(self.df[self.treatment_col].values)
        self.input_dim = len(covariate_cols)
        self.output_dim = 1 # for each arm we predict a single value, Y_hat

        # Layers:
        # "CFR is implemented as a feed-forward
        # neural network with 3 fully-connected exponential-linear
        # layers for the representation and 3 for the hypothesis. Layer
        # sizes were 200 for all layers used for Jobs and 200 and 100
        # for the representation and hypothesis used for IHDP.""
        
        self.representation_layers = nn.Sequential(
            nn.Linear(self.input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100)
        )
        self.control_arm = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, self.output_dim)
        )
        self.treatment_arm = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, self.output_dim)
        )

        # For continuous data we use mean squared loss and for binary data, we use log-loss.
        self.prediction_loss = nn.MSELoss(reduction='none')

        # Ignore regularization for now
        # self.regularizing_term = self.lambda_ * torch.norm(self.weights, p=2) 
        
        # loss = self.hypothesis_loss + self.regularizing_term + self.IPM_penalty_loss

    def forward(self, X):

        representation = self.representation_layers(X)
        y0_pred = self.control_arm(representation)
        y1_pred = self.treatment_arm(representation)
        return representation, y0_pred, y1_pred
        
    def fit(self, epochs=100, lr=0.001):

        # Get data
        X = torch.tensor(self.df[self.covariate_cols].values, dtype=torch.float32)
        y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)

        # The model is trained using Adam (Kingma & Ba, 2014). 
        optimizer_representation = optim.Adam(self.representation_layers.parameters(), lr=lr)
        optimizer_control = optim.Adam(self.control_arm.parameters(), lr=lr)
        optimizer_treatment = optim.Adam(self.treatment_arm.parameters(), lr=lr)
    
        
        for epoch in range(epochs):
            optimizer_representation.zero_grad()
            optimizer_control.zero_grad()
            optimizer_treatment.zero_grad()

            representation, y0_pred, y1_pred = self.forward(X)

            # Calculate weights
            weights = t/ 2*self.u + (1 - t) / 2*(1 - self.u)

            # Calculate loss
            control_conditon = (t == 0)
            treatment_condition = (t == 1)
            loss_y0 = (weights * self.prediction_loss(y0_pred[control_conditon], y[control_conditon])).mean() # +  add other loss later
            loss_y1 = (weights * self.prediction_loss(y1_pred[treatment_condition], y[treatment_condition])).mean() # +  add other loss later

            representation_control = representation[control_conditon]
            representation_treatment = representation[treatment_condition]

            loss_pred = loss_y0 + loss_y1  # Prediction loss

            # Squared Linear MMD = IPM loss example -- idk what they even use in the paper
            loss_ipm = self.alpha * torch.norm(representation_control.mean() - representation_treatment.mean(), p=2) ** 2 

            # Backpropagate both prediction loss (through all layers) and IPM loss (through representation layers)
            loss_pred.backward()  # Prediction loss affects all layers
            optimizer_representation.step()  # Update representation layers (affects both prediction and MMD)

            loss_ipm.backward()  # IPM loss only affects representation layers
            optimizer_control.step()  # Update control prediction head
            optimizer_treatment.step()  # Update treatment prediction head

            # Print loss
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss IPM: {loss_ipm.item()}, Prediction Loss: {loss_pred.item()}')
            
    def predict(self, X):
        return self.model(X)
    
    def plot_representation_space(self, n_components=2):
        # Get data
        X = torch.tensor(self.df[self.covariate_cols].values, dtype=torch.float32)
        y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
        
        # Get representation
        representation, _, _ = self.forward(X)

        # Index on what is control versus treatment
        representation_control = representation[t == 0]
        representation_treatment = representation[t == 1]

        # Run separate t-SNE on control and treatment representations
        tsne_control = TSNE(n_components=n_components, random_state=42)
        X_tsne_control = tsne_control.fit_transform(representation_control.detach().numpy())
        
        tsne_treatment = TSNE(n_components=n_components, random_state=42)
        X_tsne_treatment = tsne_treatment.fit_transform(representation_treatment.detach().numpy())

        # Plot t-SNE
        plt.scatter(X_tsne_control[:, 0], X_tsne_control[:, 1], color='blue', label='Control')
        plt.scatter(X_tsne_treatment[:, 0], X_tsne_treatment[:, 1], color='red', label='Treatment')
        plt.title("t-SNE Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()