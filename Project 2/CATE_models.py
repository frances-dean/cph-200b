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
class TARNet(nn.Module):
    def __init__(self, df, covariate_cols=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',
                                           'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20',
                                           'X21','X22','X23','X24','X25'],
                                           treatment_col='T', label_col='Y', alpha=0, lambda_=0,
                                           lr=0.001):
        super(TARNet, self).__init__() 
        
        # Data parameters
        self.df = df
        self.covariate_cols = covariate_cols
        self.treatment_col = treatment_col
        self.label_col = label_col
        self.lr = lr
        
        # Loss hyperparameters
        self.lambda_ = lambda_
        self.alpha = alpha
        
        self.u = max(1e-6, min(1 - 1e-6, sum(self.df[self.treatment_col].values) / len(self.df[self.treatment_col].values)))
        print(f'u: {self.u}')
        self.input_dim = len(covariate_cols)
        self.output_dim = 1 # for each arm we predict a single value, Y_hat
        self.hidden_dim = 200
        
        # # Device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.to(self.device)

        # Layers:
        # "CFR is implemented as a feed-forward
        # neural network with 3 fully-connected exponential-linear
        # layers for the representation and 3 for the hypothesis. Layer
        # sizes were 200 for all layers used for Jobs and 200 and 100
        # for the representation and hypothesis used for IHDP.""
        
        self.representation_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
        )
        self.control_arm = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.treatment_arm = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
            
        )
        # The model is trained using Adam's (Kingma & Ba, 2014). 
        self.optimizer = optim.Adam(
                               list(self.representation_layers.parameters())+
                               list(self.control_arm.parameters())+
                               list(self.treatment_arm.parameters()), lr=self.lr)
        # optimizer_representation = optim.Adam(list(self.representation_layers.parameters()), lr=lr)
        # optimizer_control = optim.Adam(list(self.control_arm.parameters()), lr=lr)
        # optimizer_treatment = optim.Adam(list(self.treatment_arm.parameters()), lr=lr)
    
        
        
        # For continuous data we use mean squared loss and for binary data, we'd use log-loss.
        self.prediction_loss = nn.MSELoss()

        print(self)

        # Ignore regularization for now
        # self.regularizing_term = self.lambda_ * torch.norm(self.weights, p=2) 
        
        # loss = self.hypothesis_loss + self.regularizing_term + self.IPM_penalty_loss

    def forward(self, X, t):

        representation = self.representation_layers(X)
        control_out = self.control_arm(representation)
        treatment_out = self.treatment_arm(representation)
        y_pred = t * treatment_out + (1 - t) * control_out
        
        return y_pred
        
    def fit(self, epochs=10, batch_size=64):
        # Set to training mode
        self.representation_layers.train()
        self.treatment_arm.train()
        #self.control_arm.train()

        # Get data
        
        print('data:')
        try:
            X = torch.tensor(self.df[self.covariate_cols].values, dtype=torch.float32)
            y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
            t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
            # X = X.to(self.device)
            # y = y.to(self.device)
            # t = t.to(self.device)
        except Exception as e:
            print(e)
            print('error')
        print(f'data shapes: {X.shape}, {y.shape}, {t.shape}')
        
        print('starting training')
        for epoch in range(epochs):
            print(epoch)

            # Shuffle data
            indices = torch.randperm(len(X))
            X = X[indices]
            y = y[indices]
            t = t[indices]

            batch_size = 64
            # Batch data
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                t_batch = t[i:i + batch_size]

                try:
                    # Zero gradients
                    self.optimizer.zero_grad()
                    # optimizer_representation.zero_grad()
                    # optimizer_control.zero_grad()
                    # optimizer_treatment.zero_grad()

                    y_pred = self.forward(X_batch, t_batch)
                except Exception as e:
                    print(e)
                    print('error')

                # Calculate weights
                weights = (t_batch / (2 * self.u)) + ((1 - t_batch) / (2 * (1 - self.u)))
                # print(weights)
                
                # Calculate prediction loss
                loss = torch.mean(weights * self.prediction_loss(y_pred, y_batch))
                #print(f'loss_prediction: {loss}')

                # representation_control = representation[control_conditon]
                # representation_treatment = representation[treatment_condition]

                #loss_pred = loss_y0 + loss_y1  # Prediction loss

                # Squared Linear MMD = IPM loss example -- idk what they even use in the paper
                #loss_ipm = self.alpha * torch.norm(representation_control.mean() - representation_treatment.mean(), p=2) ** 2 
                
                #loss_total = loss_pred + loss_ipm
                # Backpropagate both prediction loss (through all layers) and IPM loss (through representation layers)
                loss.backward()  # Prediction loss affects all layers
                
                # optimizer_representation.step()  # Update representation layers (affects both prediction and MMD)
                # optimizer_control.step()  # Update control prediction head
                # optimizer_treatment.step()  # Update treatment prediction head
                self.optimizer.step()
                # Print loss

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Prediction Loss: {loss.item()}')
            
    def predict(self, X):
        self.eval()
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