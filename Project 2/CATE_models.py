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
from sklearn.model_selection import train_test_split

# for visualizing the representation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#######################################################################################

## TARNet or CFRMMD depending on alpha ##
class TARNet(nn.Module):
    def __init__(self, df, covariate_cols=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',
                                           'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20',
                                           'X21','X22','X23','X24','X25'],
                                           treatment_col='T', label_col='Y', ite_label='ITE',alpha=0, lambda_=0,beta=1,
                                           weight_decay=0.01,
                                           lr=0.001, verbose=False, target_reg=False):
        super(TARNet, self).__init__() 
        
        # Data parameters
        self.df = df
        self.covariate_cols = covariate_cols
        self.treatment_col = treatment_col
        self.label_col = label_col
        self.ite_label = ite_label
        self.lr = lr
        self.df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        
        self.verbose = verbose
        
        # Loss hyperparameters
        self.lambda_ = lambda_
        self.alpha = alpha
        self.target_reg = target_reg
        self.beta = beta
        
        self.u = max(1e-6, min(1 - 1e-6, sum(self.df[self.treatment_col].values) / len(self.df[self.treatment_col].values)))
        print(f'u: {self.u}')
        self.input_dim = len(covariate_cols)
        self.output_dim = 1 # for each arm we predict a single value, Y_hat
        self.hidden_dim = 200
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.to(self.device)

        # Layers:
        # "CFR is implemented as a feed-forward
        # neural network with 3 fully-connected exponential-linear
        # layers for the representation and 3 for the hypothesis. Layer
        # sizes were 200 for all layers used for Jobs and 200 and 100
        # for the representation and hypothesis used for IHDP.""
        
        self.representation_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
        )
        self.control_arm = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.treatment_arm = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.BatchNorm1d(self.hidden_dim), 
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.output_dim)
            
        )
        
        self.epsilon = nn.Parameter(torch.tensor(1.0))
        
        # The model is trained using Adam's (Kingma & Ba, 2014). 
        # self.optimizer = optim.Adam(
        #                        list(self.representation_layers.parameters())+
        #                        list(self.control_arm.parameters())+
        #                        list(self.treatment_arm.parameters()), lr=self.lr)
        self.optimizer_representation = optim.Adam(list(self.representation_layers.parameters()) + [self.epsilon], lr=lr, weight_decay=weight_decay)
        self.optimizer_control = optim.Adam(list(self.control_arm.parameters()), lr=lr, weight_decay=weight_decay)
        self.optimizer_treatment = optim.Adam(list(self.treatment_arm.parameters()), lr=lr, weight_decay=weight_decay)
    
        # For continuous data we use mean squared loss and for binary data, we'd use log-loss.
        self.prediction_loss = nn.MSELoss()

    def forward(self, X, t):

        representation = self.representation_layers(X)
        control_out = self.control_arm(representation)
        treatment_out = self.treatment_arm(representation)
        y_pred = t * treatment_out + (1 - t) * control_out
        
        return y_pred, representation, control_out, treatment_out
        
    def fit(self, epochs=10, batch_size=64):
        # Set to training mode
        self.representation_layers.train()
        self.treatment_arm.train()
        self.control_arm.train()

        # Get data
        losses = []
        val_losses = []
        print('data:')
        
        X = torch.tensor(self.df[self.covariate_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
        X = X.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        
        print(f'data shapes: {X.shape}, {y.shape}, {t.shape}')
        
        print('starting training')
        for epoch in range(epochs):
            epoch_losses = []

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

               
                # Zero gradients
                #self.optimizer.zero_grad()
                self.optimizer_representation.zero_grad()
                self.optimizer_control.zero_grad()
                self.optimizer_treatment.zero_grad()

                y_pred, representation, _, _ = self.forward(X_batch, t_batch)
                

                # Calculate weights
                weights = (t_batch / (2 * self.u)) + ((1 - t_batch) / (2 * (1 - self.u)))
                # print(weights)
                
                # Calculate prediction loss
                loss_pred = torch.mean(weights * self.prediction_loss(y_pred, y_batch))
                #print(f'loss_prediction: {loss}')
                #loss = loss_pred.clone()
                
                control_condition = (t_batch==0)
                
                representation_control = representation[control_condition]
                representation_treatment = representation[~control_condition]

                #loss_pred = loss_y0 + loss_y1  # Prediction loss

                # Squared Linear MMD = IPM loss example -- idk what they even use in the paper
                loss_ipm = self.alpha * torch.norm(representation_control.mean() - representation_treatment.mean(), p=2) ** 2 
                #loss = loss.clone() + loss_ipm
                
                # Regularizing loss
                # Ignore regularization for now
                #regularizing_term = self.lambda_ * torch.norm(self.weights, p=2) 
                #loss += regularizing_term
                
                if self.target_reg:
                    target = y_batch - y_pred + self.epsilon*(t_batch/ self.u + (1-t_batch)/(1-self.u))
                    loss_target = self.beta * torch.mean(target)
                    loss_pred += loss_target

                # Backpropagate **only** prediction loss (updates all optimizers)
                loss_pred.backward(retain_graph=True)
                #self.optimizer_representation.step()
                self.optimizer_control.step()
                self.optimizer_treatment.step()

                # Backpropagate **only** IPM loss (updates only representation layers)
                #self.optimizer_representation.zero_grad()  # Zero out before IPM step
                #loss = loss_pred.clone() + loss_ipm
                loss_ipm.backward()
                self.optimizer_representation.step()
                
                # Backpropagate both prediction loss (through all layers) and IPM loss (through representation layers)
                # loss.backward()  
                # self.optimizer.step()
                
                loss = loss_pred + loss_ipm
                
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            losses.append(avg_loss)

            if epoch % 10 == 0 and self.verbose:
                print('')
                print(f'Epoch: {epoch}, Loss: {loss_pred.item()}')
                
            
            # Calculate test set loss
            test_X = torch.tensor(self.test_df[self.covariate_cols].astype(float).values, dtype=torch.float32).to(self.device)
            test_y = torch.tensor(self.test_df[self.label_col].values, dtype=torch.float32).to(self.device)
            test_t = torch.tensor(self.test_df[self.treatment_col].values, dtype=torch.float32).to(self.device)

            self.eval()
            with torch.no_grad():
                test_y_pred, _, _, _ = self.forward(test_X, test_t)
                test_loss = self.prediction_loss(test_y_pred, test_y).item()
                val_losses.append(test_loss)
                print(f'Test set loss: {test_loss}')
                
        print('done training')
        

        return losses, val_losses
        
           
    def predict_train(self, use_ite=False):
        X = torch.tensor(self.df[self.covariate_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
        X = X.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        if use_ite:
            ite = torch.tensor(self.df[self.ite_label].astype(float).values, dtype=torch.float32)
        else:
            ite = None
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed for prediction
            return self.forward(X,t), X, y, t, ite
    
    def predict_test(self, use_ite=False):
        X = torch.tensor(self.test_df[self.covariate_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(self.test_df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.test_df[self.treatment_col].values, dtype=torch.float32)
        X = X.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        if use_ite:
            ite = torch.tensor(self.test_df[self.ite_label].astype(float).values, dtype=torch.float32)
        else:
            ite = None
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed for prediction
            return self.forward(X,t), X, y, t, ite
    
    # plot the t-sne
    def plot_representation_space(self, representation=None, t=None, n_components=2):
        """Plot the representation space"""
        
        if representation is None:
            representation = torch.tensor(self.df[self.covariate_cols].astype(float).values, dtype=torch.float32)
            t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
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
        plt.legend()
        plt.show()
    
    def compute_PEHE_ATE_metrics(self, control_output, treatment_output, t, ite=None):
        """Compute comparison metrics"""
        ite_pred = treatment_output - control_output 
        if ite is not None:
            ite_true = ite 
        else:
            ite_true = torch.where(t == 1, y, -y)
        #print(ite_pred, ite_true)
        pehe = torch.mean((ite_pred - ite_true) ** 2)
        #print(pehe)
    
        ate_pred = torch.mean(ite_pred)  
        ate_true = torch.mean(ite_true)  
        ate_error = torch.abs(ate_pred - ate_true)
        
        return pehe.item(), ate_error.item(), ite_pred, ate_pred 

#######################################################################################

## Dragonnet ##
class Dragonnet(nn.Module):
    def __init__(self, df, covariate_cols=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10',
                                           'X11','X12','X13','X14','X15','X16','X17','X18','X19','X20',
                                           'X21','X22','X23','X24','X25'],
                                           treatment_col='T', label_col='Y',ite_label='ITE', target_reg=False,
                                           lr=0.001, alpha=1, beta = 1, weight_decay=0): 
        # default alpha and beta from paper directly
        # epsilon is learned
        super(Dragonnet, self).__init__() 
        
        # Data parameters
        self.df = df
        self.covariate_cols = covariate_cols
        self.treatment_col = treatment_col
        self.label_col = label_col
        self.ite_label = ite_label
        self.lr = lr
        self.df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        
        # Loss hyperparameters
        self.target_reg = target_reg
        self.alpha = alpha
        self.epsilon=nn.Parameter(torch.tensor(1.0))
        self.beta = beta
        self.weight_decay = weight_decay
        
        self.u = max(1e-6, min(1 - 1e-6, sum(self.df[self.treatment_col].values) / len(self.df[self.treatment_col].values)))
        print(f'u: {self.u}')
        self.input_dim = len(covariate_cols)
        self.output_dim = 1 # for each arm we predict a single value, Y_hat
        self.hidden_dim = 200
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.to(self.device)

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
        self.propensity_arm = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
            
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
                               list(self.propensity_arm.parameters())+
                               list(self.treatment_arm.parameters())+
                               [self.epsilon], lr=self.lr, weight_decay=self.weight_decay)
        # optimizer_representation = optim.Adam(list(self.representation_layers.parameters()), lr=lr)
        # optimizer_control = optim.Adam(list(self.control_arm.parameters()), lr=lr)
        # optimizer_treatment = optim.Adam(list(self.treatment_arm.parameters()), lr=lr)
    
        
        
        # For continuous data we use mean squared loss and for binary data, we'd use log-loss.
        self.prediction_loss = nn.MSELoss()
        self.propensity_loss = nn.BCELoss()

        

        # Ignore regularization for now
        # self.regularizing_term = self.lambda_ * torch.norm(self.weights, p=2) 
        
        # loss = self.hypothesis_loss + self.regularizing_term + self.IPM_penalty_loss

    def forward(self, X, t):

        representation = self.representation_layers(X)
        control_out = self.control_arm(representation)
        treatment_out = self.treatment_arm(representation)
        propensity = self.propensity_arm(representation)
        y_pred = t * treatment_out + (1 - t) * control_out
        
        return y_pred, representation, control_out, treatment_out, propensity
        
    def fit(self, epochs=10, batch_size=64):
        # Set to training mode
        self.representation_layers.train()
        self.treatment_arm.train()
        self.control_arm.train()
        self.propensity_arm.train()

        # Get data
        losses = []
        print('data:')
        
        X = torch.tensor(self.df[self.covariate_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
        X = X.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        
        print(f'data shapes: {X.shape}, {y.shape}, {t.shape}')
        
        print('starting training')
        for epoch in range(epochs):
            epoch_losses = []
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

                    y_pred, representation, _, _, propensity = self.forward(X_batch, t_batch)
                except Exception as e:
                    print(e)
                    print('error')

                # Calculate weights
                weights = (t_batch / (2 * self.u)) + ((1 - t_batch) / (2 * (1 - self.u)))
                # print(weights)
                
                # Calculate prediction loss
                loss = torch.mean(weights * self.prediction_loss(y_pred, y_batch))
                #print(f'loss_prediction: {loss}')
                
                # control_condition = (t_batch==0)
                
                # representation_control = representation[control_condition]
                # representation_treatment = representation[~control_condition]

                #loss_pred = loss_y0 + loss_y1  # Prediction loss

                # Propensity loss
                print(propensity.shape)
                loss_propensity = self.alpha * self.propensity_loss(propensity.squeeze(), t_batch)
                loss += loss_propensity

                if self.target_reg:
                    target = y_batch - y_pred + self.epsilon*(t_batch/ propensity.squeeze() + (1-t_batch)/(1-propensity.squeeze()))
                    loss_target = self.beta * torch.mean(target)
                    loss+= loss_target
                
                #loss_total = loss_pred + loss_ipm
                # Backpropagate both prediction loss (through all layers) and IPM loss (through representation layers)
                loss.backward()  # Prediction loss affects all layers
                
                # optimizer_representation.step()  # Update representation layers (affects both prediction and MMD)
                # optimizer_control.step()  # Update control prediction head
                # optimizer_treatment.step()  # Update treatment prediction head
                self.optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
        print('done training')
        return losses
        
            
    def predict_train(self, use_ite=False):
        X = torch.tensor(self.df[self.covariate_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(self.df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
        X = X.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        if use_ite:
            ite = torch.tensor(self.df[self.ite_label].astype(float).values, dtype=torch.float32)
        else:
            ite = None
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed for prediction
            return self.forward(X,t), X, y, t, ite
    
    def predict_test(self, use_ite=False):
        X = torch.tensor(self.test_df[self.covariate_cols].astype(float).values, dtype=torch.float32)
        y = torch.tensor(self.test_df[self.label_col].values, dtype=torch.float32)
        t = torch.tensor(self.test_df[self.treatment_col].values, dtype=torch.float32)
        X = X.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        if use_ite:
            ite = torch.tensor(self.test_df[self.ite_label].astype(float).values, dtype=torch.float32)
        else:
            ite = None
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed for prediction
            return self.forward(X,t), X, y, t, ite
    
    # plot the t-sne
    def plot_representation_space(self, representation=None, t=None, n_components=2):
        """Plot the representation space"""
        
        if representation is None:
            representation = torch.tensor(self.df[self.covariate_cols].astype(float).values, dtype=torch.float32)
            t = torch.tensor(self.df[self.treatment_col].values, dtype=torch.float32)
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
        plt.legend()
        plt.show()
    
    def compute_PEHE_ATE_metrics(self, control_output, treatment_output, t, ite=None):
        """Compute comparison metrics"""
        ite_pred = treatment_output - control_output 
        if ite is not None:
            ite_true = ite 
        else:
            ite_true = torch.where(t == 1, y, -y)
        #print(ite_pred, ite_true)
        pehe = torch.mean((ite_pred - ite_true) ** 2)
        #print(pehe)
    
        ate_pred = torch.mean(ite_pred)  
        ate_true = torch.mean(ite_true)  
        ate_error = torch.abs(ate_pred - ate_true)
        
        return pehe.item(), ate_error.item(), ite_pred, ate_pred 