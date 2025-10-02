import numpy as np
import pandas as pd
from basemodel import LinearModel
from basetrainer import Trainer

def load_and_preprocess_data(data_file: str = "data/train.csv"):
    dataset = pd.read_csv(data_file)
    """
    Divide the dataset into features and target

    You can do all possible modifications to features, but DO NOT change the targets

    return:
        features (np.ndarray): Input features, shape [num_samples, in_features]
        targets (np.ndarray): Target values, shape [num_samples]
    """

    if "Run_time" in dataset.columns:
        targets = dataset["Run_time"].values
        features = dataset.drop(columns=["Run_time"])
        
    #Standardization
    features = (features-features.mean(axis=0))/(features.std(axis=0) + 1e-8)
    
    print(f"Data size: {features.shape[0]}. Features num: {features.shape[1]}")
    return features, targets

class LinearRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Linear regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1).
        """
        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.random.randn(in_features,out_features) * 0.01

        self.bias = np.zeros((1,out_features))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        return np.dot(features, self.weight) + self.bias


    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for MSE loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): Predicted values, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        N = features.shape[0]
        error = predictions-targets
        dw = (2/N) * np.dot(features.T, error)
        db = (2/N) * np.sum(error, axis=0, keepdims=True)

        return dw, db

    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute MSE loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): True values, shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.
        """
        N = features.shape[0]
        loss = np.mean((predictions-targets) ** 2)

        dw, db = self.gradient(features,targets,predictions)

        self.weight -= learning_rate*dw
        self.bias -= learning_rate*db

        return loss

class LinearRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="mae"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)

    def compute_loss(self, batch_pred, batch_grd):
        """
        Compute loss based on model type with detailed checks for linear regression.

        Args:
            batch_pred: Predicted values, shape [batch_size, out_features].
            batch_grd: True values/labels, shape [batch_size, out_features].

        Returns:
            float: Mean loss for the batch.
        """
        assert batch_pred.shape == batch_grd.shape,\
            f"Shape mismatch: batch_pred {batch_pred.shape}, batch_grd {batch_grd.shape}"
        
        error = batch_pred - batch_grd
        loss = np.mean(np.square(error))

        return loss

def linear_regression_analytic(X, y):
    """
    Calculate the analytical linear regression results.

    Args:
        X (np.ndarray): Input features, shape [num_samples, in_features]
        y (np.ndarray): True values, shape [num_samples, out_features] or [num_samples,]

    Return:
        weight (np.ndarray): Model weight
        bias (np.ndarray | float): Model bias
    """
    N = X.shape[0]
    X_aug = np.hstack((X, np.ones((N,1))))
    y = np.atleast_2d(y).T if y.ndim == 1 else y  # 保证y为二维

    theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y  # 用pinv更稳健

    weight = theta[:-1]
    bias = theta[-1:]

    return weight, bias

class LogisticRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Logistic regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1 for binary classification).
        """
        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.random.randn(in_features,out_features) * 0.01

        self.bias = np.zeros((1,out_features))

 
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        linear_output = np.dot(features, self.weight) + self.bias
        return self._sigmoid(linear_output)

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for binary cross-entropy loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            predictions (np.ndarray): Predicted probabilities, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        m = features.shape[0]
        error = predictions - targets  # shape [m, out_features]
        dw = np.dot (features.T, error) / m
        db = np.sum(error, axis=0, keepdims=True) / m
        return dw, db

    
    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute binary cross-entropy loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.

        Returns:
            float: Binary cross-entropy loss for the batch.
        """
        # 1. 计算 loss
        m = features.shape[0]
        # 为防止 log(0)，加一个小 epsilon
        epsilon = 1e-8
        loss = - np.mean(targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))
        
        # 2. 计算梯度
        dw, db = self.gradient(features, targets, predictions)
        
        # 3. 参数更新
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return loss

class LogisticRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="f1"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)
        
    def compute_loss(self, batch_pred, batch_grd):
        epsilon = 1e-8  # 防止 log(0)
        loss = - np.mean(batch_grd * np.log(batch_pred + epsilon) + 
                        (1 - batch_grd) * np.log(1 - batch_pred + epsilon))
        return loss
