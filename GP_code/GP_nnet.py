# Deep Kernel Learning using MLP + Sparse GP with ARD

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyro
import pyro.contrib.gp as gp
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pyro.distributions as dist

# Load data

pyro.clear_param_store()

df = pd.read_csv(r'csv_files\df_baseline.csv')

# Define columns
feature_cols_base = [
    'Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags',
    'Turbo', 'Age', 'Category_encoded', 'GearBox_encoded',
    'Drive_4x4', 'Drive_Front', 'Drive_Rear'
]
target_col = 'Price'

# Split
train_df, test_df = train_test_split(df, test_size=0.2)

# Manufacturer encoding
train_df_for_encoding = train_df.copy()
train_df_for_encoding['Price'] = np.expm1(train_df[target_col])
avg_price_by_manufacturer = train_df_for_encoding.groupby('Manufacturer')['Price'].mean()
bins = [2000, 10000, 15000, 20000, 25000, 32000, float('inf')]
labels = [1,2,3,4,5,6]
manufacturer_price_bins = pd.cut(avg_price_by_manufacturer, bins=bins, labels=labels)
train_df['Manufacturer_encoded'] = train_df['Manufacturer'].map(manufacturer_price_bins)
test_df['Manufacturer_encoded'] = test_df['Manufacturer'].map(manufacturer_price_bins)
median_bin = train_df['Manufacturer_encoded'].astype(float).median()
test_df['Manufacturer_encoded'].fillna(median_bin, inplace=True)

# Merge features
feature_cols = feature_cols_base + ['Manufacturer_encoded']

# Convert to numeric
for df_ in [train_df, test_df]:
    df_[feature_cols] = df_[feature_cols].apply(lambda col: pd.to_numeric(col.astype(str).replace({'True': 1, 'False': 0}), errors='coerce'))
train_df.dropna(subset=feature_cols + [target_col], inplace=True)
test_df.dropna(subset=feature_cols + [target_col], inplace=True)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
log_train_price = np.log(train_df[target_col].values.astype(np.float32)).reshape(-1, 1)
log_test_price = np.log(test_df[target_col].values.astype(np.float32)).reshape(-1, 1)
y_train = scaler.fit_transform(log_train_price).flatten()
y_test = scaler.transform(log_test_price).flatten()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# MLP feature extractor
class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=len(feature_cols)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
    
output_dim = len(feature_cols)
feature_net = MLPFeatureExtractor(input_dim=X_train_tensor.shape[1])

# Custom kernel wrapper
class FeatureTransformedKernel(gp.kernels.Kernel):
    def __init__(self, base_kernel, feature_net, output_dim):
        super().__init__(input_dim=output_dim)
        self.base_kernel = base_kernel
        self.feature_net = feature_net
        self.name = "FeatureTransformedKernel"

    def forward(self, X, Z=None, diag=False):
        X_feat = self.feature_net(X)
        Z_feat = self.feature_net(Z) if Z is not None else None
        return self.base_kernel(X_feat, Z_feat, diag=diag)

# RBF kernel with ARD
base_kernel = gp.kernels.RBF(input_dim=output_dim, lengthscale=torch.ones(output_dim))
kernel = FeatureTransformedKernel(base_kernel, feature_net, output_dim)
likelihood = gp.likelihoods.Gaussian()

# Select inducing points (raw feature space, not transformed)
M = 300
kmeans = KMeans(n_clusters=M).fit(X_train)
Xu_np = kmeans.cluster_centers_
Xu_tensor = torch.tensor(Xu_np, dtype=torch.float32)

# Do not transform Xu_tensor — let the GP handle it via the kernel
vsgp = gp.models.VariationalSparseGP(
    X_train_tensor, y_train_tensor,
    kernel, Xu=Xu_tensor,
    likelihood=likelihood, whiten=True, jitter=1e-1
)

pyro.clear_param_store()

# Train
losses = gp.util.train(vsgp, num_steps=1500)
plt.plot(losses)
plt.title("ELBO Loss (Deep Kernel GP)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Predict
with torch.no_grad():
    pred_mean, pred_var = vsgp(X_test_tensor, full_cov=False)

pred_mean_unscaled = scaler.inverse_transform(pred_mean.unsqueeze(-1)).squeeze()
pred_sd_unscaled = pred_var.sqrt().unsqueeze(-1).numpy() * scaler.scale_[0]

pred_mean_unscaled = np.array(pred_mean_unscaled)

pred_price = np.exp(pred_mean_unscaled)
pred_price_upper = np.exp(pred_mean_unscaled + 2 * pred_sd_unscaled.squeeze())
pred_price_lower = np.exp(pred_mean_unscaled - 2 * pred_sd_unscaled.squeeze())
true_price = np.exp(scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze())

plt.figure(figsize=(10, 6))
plt.errorbar(true_price, pred_price, yerr=[pred_price - pred_price_lower, pred_price_upper - pred_price], fmt='o', alpha=0.4, label="Predicted ±2σ")
plt.plot([true_price.min(), true_price.max()], [true_price.min(), true_price.max()], 'k--', label="Perfect Prediction")
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Deep Kernel GP: Predicted vs True Price")
plt.legend()
plt.grid(True)
plt.show()

residuals = true_price - pred_price
plt.figure(figsize=(8,5))
plt.scatter(true_price, residuals, alpha=0.25)
plt.axhline(0, ls='--', lw=1)
plt.xlabel("True price")
plt.ylabel("Residual (True − Pred)")
plt.title("Residuals vs. True Price")
plt.grid(True)
plt.show()

mae = mean_absolute_error(true_price, pred_price)
rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

lengthscales = base_kernel.lengthscale.detach().cpu().numpy().flatten()
relevance = 1 / lengthscales

print("Feature relevance by original input feature (via ARD):")
for feat, score in sorted(zip(feature_cols, relevance), key=lambda x: -x[1]):
    print(f"{feat:15s}  relevance: {score:.4f}")

def plot(model, feature_index=2, X_train_tensor=None, y_train_tensor=None, n_test=500, scaler=None, num_points=50):
    """
    Plot GP posterior mean and uncertainty along a single feature index,
    keeping other features fixed at their mean. Shows actual price.
    Limits training data plot to num_points for clarity.
    """
    assert X_train_tensor is not None and y_train_tensor is not None and scaler is not None

    # Create test inputs: all features = mean
    X_base = X_train_tensor.mean(dim=0).repeat(n_test, 1)
    
    # Vary only one feature
    x_range = torch.linspace(-4,4, n_test)
    X_base[:, feature_index] = x_range

    # Predict

    with torch.no_grad():
        mean, cov = model(X_base, full_cov=True)
        std = cov.diag().sqrt()

    # Undo standardization
    mean_unscaled = scaler.inverse_transform(mean.unsqueeze(-1)).squeeze()
    std_unscaled = std.numpy() * scaler.scale_[0]

    # Undo log transform
    price_pred = np.exp(mean_unscaled)
    price_upper = np.exp(mean_unscaled + 2 * std_unscaled)
    price_lower = np.exp(mean_unscaled - 2 * std_unscaled)

    # Plot GP prediction
    plt.figure(figsize=(10, 6))
    plt.plot(x_range.numpy(), price_pred, 'r', label=" GP Posterior Mean")
    plt.fill_between(x_range.numpy(), price_lower, price_upper,
                     alpha=0.3, color='C0', label='±2σ')
    
    # Sample a few training points to plot
    total_points = X_train_tensor.shape[0]
    indices = torch.randperm(total_points)[:num_points]
    x_sample = X_train_tensor[indices, feature_index].numpy()
    y_sample = np.exp(scaler.inverse_transform(y_train_tensor[indices].unsqueeze(-1)).squeeze())

    # Plot reduced training data
    plt.scatter(x_sample, y_sample, color='k', s=20, label=f"data", alpha=0.7)

    plt.xlabel(f"Mileage (standardized)")
    plt.ylabel("True Price ")
    plt.title(f"SE kernel")
    plt.legend()
    plt.grid(True)
    plt.show()

plot(model=vsgp,
     feature_index=2,
     X_train_tensor=X_train_tensor,
     y_train_tensor=y_train_tensor,
     scaler=scaler,
     num_points=500)
