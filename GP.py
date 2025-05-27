import torch
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv(r'csv_files\df_baseline.csv')

feature_cols = [
    'Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags',
    'Turbo', 'Age',
    'Manufacturer_encoded', 'Category_encoded', 'GearBox_encoded',
    'Drive_4x4', 'Drive_Front', 'Drive_Rear'
]
target_col = 'Price'

# Clean and prepare features
df[feature_cols] = df[feature_cols].apply(lambda col: pd.to_numeric(
    col.astype(str).replace({'True': 1, 'False': 0}), errors='coerce'))
df = df.dropna(subset=feature_cols + [target_col])
df = df[df[target_col] > 0]  # Remove zero or negative prices

# Log-transform and standardize the target
log_price = np.log(df[target_col].values.astype(np.float32)).reshape(-1, 1)
log_price_scaler = StandardScaler()
y_log_scaled = log_price_scaler.fit_transform(log_price).flatten()

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# Convert to tensors
X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_log_scaled, dtype=torch.float32)

# GP setup
pyro.clear_param_store()
input_dim = X_train_tensor.shape[1]
kernel = gp.kernels.RBF(input_dim=input_dim)
kernel.lengthscale = torch.nn.Parameter(torch.ones(input_dim))  # Enable ARD

likelihood = gp.likelihoods.Gaussian()

# Select inducing points
M = 100
Xu = X_train_tensor[torch.randperm(X_train_tensor.size(0))[:M]]

vsgp = gp.models.VariationalSparseGP(
    X_train_tensor,
    y_train_tensor,
    kernel,
    Xu=Xu,
    likelihood=likelihood,
    whiten=True,
    jitter=1e-1
)

# Train model
losses = gp.util.train(vsgp, num_steps=1500)
plt.plot(losses)
plt.title("ELBO Loss (Sparse GP on Standardized Log-Price)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Predict on training data
with torch.no_grad():
    pred_mean, pred_var = vsgp(X_train_tensor, full_cov=False)

# Unstandardize and exponentiate
pred_mean_unscaled = log_price_scaler.inverse_transform(pred_mean.unsqueeze(-1)).squeeze()
pred_sd_unscaled = pred_var.sqrt() * log_price_scaler.scale_[0]  # scale the std deviation
# Ensure both are NumPy arrays (check before converting)
if torch.is_tensor(pred_mean_unscaled):
    pred_mean_unscaled = pred_mean_unscaled.numpy()
if torch.is_tensor(pred_sd_unscaled):
    pred_sd_unscaled = pred_sd_unscaled.numpy()
pred_price = np.exp(pred_mean_unscaled)
pred_price_upper = np.exp(pred_mean_unscaled + 2 * pred_sd_unscaled)
pred_price_lower = np.exp(pred_mean_unscaled - 2 * pred_sd_unscaled)
# Plot
true_price = df[target_col].values
plt.figure(figsize=(10, 6))
plt.errorbar(true_price, pred_price, 
             yerr=[(pred_price - pred_price_lower), 
                   (pred_price_upper - pred_price)],
             fmt='o', alpha=0.4, label="Predicted ±2σ")
plt.plot([true_price.min(), true_price.max()],
         [true_price.min(), true_price.max()],
         'k--', label="Perfect Prediction")
plt.xlabel("True Price")
plt.ylabel("Predicted Price (exp GP mean)")
plt.title("GP Predicted Price vs True Price (Log-Scaled Target)")
plt.legend()
plt.grid(True)
plt.show()
