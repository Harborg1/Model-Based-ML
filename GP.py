import torch
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Load data
df = pd.read_csv(r'csv_files\df_baseline.csv')

# Define columns
feature_cols_base = [
    'Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags',
    'Turbo', 'Age', 'Category_encoded', 'GearBox_encoded',
    'Drive_4x4', 'Drive_Front', 'Drive_Rear'
]
target_col = 'Price'

# Split first
train_df, test_df = train_test_split(df, test_size=0.2)

# Compute log-untransformed price in train
train_df_for_encoding = train_df.copy()
train_df_for_encoding['Price'] = np.expm1(train_df[target_col])

# Compute mean price per manufacturer using only training data
avg_price_by_manufacturer = train_df_for_encoding.groupby('Manufacturer')['Price'].mean()

# Bin the average prices into ordinal categories
bins = [2000, 10000, 15000, 20000, 25000, 32000, float('inf')]
labels = [1, 2, 3, 4, 5, 6]
manufacturer_price_bins = pd.cut(avg_price_by_manufacturer, bins=bins, labels=labels)

# Map the encoding to train and test separately
train_df['Manufacturer_encoded'] = train_df['Manufacturer'].map(manufacturer_price_bins)
test_df['Manufacturer_encoded'] = test_df['Manufacturer'].map(manufacturer_price_bins)

# Fill missing values in test set with the median bin from training set
median_bin = train_df['Manufacturer_encoded'].astype(float).median()
test_df['Manufacturer_encoded'].fillna(median_bin, inplace=True)

# Merge feature columns
feature_cols = feature_cols_base + ['Manufacturer_encoded']

# Apply numeric conversion
for df_ in [train_df, test_df]:
    df_[feature_cols] = df_[feature_cols].apply(lambda col: pd.to_numeric(
        col.astype(str).replace({'True': 1, 'False': 0}), errors='coerce'))
    
# Drop missing
train_df.dropna(subset=feature_cols + [target_col], inplace=True)
test_df.dropna(subset=feature_cols + [target_col], inplace=True)

# Standardize inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])

# Target: log-transform and standardize
log_train_price = np.log(train_df[target_col].values.astype(np.float32)).reshape(-1, 1)
log_test_price = np.log(test_df[target_col].values.astype(np.float32)).reshape(-1, 1)

y_train = scaler.fit_transform(log_train_price).flatten()
y_test = scaler.transform(log_test_price).flatten()


#CELL 2 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# GP setup
pyro.clear_param_store()
input_dim = X_train_tensor.shape[1]
kernel = gp.kernels.RBF(input_dim=input_dim)
kernel.lengthscale = torch.nn.Parameter(torch.ones(input_dim)*0.5)  # Enable ARD

likelihood = gp.likelihoods.Gaussian()

# Select inducing points from training data only
M = 300
kmeans = KMeans(n_clusters=M).fit(X_train)
Xu = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

vsgp = gp.models.VariationalSparseGP(
    X_train_tensor,
    y_train_tensor,
    kernel,
    Xu=Xu,
    likelihood=likelihood,
    whiten=True,
    jitter=1e-1
)

#CELL 3 

# Train model

losses = gp.util.train(vsgp, num_steps=1500)
plt.plot(losses)
plt.title("ELBO Loss (Sparse GP on Standardized Log-Price)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

#Cell 4 Predict and plot

# ── Predict on TEST set ──
with torch.no_grad():
    pred_mean, pred_var = vsgp(X_test_tensor, full_cov=False)

# Unstandardize and exponentiate
pred_mean_unscaled = scaler.inverse_transform(pred_mean.unsqueeze(-1)).squeeze()
pred_sd_unscaled = pred_var.sqrt() * scaler.scale_[0]

# Convert to numpy
if torch.is_tensor(pred_mean_unscaled):
    pred_mean_unscaled = pred_mean_unscaled.numpy()
if torch.is_tensor(pred_sd_unscaled):
    pred_sd_unscaled = pred_sd_unscaled.numpy()

pred_price = np.exp(pred_mean_unscaled)
pred_price_upper = np.exp(pred_mean_unscaled + 2 * pred_sd_unscaled)
pred_price_lower = np.exp(pred_mean_unscaled - 2 * pred_sd_unscaled)

# True prices from TEST set
true_price = np.exp(scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze())

# ── Plot prediction vs. true ──
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
plt.title("GP Predicted Price vs True Price (Test set)")
plt.legend()
plt.grid(True)
plt.show()

# ── Residuals ──
residuals = true_price - pred_price
plt.figure(figsize=(8,5))
plt.scatter(true_price, residuals, alpha=0.25)
plt.axhline(0, ls='--', lw=1)
plt.xlabel("True price")
plt.ylabel("Residual (True − Pred)")
plt.title("Residuals vs. True Price (Test Set)")
plt.grid(True)
plt.show()

# ── MAE ──
mae = mean_absolute_error(true_price, pred_price)


# Scatter plot with MAE
plt.figure(figsize=(8,6))
plt.scatter(true_price, pred_price, alpha=0.3, label='Predictions')
plt.plot([true_price.min(), true_price.max()],
         [true_price.min(), true_price.max()],
         'k--', label='Perfect Prediction')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("GP Predicted Price vs True Price (Test Set)\nMAE = {:.2f}".format(mae))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)
print("MAE:",mae)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# ── ARD relevance ──
lengthscales = kernel.lengthscale.detach().cpu().numpy().flatten()
relevance = 1 / lengthscales

print("\nFeature relevance via ARD (lower lengthscale = more relevant):")
for feat, score in sorted(zip(feature_cols, relevance), key=lambda x: -x[1]):
    print(f"{feat:20s}  relevance: {score:.4f}")

