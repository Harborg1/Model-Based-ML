#!/usr/bin/env python
# coding: utf-8

# # Model Based Machine Learning - Used Cars Price Prediction

# ## Imports

# In[256]:


# Uncomment the next line and run the cell to install Pyro for Jupyter Notebook:
#!pip install pyro-ppl


# In[257]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pyro.infer import Predictive
from pyro.optim import ClippedAdam, Adam
from torch.distributions import constraints
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.image as mpimg


# In[258]:


"""Command to create a python file of the notebook (change path as needed):"""""
#c:\Users\cahar\AppData\Local\Programs\Python\Python311\python.exe -m jupyter nbconvert --to script model_based_ml_project.ipynb


# # Data: Description, Preprocessing, and Analysis

# In[259]:


# Loading the data and inspect the first rows
df = pd.read_csv('csv_files\car_price_prediction.csv')
df.head()


# In[260]:


# Inspect the dataset: we start with the shape of the dataset, columns, and types
print("Dataset shape: ", df.shape)
print("Columns and Types:")
print(df.dtypes)


# In[261]:


#check for duplications
df.duplicated().sum()


# In[262]:


# we will drop duplications and check for missing values
df.drop_duplicates(inplace= True)
df.isnull().sum()


# In[263]:


# Number of columns per row in the plot grid
n_cols = 3
n_rows = -(-len(df.columns) // n_cols)  # Ceiling division

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(df.columns):
    ax = axes[i]
    try:
        if col == "Price":
            price_data = df[col].astype(float)
            upper_limit = price_data.quantile(0.995)
            price_data = price_data[price_data <= upper_limit]
            price_data.plot(kind='hist', bins=30, ax=ax)
            ax.set_title(f"Histogram of {col} (≤ 99.5th percentile)")
        else:
            df[col].astype(float).plot(kind='hist', bins=30, ax=ax)
            ax.set_title(f"Histogram of {col}") 
    except:
        # For categorical data, show only the top 20 most frequent categories
        vc = df[col].value_counts(dropna=True)
        vc[:10].plot(kind='bar', ax=ax)
        ax.set_title(f"Histogram of {col} (Top 10)" if df.nunique()[col]>=10 else f"Histogram of {col}" , fontsize=11)
        ax.tick_params(axis='x', rotation=60)
    ax.set_xlabel('')
    ax.set_ylabel('Count')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
fig.savefig(r'images\histograms_dataframe.png', bbox_inches='tight')
plt.show()
plt.close()


# As expected, there are many values that need preprocessing, and we can also ignore some features that we believe that do not have influence over the price (e.g. Steering wheels side, color and ID). Other features need to be treated differently: for example, milage contains the string "KM". This should be removed and the values should contain only the numerical value. Other features such as engine volume, contain some additional information about the engine like "turbo" - which we will split into another binary feature that will contain values 0 or 1 for whether the engine is a turbo engine (0 if not, else 1). Additionally, many of the features are categorical, hence we will use one-hot encoding or ordinal encoding to fix them and fit them to a regression model.
# 
# Another desicion is to drop the attribute "model". This is due to high cardinality, as we have 1590 unique values (see the list below). Moreover, the "model" feature can be captured by other features of the car. i.e. the common sense is saying that the model is a major factor in the price, however, we believe that the rest of the features will capture the quality of a certain model (e.g. engine volume, leather seats, number of airbags etc). 

# In[264]:


# Check the number of unique values of each column
df.nunique()


# In[265]:


# Drop irrelevant columns -4 columns
df_cleaned = df.drop(columns=['ID', 'Model', 'Color', 'Wheel','Doors'])

# Levy to numeric
df_cleaned['Levy'] = df_cleaned['Levy'].replace('-', pd.NA)
df_cleaned['Levy'] = pd.to_numeric(df_cleaned['Levy'], errors='coerce')

#Leather interior to binary
df_cleaned['Leather interior'] = df_cleaned['Leather interior'].map({'Yes': 1, 'No': 0})


# In[266]:


#check the unique values and additional strings in engine volume
engine_volume_values = df_cleaned['Engine volume'].unique()
sorted(engine_volume_values)


# In[267]:


# Create 'Turbo' binary column
df_cleaned['Turbo'] = df_cleaned['Engine volume'].str.contains('Turbo').astype(int)

# Extract numeric part of engine volume
df_cleaned['Engine volume'] = df_cleaned['Engine volume'].str.replace(' Turbo', '', regex=False).astype(float)


# In[268]:


#Drop 'Fuel type'
df_cleaned = df_cleaned.drop(columns=['Fuel type'])

# Mileage - remove ' km' and convert to numeric
df_cleaned['Mileage'] = df_cleaned['Mileage'].str.replace(' km', '', regex=False).str.replace(',', '')
df_cleaned['Mileage'] = pd.to_numeric(df_cleaned['Mileage'], errors='coerce')

# Cylinders and Airbags - numeric
df_cleaned['Cylinders'] = pd.to_numeric(df_cleaned['Cylinders'], errors='coerce')
df_cleaned['Airbags'] = pd.to_numeric(df_cleaned['Airbags'], errors='coerce')

# Preprocessing production year to Age
dataset_year = 2022 #The year that the used car dataset was published.
df_cleaned['Age'] = dataset_year - df_cleaned['Prod. year']
df_cleaned = df_cleaned.drop(columns=['Prod. year'])

# Preview the cleaned dataframe
df_cleaned.head()
print(df_cleaned.columns)


# In[269]:


# Number of columns per row in the plot grid
n_cols = 3
n_rows = -(-len(df_cleaned.columns) // n_cols)  # Ceiling division

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(df_cleaned.columns):
    ax = axes[i]
    try:
        if col == "Price":
            price_data = df_cleaned[col].astype(float)
            upper_limit = price_data.quantile(0.995)
            price_data = price_data[price_data <= upper_limit]
            price_data.plot(kind='hist', bins=30, ax=ax)
            ax.set_title(f"Histogram of {col} (≤ 99.9th percentile)")
        else:
            df_cleaned[col].astype(float).plot(kind='hist', bins=30, ax=ax)
            ax.set_title(f"Histogram of {col}")
    except:
        vc = df_cleaned[col].value_counts(dropna=True)
        vc[:10].plot(kind='bar', ax=ax)
        ax.set_title(f"Histogram of {col} (Top 10)" if df_cleaned.nunique()[col] >= 10 else f"Histogram of {col}", fontsize=11)
        ax.tick_params(axis='x', rotation=60)
    ax.set_xlabel('')
    ax.set_ylabel('Count')


# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
fig.savefig(r'images\histograms_dataframe_cleaned.png', bbox_inches='tight')
plt.show()
plt.close()


# In[270]:


print("Descriptive Statistics:")
df_cleaned.describe().T


# In[271]:


# Select only numerical columns
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Set up plot
n_cols = 3
n_rows = -(-len(numeric_cols) // n_cols)  # Ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

# Create boxplot for each numeric feature
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df_cleaned, x=col, ax=axes[i])
    axes[i].set_title(f'Boxplot: {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
fig.savefig(r'images\boxplots.png', bbox_inches='tight')
plt.show()
plt.close()


# It looks like the data for the price and mileage, Levy and age, got some outliers the influence the distribution. We will inspect the prices further:

# In[272]:


print(df_cleaned['Price'].describe())

print(" ")

Q1 = df_cleaned['Price'].quantile(0.25)
Q3 = df_cleaned['Price'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

print("Upper bound:", upper_bound)


# In[273]:


print(f"Number of entries: {len(df_cleaned[df_cleaned['Price'] > 50000])}")


# In[274]:


print(f"Number of entries: {len(df_cleaned[df_cleaned['Price'] < 2000])}")


# There are 908 records of cars that their price is greater than 50,000. Also, 3,141 cars that are priced under 2,000. We will drop them, and plot the price distribution again:

# In[275]:


df_cleaned = df_cleaned[(df_cleaned['Price'] >= 2000) & (df_cleaned['Price'] <= 50000)]


# In[276]:


# Plot without those extreme prices
plt.figure(figsize=(8, 4))
plt.hist(df_cleaned['Price'], bins=50)
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Distribution of Car Prices')
plt.show()


# Due to heavy skewness of the prices, we will try to plot the distribution in it's log-values. It might help us later in the baseline model, to improve linearity between the predicted variable and the predictors.

# In[277]:


plt.figure(figsize=(8,4))
plt.hist(np.log1p(df_cleaned['Price']), bins=50)
plt.xlabel('log(price + 1)')
plt.ylabel('Count')
plt.title('Log-Transformed Distribution of Car Price')
plt.show()


# In[278]:


#we do the same for mileage:
print(df_cleaned['Mileage'].describe())

print(" ")

Q1 = df_cleaned['Mileage'].quantile(0.25)
Q3 = df_cleaned['Mileage'].quantile(0.75)
IQR = Q3 - Q1
print("upper bound: ", Q3 + 1.5 * IQR)


# since the upper bound for mileage is 352,000, we will check for removing a reasonable number of recordings, to avoid removing to much rows:

# In[279]:


print(f"Number of entries: {len(df_cleaned[df_cleaned['Mileage'] > 370000])}")


# In[280]:


df_cleaned = df_cleaned[df_cleaned['Mileage'] <= 370000]


# In[281]:


lower = df_cleaned['Mileage'].quantile(0.005)
upper = df_cleaned['Mileage'].quantile(0.995)

plt.figure(figsize=(8,4))
plt.hist(df_cleaned['Mileage'][(df_cleaned['Mileage'] >= lower) & (df_cleaned['Mileage'] <= upper)], bins=50)
plt.xlabel('Mileage')
plt.ylabel('Count')
plt.title('Distribution of Car Mileage (0.005-99.5th Percentile)')
plt.tight_layout()
plt.show()


# In[282]:


plt.figure(figsize=(8,4))
plt.hist(np.log1p(df_cleaned['Mileage']), bins=50)
plt.xlabel('log(Mileage + 1)')
plt.ylabel('Count')
plt.title('Log-Transformed Distribution of Car Mileage')
plt.show()


# In[283]:


# another check for our features and missing values:
print("Dataset shape: ", df_cleaned.shape)
print("Columns and Types:")
print(df_cleaned.dtypes)
print(" ")
print("Missing Values:")
df_cleaned.isnull().sum()


# We can still see some missing values for the Levy. Levy generally refers to a tax or fee imposed by the government based on characteristics of the vehicle. Levy is influenced by the cars features like engine volume, age, vehicle type etc. We will impute missing data for levy based on engine volume and category - we will find the median value for engine volume and category and fill the missing levy based on this value.

# In[284]:


group_medians = df_cleaned.groupby(['Engine volume', 'Category'])['Levy'].median()
group_medians.isnull().sum()


# In[285]:


#Since there are some in group_median that are NaN, 
# we will create another group, a more general group_medians, this time by engine volume only:
engine_volume_medians = df_cleaned.groupby('Engine volume')['Levy'].median()
print("Number of engine_volume_medians NaNs: ", engine_volume_medians.isnull().sum())

# There are still NaNs, hence we will create a global median for the rest
global_median_levy = df_cleaned['Levy'].median()


# In[286]:


def impute_levy(row):
    if pd.isnull(row['Levy']):
        key = (row['Engine volume'], row['Category']) # first group_median
        levy_value = group_medians.get(key, np.nan)

        if pd.isnull(levy_value):  # If group_median not found, use engine_volume_medians
            levy_value = engine_volume_medians.get(row['Engine volume'], np.nan)

        if pd.isnull(levy_value):  # If not found, use global_median_levy
            levy_value = global_median_levy

        return levy_value
    return row['Levy']


# In[287]:


# applying the imputation
# we will use the "regular" imputation for df_baseline, 
# because maybe later we will use different approach for imputation
df_cleaned['Levy'] = df_cleaned.apply(impute_levy, axis=1)


# In[288]:


# Checking again for missing values after the data cleansing
df_cleaned.isnull().sum()


# Now it looks like we are done with the missing values. We will proceed with outliers clean up:

# In[289]:


columns_to_check = ['Levy', 'Engine volume', 'Airbags', 'Cylinders', 'Age']

for col in columns_to_check:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
    total = df_cleaned[col].shape[0]

    print(f"{col}: {outliers} outliers ({round(100 * outliers / total, 2)}%) — bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")


# It doesn't look like much outliers. We can remove them, except Cylinders, that we will clean a certain and logical range:

# In[290]:


df_cleaned = df_cleaned[df_cleaned['Levy'] <= 1584]
df_cleaned = df_cleaned[df_cleaned['Engine volume'] <= 3.8]
df_cleaned = df_cleaned[(df_cleaned['Cylinders'] >= 2) & (df_cleaned['Cylinders'] <= 12)]
df_cleaned = df_cleaned[(df_cleaned['Age'] >= 2) & (df_cleaned['Age'] <= 26)]


# In[291]:


df_baseline = df_cleaned.copy()


# We still have some features remained to encode:
# 
# * **Manufacturer** - Since we believe that there is a strong correlation between the brand and the price, we will encode this by ranking the average price per manufacturer. This captures the average value impact of each manufacturer on price, which makes intuitive sense (e.g. Mercedes vs. Hyundai). It will also will help us to reduce cardinality to a single numeric column, unlike one-hot encoding. However, we're familiar with this problematic feature transformation since we are encoding the target variable. Instead of assigning each manufacturer a unique rank, we will group them into price-based ranges that will hopefully prevent over-fitting and reduces sensitivity to small price differences between manufacturers.
# * **Category** - We will check whether to use dummy variables or ordinal values.
# * **Gear-Box Type** - We will check whether to use dummy variables or ordinal values.
# * **Drive Wheels** - We will check whether to use dummy variables or ordinal values.

# In[292]:


# before proceeding to the last 3 variables, we'll check if there is some correlation between the price
# and those variable. If there is, we would prefer to transform them into a ordinal variable
# rather than one-hot encoding

avg_price_by_category = df_baseline.groupby('Category')['Price'].mean().sort_values(ascending=False)
avg_price_by_gear = df_baseline.groupby('Gear box type')['Price'].mean().sort_values(ascending=False)
avg_price_by_Drive_wheels = df_baseline.groupby('Drive wheels')['Price'].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

sns.barplot(x='Category', y='Price', data=df_baseline, color='steelblue', order=avg_price_by_category.index, ax=axes[0])
axes[0].set_title('Avg Price by Category')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x='Drive wheels', y='Price', data=df_baseline, color='steelblue', order=avg_price_by_Drive_wheels.index, ax=axes[1])
axes[1].set_title('Avg Price by Drive wheels')
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(x='Gear box type', y='Price', data=df_baseline, color='steelblue', order=avg_price_by_gear.index, ax=axes[2])
axes[2].set_title('Avg Price by Gear box type')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
fig.savefig(r'images\categorical_vars.png', bbox_inches='tight')
plt.show()
plt.close()


# The three bar plots above show the average car price for each category within the variables Category, Gear box type, and Drive wheels. These plots reveal the patterns in how these features relate to the target variable (Price):
# 
# * Category: There is a clear ranking among vehicle types. High-priced categories like Pickup, Cabriolet, and jeep, consistently outperform lower-end types like Goods wagon and Hatchback.
# 
# * Gear box type: Average prices decrease consistently from Tiptronic to Manual, indicating a strong ordinal relationship. This also corresponds to common knowledge that e.g. automatic car is more expenssive than manual car.
# 
# * Drive wheels: Since differences are smaller, we will use one-hot encoding.
# 
# Given these ordered relationships with price, it's appropriate to encode these Category and Gear box type variables as ordinal values, rather than one-hot encoding. For Drive wheels we will use one-hot encoding since the price differences are not that significant. This approach reduces dimensionality while preserving the underlying value hierarchy these features represent.

# In[293]:


# Rank encoding for Category
avg_price_by_category = df_baseline.groupby('Category')['Price'].mean().sort_values()
category_rank = avg_price_by_category.rank(method='first').astype(int)
df_baseline['Category_encoded'] = df_baseline['Category'].map(category_rank)

# Rank encoding for Gear box type
avg_price_by_gear = df_baseline.groupby('Gear box type')['Price'].mean().sort_values()
gearbox_rank = avg_price_by_gear.rank(method='first').astype(int)
df_baseline['GearBox_encoded'] = df_baseline['Gear box type'].map(gearbox_rank)

# One-hot encoding for Drive wheels
drive_dummies = pd.get_dummies(df_baseline['Drive wheels'], prefix='Drive')
df_baseline = pd.concat([df_baseline, drive_dummies], axis=1)


# We will check correlations again, now that the data is fully ready:

# In[294]:


numeric_cols = df_baseline.select_dtypes(include=['float64', 'int64', 'int32','int64']).columns
features_to_plot = [col for col in numeric_cols if col != 'Price']

n_cols = 3
n_rows = -(-len(features_to_plot) // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(features_to_plot):
    ax = axes[i]
    ax.scatter(df_baseline[col], df_baseline['Price'], alpha=0.5)
    ax.set_xlabel(col)
    ax.set_ylabel('Price')
    ax.set_title(f'Price vs {col}')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.suptitle('Scatter Plots: Price vs Other Features', fontsize=16, y=1.02)
fig.savefig(r'images\correlation_plots.png', bbox_inches='tight')
plt.show()
plt.close()


# In[295]:


numeric_data = df_baseline.select_dtypes(include=[np.number])

corr = numeric_data.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

fig = plt.figure(figsize=(12, 10))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    square=True,
    annot_kws={"size": 10},)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title("Correlation Matrix of Numeric Features", fontsize=16)
plt.tight_layout()
fig.savefig(r'images\correlation_heat_plot.png', bbox_inches='tight')
plt.show()
plt.close()


# #### Key insights:
# 
# * Price is negatively correlated with Age (-0.42). As expected, older cars tend to be significantly cheaper.
# 
# * Manufacturer and Category encodings show moderate correlation with Price. Manufacturer_encoded (-0.38) and Category_encoded (0.34) suggest your rank-based encoding reflects meaningful pricing tiers.
# 
# * Engine volume and Cylinders are strongly correlated (0.64). This is logical, as bigger engines usually have more cylinders.

# In[296]:


# Define features and target
X = df_baseline.drop(columns=[
    'Price',                # target
    'Manufacturer',         # original categorical
    'Category',
    'Gear box type',
    'Drive wheels'          # now one-hot encoded
])
y = df_baseline.loc[X.index, 'Price']  # Align y with X

# Compute mutual information
mi_scores = mutual_info_regression(X, y, discrete_features='auto')
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
# Plot
plt.figure(figsize=(12, 7))
sns.barplot(x=mi_series.values, y=mi_series.index)
plt.title("Feature Importance (Mutual Information with Price)", fontsize=14)
plt.xlabel("MI Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# #### Key Insights
# 
# The above chart shows feature importance using a method called Mutual Information (MI). It's a way of measuring how much each feature in the dataset tells us about the target variable. Mutual Information measures how much knowing the value of one variable reduces uncertainty about another (e.g. mileage and price). It captures any kind of relationship, linear or nonlinear, unlike correlation which only captures linear relationships.
# 
# In our case, it looks like mileage, Levy, airbags, and age are prominent in predicting the price, as they got a higer MI score. i.e. newer cars with more airbags and bigger engines tend to be pricier. Manufacturer_encoded, and Category_encoded matter, but to a lesser degree.

# ### Data Preprocessing Summary
# 
# Before building a regression model, several steps were taken to clean and prepare the data:
# 
# * Dropped unnecessary columns: Features like ID, Color, Model, and Steering Wheel Side were removed because they either had too many unique values, were unlikely to influence price directly, or were redundant.
# 
# * Fixed data formatting issues: Some columns had text mixed with numbers (e.g. "km" in mileage or "Turbo" in engine volume). These were cleaned so that only numerical values remained. For instance, we also created a new column to indicate whether a car had a turbo engine.
# 
# * Converted categorical variables: Categorical features (like Leather Interior) were converted to numerical form to work with machine learning models.
# 
# * Handled missing and invalid values: Placeholder values like "-" were replaced with actual missing values (NaN) and converted to proper numerical types.
# 
# * Created a new feature for car age: Instead of using the production year directly, we calculated the car's age by subtracting the production year from the current year.
# 
# * Removed outliers: Extremely high values for price and mileage were removed because they could distort the results of the model. The cutoffs were based on distribution analysis using boxplots and interquartile range (IQR).
# 
# * Applied log transformation: Car prices and mileage were both highly skewed. Applying a logarithmic transformation helps to reduce skewness and may improve model performance by making the data more normally distributed.
# 
# * Handling Levy missing values - we created a few base groups that Levy is influenced by (engine volume, and category, engine volume, and general) and applied imputation using the median of those groups.
# 
# * Handling manufacturer, Category, and Gear box type: we turned those variables into ordinal variables based on their pattern with the price. The pattern also corresponded with believes and knowledge - for instance, a manual car is cheaper than automatic car (for gear box). for Drive wheels we decided to keep the one-hot encoding due to the insignificant prices between the 3. For the manufacturer feature, we believe that there is a strong correlation between the brand and the price. Hence, we groupped the average price by manufacturer, this captures the average value impact of each manufacturer on price, which makes intuitive sense (e.g. Mercedes vs. Hyundai). While keeping in mind this problematic feature transformation we splitted the brands into bins based on average price ranges to prevent over-fitting.

# Now that we are done analyzing and processing the data, we can save it to a .csv file.

# In[297]:


#Save the df_baseline to a .csv file
df_baseline.to_csv("csv_files\df_baseline.csv", index=False)


# # Baseline Model: Ridge Regression

# Now that the data is processed, and analyzed, we can introduce the first model for this project, Ridge regression. This model will serve as a baseline reference for the next models that we will introduce.

# In[298]:


# we drop redundant original categorical columns + 1 dummy to avoid multicollinearity
cols_to_drop = ['Price', 'Category', 'Gear box type', 'Drive wheels', 'Drive_Rear']
target = np.log1p(df_baseline['Price'])

X = df_baseline.drop(columns=cols_to_drop)
y = target

#we split first before manufacturer encoding
X_train, X_test, y_train, y_test = train_test_split(X.copy(), y, test_size=0.2, random_state=42)

# we encode Manufacturer based only on training data
manufacturer_avg_price_train = X_train.copy()
manufacturer_avg_price_train['Price'] = np.expm1(y_train) 

avg_price_by_manufacturer = manufacturer_avg_price_train.groupby('Manufacturer')['Price'].mean()
bins = [2000, 10000, 15000, 20000, 25000, 32000, float('inf')]
labels = [1, 2, 3, 4, 5, 6]
manufacturer_price_bins = pd.cut(avg_price_by_manufacturer, bins=bins, labels=labels)

# Apply encoding to both train and test
X_train['Manufacturer_encoded'] = X_train['Manufacturer'].map(manufacturer_price_bins)
X_test['Manufacturer_encoded'] = X_test['Manufacturer'].map(manufacturer_price_bins)

# Handle unknown manufacturers in test set - in case a manufacturer exists in the test but not in the train
median_bin = X_train['Manufacturer_encoded'].astype(int).median()
X_test['Manufacturer_encoded'].fillna(median_bin, inplace=True)


# In[299]:


# Drop original 'Manufacturer' after encoding
X_train.drop(columns=['Manufacturer'], inplace=True)
X_test.drop(columns=['Manufacturer'], inplace=True)

# Standardize numeric features (excluding encoded/binary/one-hot)
ordinal_cols = ['Manufacturer_encoded', 'Category_encoded', 'GearBox_encoded']
one_hot_cols = ['Drive_4x4', 'Drive_Front']
binary_cols = ['Leather interior', 'Turbo']
to_scale = [col for col in X_train.columns if col not in ordinal_cols + one_hot_cols + binary_cols]

scaler = StandardScaler()
X_train[to_scale] = scaler.fit_transform(X_train[to_scale])
X_test[to_scale] = scaler.transform(X_test[to_scale])

print("X_train shape:", X_train.shape)


# In[300]:


# Initialize and fit Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict
y_pred_train = ridge_model.predict(X_train)
y_pred_test = ridge_model.predict(X_test)

# Inverse-transform predictions to original price scale
y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)
y_pred_train_actual = np.expm1(y_pred_train)
y_pred_test_actual = np.expm1(y_pred_test)


# In[301]:


# we will define an evaluation function
def evaluate(y_true, y_pred, label):
    print(f"{label}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))
    print()


# In[302]:


# and show the model evaluation
evaluate(y_train_actual, y_pred_train_actual, "Training Set")
evaluate(y_test_actual, y_pred_test_actual, "Test Set")


# In[303]:


# Print intercept
print(f"Intercept: {ridge_model.intercept_:.4f}")

# Print coefficients with feature names
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': ridge_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop Coefficients:")
print(coef_df.to_string(index=False))


# In[304]:


# Lastly we plot predictions vs true values
fig = plt.figure(figsize=(7, 6))
plt.scatter(y_test_actual, y_pred_test_actual, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Ridge Regression: Predicted vs True (Test Set)")
plt.tight_layout()
fig.savefig(r'images\true_vs_pred_Ridge_baseline.png', bbox_inches='tight')
plt.show()
plt.close()


# #### Key insights
# 
# * The model achieves an R_2 of =~ 0.34, meaning it explains about 34% of the variation in car prices.
# * The train and test R_2 scores are nearly identical, suggesting the model is not overfitting. It generalizes reasonably well to unseen data.
# 
# * With a mean absolute error (MAE) of =~6,500, the average prediction is off by that amount. Given that the dataset includes car prices ranging from ~2,000 to 50,000.
# 
# * Underestimation of Higher Prices: The scatterplot shows the model tends to underpredict expensive cars. We assume that the linear model can’t capture complex, nonlinear relationships unless explicitly modeled.

# We will try to create another model that considers the top 6 features from the MI chart:

# In[305]:


# Select top MI features from already-prepared X_train and X_test
top_features = ['Levy', 'Mileage', 'Airbags', 'Age', 'Engine volume', 'Manufacturer_encoded']

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
y_train_top = y_train
y_test_top = y_test

# Train model
ridge_top = Ridge(alpha=1.0)
ridge_top.fit(X_train_top, y_train_top)

# Predict and evaluate
y_pred_top = ridge_top.predict(X_test_top)
y_test_actual = np.expm1(y_test_top)
y_pred_actual = np.expm1(y_pred_top)

print("MAE:", mean_absolute_error(y_test_actual, y_pred_actual))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)))
print("R²:", r2_score(y_test_actual, y_pred_actual))


# We will try to create another model that considers the top 6 correlated (in absolut values) features from heatmap chart:

# In[306]:


correlated_features = ['Age', 'Manufacturer_encoded', 'Category_encoded',
    'Leather interior', 'GearBox_encoded', 'Engine volume']

X_train_corr = X_train[correlated_features].copy()
X_test_corr = X_test[correlated_features].copy()
y_train_corr = y_train
y_test_corr = y_test

features_to_scale_corr = ['Age', 'Engine volume']
scaler_corr = StandardScaler()
X_train_corr[features_to_scale_corr] = scaler_corr.fit_transform(X_train_corr[features_to_scale_corr])
X_test_corr[features_to_scale_corr] = scaler_corr.transform(X_test_corr[features_to_scale_corr])

ridge_corr = Ridge(alpha=1.0)
ridge_corr.fit(X_train_corr, y_train_corr)

y_pred_corr = ridge_corr.predict(X_test_corr)
y_test_actual_corr = np.expm1(y_test_corr)
y_pred_actual_corr = np.expm1(y_pred_corr)

print("MAE:", mean_absolute_error(y_test_actual_corr, y_pred_actual_corr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual_corr, y_pred_actual_corr)))
print("R²:", r2_score(y_test_actual_corr, y_pred_actual_corr))


# ## Bayesian models
# 
# For this part, we will use the same dataset as before, but we will try to apply a hierarchical Bayesian model to better describe the data. It was noticed that the parameters in the dataset fit with three key factors that could be relevent to explain the choice of the consumer and price of the car. The Performance of the car, its Seniority and the Status given by the car. Though we organised our different parameters to describe each factor.

# In[307]:


img = mpimg.imread('MBML_PGM.png')
plt.imshow(img)
plt.axis('off')
plt.show()



# In[308]:


df_pgm1 = pd.read_csv("csv_files\df_baseline.csv")


# Data Preparation : Drop unused parameters and scale some columns

# In[309]:


columns_to_scale =['Mileage']
scaler = StandardScaler()
X_scaled = df_pgm1.copy()
X_scaled[columns_to_scale] = scaler.fit_transform(X_scaled[columns_to_scale])

#In this part the unit of the target variable, Price, is in thousands of euros in order to help the model during th training party = df_pgm1['Price']
y = df_pgm1['Price']
y_scaled = y/1000

X_scaled['Levy'] = df_pgm1['Levy']/1000


# In[310]:


columns_quality = ['Turbo', 'Engine volume', 'Cylinders']
columns_seniority = ['Age', 'Mileage']
columns_performance = ['Category_encoded', 'GearBox_encoded']
columns_status = ['Leather interior','Category_encoded','Manufacturer_encoded']
column_levy =['Levy']


# Get train and test set

# In[311]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)


# Encode data

# In[312]:


# Compute mean price per manufacturer using only training data
avg_price_by_manufacturer = X_train.groupby('Manufacturer')['Price'].mean()

# Bin the average prices into ordinal categories
bins = [2000, 10000, 15000, 20000, 25000, 32000, float('inf')]
labels = [1, 2, 3, 4, 5, 6]
manufacturer_price_bins = pd.cut(avg_price_by_manufacturer, bins=bins, labels=labels)

# Map the encoding to train and test separately
X_train['Manufacturer_encoded'] = X_train['Manufacturer'].map(manufacturer_price_bins)
X_test['Manufacturer_encoded'] = X_test['Manufacturer'].map(manufacturer_price_bins)

# Fill missing values in test set with the median bin from training set
median_bin = X_train['Manufacturer_encoded'].astype(float).median()
X_test['Manufacturer_encoded'].fillna(median_bin, inplace=True)


# In[313]:


X_train_quality = X_train[columns_quality]
X_train_seniority = X_train[columns_seniority]
X_train_performance = X_train[columns_performance]
X_train_status = X_train[columns_status]
X_train_levy = X_train[column_levy]


X_test_quality = X_test[columns_quality]
X_test_seniority = X_test[columns_seniority]
X_test_performance = X_test[columns_performance]
X_test_status = X_test[columns_status]
X_test_levy = X_test[column_levy]


X_train_quality = torch.tensor(X_train_quality.values, dtype=torch.float32)
X_train_performance = torch.tensor(X_train_performance.values, dtype=torch.float32)
X_train_status = torch.tensor(X_train_status.values, dtype=torch.float32)
X_train_seniority = torch.tensor(X_train_seniority.values, dtype=torch.float32)
X_train_levy = torch.tensor(X_train_levy.values, dtype=torch.float32)


X_test_quality = torch.tensor(X_test_quality.values, dtype=torch.float32)
X_test_performance = torch.tensor(X_test_performance.values, dtype=torch.float32)
X_test_status = torch.tensor(X_test_status.values, dtype=torch.float32)
X_test_seniority = torch.tensor(X_test_seniority.values, dtype=torch.float32)
X_test_levy = torch.tensor(X_test_levy.values, dtype=torch.float32)


y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


# Define model

# In[314]:


def model2(X_quality, X_performance, X_status, X_seniority,X_levy, y=None):
    N = X_quality.shape[0]

    # Quality latent parameters
    D_q = X_quality.shape[1]
    w_q = pyro.sample("w_quality", dist.Normal(torch.zeros(D_q), torch.ones(D_q)).to_event(1)).squeeze()
    b_q = pyro.sample("b_quality", dist.Normal(0., 1.))

    # Performance latent parameters
    D_p = X_performance.shape[1]
    w_p_x = pyro.sample("w_performance_x", dist.Normal(torch.zeros(D_p), torch.ones(D_p)).to_event(1)).squeeze()
    w_p_q = pyro.sample("w_performance_q", dist.Normal(0., 1.))
    b_p = pyro.sample("b_performance", dist.Normal(0., 1.))

    # Status latent parameters
    D_s = X_status.shape[1]
    w_s = pyro.sample("w_status", dist.Normal(torch.zeros(D_s), torch.ones(D_s)).to_event(1)).squeeze()
    b_s = pyro.sample("b_status", dist.Normal(0., 1.))

    # Seniority latent parameters
    D_se = X_seniority.shape[1]
    w_se = pyro.sample("w_seniority", dist.Normal(torch.zeros(D_se), torch.ones(D_se)).to_event(1)).squeeze()
    b_se = pyro.sample("b_seniority", dist.Normal(0., 1.))

    # Levy parameter
    D_l = X_levy.shape[1]
    w_levy = pyro.sample("w_levy", dist.Normal(torch.zeros(D_l), torch.ones(D_l)).to_event(1))
    b_levy = pyro.sample("b_levy", dist.Normal(0., 1.))

    # Global weights
    w_perf = pyro.sample("w_perf", dist.Normal(0., 1.))
    w_stat = pyro.sample("w_stat", dist.Normal(0., 1.))
    w_sen = pyro.sample("w_sen", dist.Normal(0., 1.))

    b_global = pyro.sample("bias_global", dist.Normal(0., 1.))

    #sigma = pyro.sample("sigma", dist.LogNormal(-1.0, 0.5))
    sigma = 2

    with pyro.plate("data", N):
        # Quality
        mu_q = torch.matmul(X_quality, w_q) + b_q.expand(N)
        quality_latent = pyro.sample("quality_latent", dist.Normal(mu_q, 1.0))

        # Performance
        mu_p = torch.matmul(X_performance, w_p_x) + w_p_q * quality_latent + b_p.expand(N)
        performance_latent = pyro.sample("performance_latent", dist.Normal(mu_p, 1.0))

        # Status
        mu_s = torch.matmul(X_status, w_s) + b_s.expand(N)
        status_latent = pyro.sample("status_latent", dist.Normal(mu_s, 1.0))

        # Seniority
        mu_se = torch.matmul(X_seniority, w_se) + b_se.expand(N)
        seniority_latent = pyro.sample("seniority_latent", dist.Normal(mu_se, 1.0))

        # Levy
        mu_levy = torch.matmul(X_levy, w_levy) + b_levy.expand(N)
        levy_latent = pyro.sample("levy_latent", dist.Normal(mu_levy, 1.0))

        # Global model
        mu_global_raw = (
            w_perf * performance_latent +
            w_stat * status_latent +
            w_sen * seniority_latent +
            levy_latent +
            b_global.expand(N)
        )
        mu_global = torch.nn.functional.softplus(mu_global_raw) + 1.0


        obs = pyro.sample("obs", dist.Normal(mu_global, sigma), obs=y)

    return obs


# In[315]:


def guide2(X_quality, X_performance, X_status, X_seniority, X_levy, y=None):
    N = X_quality.shape[0]

    # Quality
    D_q = X_quality.shape[1]
    w_q_loc = pyro.param("w_q_loc", torch.zeros(D_q))
    w_q_scale = pyro.param("w_q_scale", torch.ones(D_q), constraint=constraints.positive)
    pyro.sample("w_quality", dist.Normal(w_q_loc, w_q_scale).to_event(1))

    b_q_loc = pyro.param("b_q_loc", torch.tensor(0.))
    b_q_scale = pyro.param("b_q_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("b_quality", dist.Normal(b_q_loc, b_q_scale))

    # Performance
    D_p = X_performance.shape[1]
    w_p_x_loc = pyro.param("w_p_x_loc", torch.zeros(D_p))
    w_p_x_scale = pyro.param("w_p_x_scale", torch.ones(D_p), constraint=constraints.positive)
    pyro.sample("w_performance_x", dist.Normal(w_p_x_loc, w_p_x_scale).to_event(1))

    w_p_q_loc = pyro.param("w_p_q_loc", torch.tensor(0.))
    w_p_q_scale = pyro.param("w_p_q_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("w_performance_q", dist.Normal(w_p_q_loc, w_p_q_scale))

    b_p_loc = pyro.param("b_p_loc", torch.tensor(0.))
    b_p_scale = pyro.param("b_p_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("b_performance", dist.Normal(b_p_loc, b_p_scale))

    # Status
    D_s = X_status.shape[1]
    w_s_loc = pyro.param("w_s_loc", torch.zeros(D_s))
    w_s_scale = pyro.param("w_s_scale", torch.ones(D_s), constraint=constraints.positive)
    pyro.sample("w_status", dist.Normal(w_s_loc, w_s_scale).to_event(1))

    b_s_loc = pyro.param("b_s_loc", torch.tensor(0.))
    b_s_scale = pyro.param("b_s_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("b_status", dist.Normal(b_s_loc, b_s_scale))

    # Seniority
    D_se = X_seniority.shape[1]
    w_se_loc = pyro.param("w_se_loc", torch.zeros(D_se))
    w_se_scale = pyro.param("w_se_scale", torch.ones(D_se), constraint=constraints.positive)
    pyro.sample("w_seniority", dist.Normal(w_se_loc, w_se_scale).to_event(1))

    b_se_loc = pyro.param("b_se_loc", torch.tensor(0.))
    b_se_scale = pyro.param("b_se_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("b_seniority", dist.Normal(b_se_loc, b_se_scale))

    # Levy (nouvelle partie)
    D_l = X_levy.shape[1]
    w_levy_loc = pyro.param("w_levy_loc", torch.zeros(D_l))
    w_levy_scale = pyro.param("w_levy_scale", torch.ones(D_l), constraint=constraints.positive)
    pyro.sample("w_levy", dist.Normal(w_levy_loc, w_levy_scale).to_event(1))

    b_levy_loc = pyro.param("b_levy_loc", torch.tensor(0.))
    b_levy_scale = pyro.param("b_levy_scale", torch.tensor(1.), constraint=constraints.positive)
    pyro.sample("b_levy", dist.Normal(b_levy_loc, b_levy_scale))

    # Global weights
    for name in ["w_perf", "w_stat", "w_sen", "bias_global"]:
        loc = pyro.param(f"{name}_loc", torch.tensor(0.))
        scale = pyro.param(f"{name}_scale", torch.tensor(1.), constraint=constraints.positive)
        pyro.sample(name, dist.Normal(loc, scale))

    # Observation noise
    sigma_loc = pyro.param("sigma_loc", torch.tensor(-1.0))
    sigma_scale = pyro.param("sigma_scale", torch.tensor(0.5), constraint=constraints.positive)
    #pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))

    # Latent variables
    with pyro.plate("data", N):
        quality_loc = torch.matmul(X_quality, w_q_loc) + b_q_loc
        pyro.sample("quality_latent", dist.Normal(quality_loc, 1.0))

        performance_loc = torch.matmul(X_performance, w_p_x_loc) + w_p_q_loc * quality_loc + b_p_loc
        pyro.sample("performance_latent", dist.Normal(performance_loc, 1.0))

        status_loc = torch.matmul(X_status, w_s_loc) + b_s_loc
        pyro.sample("status_latent", dist.Normal(status_loc, 1.0))

        seniority_loc = torch.matmul(X_seniority, w_se_loc) + b_se_loc
        pyro.sample("seniority_latent", dist.Normal(seniority_loc, 1.0))

        levy_loc = torch.matmul(X_levy, w_levy_loc) + b_levy_loc
        pyro.sample("levy_latent", dist.Normal(levy_loc, 1.0))


# Train the model 

# In[316]:


train_elbos = []
test_elbos = []

pyro.clear_param_store()

n_steps = 5000

# Setup the optimizer
adam_params = {"lr": 0.01}
optimizer = Adam(adam_params)

# Setup the inference algorithm
elbo = Trace_ELBO(num_particles=3)
svi = SVI(model2, guide2, optimizer, loss=elbo)

# Do gradient steps
for step in range(n_steps):
    elbo = svi.step(X_train_quality, X_train_performance, X_train_status, X_train_seniority,X_train_levy, y_train_tensor)
    train_elbos.append(elbo)

    if step % 100 == 0:
        print("[%d] ELBO: %.1f" % (step, elbo))


# Plot the ELBO 

# In[317]:


plt.plot(train_elbos, label="Train ELBO")
plt.xlabel("Steps")
plt.ylabel("ELBO Loss")
plt.legend()
plt.title("ELBO during training")
plt.show()


# Compute learned weights 

# In[318]:


# Get the learned parameters from the param store
param_store = pyro.get_param_store()

weights_quality_features = param_store["w_q_loc"].detach().cpu().numpy()
weights_status_features = param_store["w_s_loc"].detach().cpu().numpy()
weights_seniority = param_store["w_se_loc"].detach().cpu().numpy()
weights_performance_features = param_store["w_p_x_loc"].detach().cpu().numpy()
weights_performance_q = param_store["w_p_q_loc"].item()  # scalaire

print("Mean of the weights learned for each feature:")
for name, w in zip(columns_quality, weights_quality_features):
    print(f"{name:20s} : {w:.4f}")
for name, w in zip(columns_performance, weights_performance_features):
    print(f"{name:20s} : {w:.4f}")
print("Motor Quality :", weights_performance_q)
for name, w in zip(columns_status, weights_status_features):
    print(f"{name:20s} : {w:.4f}")
for name, w in zip(columns_seniority, weights_seniority):
    print(f"{name:20s} : {w:.4f}")

print("Status: ",param_store["w_stat_loc"].item())
print("Performance: ",param_store["w_perf_loc"].item())
print("Seniority: ",param_store["w_sen_loc"].item())
print("Levy: ",param_store["w_levy_loc"].item())


# # Gaussian Process with ARD
# This section implements a Variational Sparse Gaussian Process (GP) with Automatic Relevance Determination (ARD) to predict used car prices, using an RBF kernel and sparse inducing points. We train the model, evaluate on a test set, visualize predictions, and analyze feature relevance via ARD lengthscales.

# In[319]:


# Additional imports for Gaussian Process
plt.close('all')
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

# Clear Pyro parameter store to avoid conflicts
pyro.clear_param_store()

# Load data
df = pd.read_csv(r'csv_files\df_baseline.csv')


# Data Preparation (Train-Test Split, Standardization)

# In[320]:


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


# Gaussian Process Model Setup

# In[321]:


#Convert data to tensors
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


# Training the GP

# In[322]:


# Train model
losses = gp.util.train(vsgp, num_steps=1500)


# ELBO loss

# In[323]:


fig = plt.figure(figsize=(7, 6))
plt.plot(losses)
plt.title("ELBO Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
fig.savefig(r'images\GP\ELBO.png', bbox_inches='tight')
plt.show()
plt.close()


# Predictions

# In[324]:


# Predict on TEST set
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


# Plot of prediction vs true price

# In[325]:


fig = plt.figure(figsize=(7, 6))
plt.errorbar(true_price, pred_price, 
             yerr=[(pred_price - pred_price_lower), 
                   (pred_price_upper - pred_price)], 
             fmt='o', alpha=0.4, label="Predicted ±2σ")
plt.plot([true_price.min(), true_price.max()], 
         [true_price.min(), true_price.max()], 
         'k--', label="Perfect Prediction")
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("GP Predicted Price vs True Price (Test Set)")
plt.legend()
plt.grid(True)
fig.savefig(r'images\GP\GP_pred_vs_true.png', bbox_inches='tight')
plt.show()
plt.close()


# Residuals vs True Price

# In[326]:


fig = plt.figure(figsize=(7, 6))
residuals = true_price - pred_price
plt.scatter(true_price, residuals, alpha=0.25)
plt.axhline(0, ls='--', lw=1)
plt.xlabel("True price")
plt.ylabel("Residual (True − Pred)")
plt.title("Residuals vs. True Price (Test Set)")
plt.grid(True)
fig.savefig(r'images\GP\residuals_GP.png', bbox_inches='tight')
plt.show()
plt.close()


# In[327]:


fig = plt.figure(figsize=(7, 6))
mae = mean_absolute_error(true_price, pred_price)
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
fig.savefig(r'images\GP\MAE.png', bbox_inches='tight')
plt.show()
plt.close()


# MAE,RMSE and R-Squared

# In[328]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)

print("MAE:",mae)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


# ARD Feature Relevance Analysis

# In[329]:


lengthscales = kernel.lengthscale.detach().cpu().numpy().flatten()
relevance = 1 / lengthscales
print("\nFeature relevance via ARD")
for feat, score in sorted(zip(feature_cols, relevance), key=lambda x: -x[1]):
    print(f"{feat:20s}  relevance: {score:.4f}")


# Gaussian Posterior Mean (1 dimensional analysis where Mileage is kept and all other input features are assigned to their respective mean)

# In[330]:


def plot(model, feature_index=2, X_train_tensor=None, y_train_tensor=None, n_test=500, scaler=None, num_points=200):
    """
    Plot GP posterior mean and uncertainty along a single feature index,
    keeping other features fixed at their mean. Shows actual price.
    Limits training data plot to num_points for clarity.
    """
    assert X_train_tensor is not None and y_train_tensor is not None and scaler is not None

    # Create test inputs: all features = mean
    X_base = X_train_tensor.mean(dim=0).repeat(n_test, 1)

    # Vary only one feature
    x_range = torch.linspace(-4, 4, n_test)
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
    fig = plt.figure(figsize=(10, 6))
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
    plt.ylabel("Predicted Price ")
    plt.title(f"SE kernel")
    plt.legend()
    plt.grid(True)
    fig.savefig(r'images\GP\Gaussian_Posterior_Mean.png', bbox_inches='tight')
    plt.show()

plot(model=vsgp,
     feature_index=2,
     X_train_tensor=X_train_tensor,
     y_train_tensor=y_train_tensor,
     scaler=scaler,
     num_points=500)


# Now that we have made the Gaussian proces, we want to see if we can improve the results by adding an neural network that learns relationships about the 
# input features before they are used for training and testing.

# Setup the model

# In[331]:


import torch
import torch.nn as nn
import torch.nn.functional as F
pyro.clear_param_store()

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

pyro.clear_param_store()

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




# Train the model

# In[332]:


fig = plt.figure(figsize=(7, 6))
# Train
losses = gp.util.train(vsgp, num_steps=1500)
plt.plot(losses)
plt.title("ELBO Loss (Deep Kernel GP)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
fig.savefig(r'images\GP\ELBO_Deep_Kernel_GP.png', bbox_inches='tight')
plt.show()
plt.close()


# Plot true price vs. predicted price with uncertainty bounds

# In[333]:


fig = plt.figure(figsize=(7, 6))
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
plt.title("Deep Kernel GP: Predicted vs True Price (Test Set)")
plt.legend()
plt.grid(True)
fig.savefig(r'images\GP\true_vs_pred_price_with_uncertainty.png', bbox_inches='tight')
plt.show()


# In[334]:


# Scatter plot with MAE
fig = plt.figure(figsize=(7, 6))
mae = mean_absolute_error(true_price, pred_price)
plt.scatter(true_price, pred_price, alpha=0.3, label='Predictions')
plt.plot([true_price.min(), true_price.max()],
         [true_price.min(), true_price.max()],
         'k--', label='Perfect Prediction')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Deep Kernel GP Predicted Price vs True Price (Test Set)\nMAE = {:.2f}".format(mae))
plt.legend()
plt.grid(True)
plt.tight_layout()
fig.savefig(r'images\GP\true_vs_pred_price_MAE.png', bbox_inches='tight')
plt.show()
plt.close()


# Plot residuals

# In[335]:


fig = plt.figure(figsize=(7, 6))
residuals = true_price - pred_price
plt.scatter(true_price, residuals, alpha=0.25)
plt.axhline(0, ls='--', lw=1)
plt.xlabel("True price")
plt.ylabel("Residual (True − Pred)")
plt.title("Residuals vs. True Price")
plt.grid(True)
fig.savefig(r'images\GP\residuals_deep_GP.png', bbox_inches='tight')
plt.show()
plt.close()

mae = mean_absolute_error(true_price, pred_price)
rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)


# MAE, RMSE and R^2

# In[336]:


rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


# ARD relevance
