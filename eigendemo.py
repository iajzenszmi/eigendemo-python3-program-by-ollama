import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Generate some sample data with different scales
np.random.seed(0)
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Standardization
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data)
pca_standard = PCA(n_components=2)
principal_components_standard = pca_standard.fit_transform(data_standardized)

print("Standardized Data:")
print(data_standardized)
print("Principal Components (Standardized):")
print(principal_components_standard)

# Min-max normalization
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data)
pca_minmax = PCA(n_components=2)
principal_components_minmax = pca_minmax.fit_transform(data_minmax)

print("Min-max Normalized Data:")
print(data_minmax)
print("Principal Components (Min-max):")
print(principal_components_minmax)

# No Scaling
pca_no_scaling = PCA(n_components=2)
principal_components_no_scaling = pca_no_scaling.fit_transform(data)

print("No Scaling Data:")
print(data)
print("Principal Components (No Scaling):")
print(principal_components_no_scaling)
