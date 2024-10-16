import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define column names
columns = [
    'Cultivar', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 
    'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 
    'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280/OD315', 'Proline'
]

# Load the data from the file
df = pd.read_csv('wine.data', header=None, names=columns)

# Display the first few rows
print(df.head())

# Pairplot to visualize relationships
sns.pairplot(df, hue='Cultivar', vars=['Alcohol', 'Malic_Acid', 'Ash', 'Total_Phenols'])
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Clustering - KMeans
X_clustering = df[['Alcohol', 'Malic_Acid', 'Total_Phenols', 'Color_Intensity']] # Select relevant features for clustering

scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

kmeans = KMeans(n_clusters=3, random_state=42)  # Assuming 3 clusters based on wine cultivars
df['Cluster'] = kmeans.fit_predict(X_clustering_scaled)

# Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Alcohol'], df['Malic_Acid'], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k')
plt.title('KMeans Clustering (Alcohol vs Malic Acid)')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.show()

# Classification - Logistic Regression
X_classification = df.drop(columns=['Cultivar', 'Cluster'])  # Drop 'Cultivar' and 'Cluster' for classification
y_classification = df['Cultivar']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_cls, y_train_cls)

y_pred_cls = logreg.predict(X_test_cls)

# Evaluate the classification model
accuracy = accuracy_score(y_test_cls, y_pred_cls)
conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)

print(f"Classification Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize Classification Results (Alcohol vs Malic Acid with True Cultivar and Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(X_test_cls['Alcohol'], X_test_cls['Malic_Acid'], c=y_pred_cls, cmap='viridis', marker='o', edgecolor='k', label='Predicted Cultivar')
plt.title('Logistic Regression Classification (Predicted Cultivar)')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.show()

# Linear Regression (Your existing code)
X = df[['Malic_Acid']]
y = df[['Alcohol']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Malic Acid')
plt.ylabel('Alcohol')
plt.title('Linear regression: malic acid vs alcohol')
plt.legend()
plt.show()
