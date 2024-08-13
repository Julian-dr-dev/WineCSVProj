import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



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




X = df[['Malic_Acid']]
y = df[['Alcohol']]

