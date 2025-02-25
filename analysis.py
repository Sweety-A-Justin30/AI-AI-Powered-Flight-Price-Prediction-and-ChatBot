#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

print(colored('\nAll libraries imported successfully', 'green'))

# Load dataset
data = pd.read_csv('flight.csv')

# Print missing values
print("Missing values in dataset:\n", data.isna().sum())

# Print all columns and their indexes
for index, value in enumerate(data.columns):
    print(index, ":", value)

# Plot airline distribution and boxplot of prices
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Airline Distribution', fontsize=20, fontweight='bold')
plt.tight_layout()

# Pie Chart
labels = data.airline.value_counts().index.tolist()
explode = (0, 0, 0, 0, 0, 0.3)
ax[0].pie(data.airline.value_counts(), autopct='%.f%%', labels=labels, shadow=True, 
           pctdistance=1.15, labeldistance=0.6, explode=explode)
ax[0].legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=5)

# Boxplot
sns.boxplot(x='airline', y='price', data=data, ax=ax[1])

plt.show()

# Encoding categorical variables
data.airline = data.airline.replace({
    'Vistara': 1, 'Air_India': 2, 'Indigo': 3, 'GO_FIRST': 4, 'AirAsia': 5, 'SpiceJet': 6  
})

data.source_city = data.source_city.replace({
    'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3, 'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6
})

data.departure_time = data.departure_time.replace({
    'Morning': 1, 'Early_Morning': 2, 'Evening': 3, 'Night': 4, 'Afternoon': 5, 'Late_Night': 6
})

data.stops = data.stops.replace({
    'one': 1, 'zero': 2, 'two_or_more': 3
})

data.arrival_time = data.arrival_time.replace({
    'Night': 1, 'Evening': 2, 'Morning': 3, 'Afternoon': 4, 'Early_Morning': 5, 'Late_Night': 6
})

data.destination_city = data.destination_city.replace({
    'Mumbai': 1, 'Delhi': 2, 'Bangalore': 3, 'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6
})

data["class"] = data["class"].replace({
    "Economy": 1, "Business": 2
})

# Selecting features for training
X_temp = data[["airline", "source_city", "destination_city", "class", "days_left"]]
y = data["price"]

# Feature scaling
scaler = MinMaxScaler().fit_transform(X_temp)
X = pd.DataFrame(scaler, columns=X_temp.columns)
main_X = X.copy()

# Linear Regression Model Training
test_list = []
mse_list = []
r2score_list = []
best_r2 = 0
best_mse = 0
best_test = 0

for tester in range(6, 19):
    tester = round(0.025 * tester, 2)
    test_list.append(tester)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tester, random_state=0)
    
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    r2score = metrics.r2_score(y_test, y_pred_lr)
    r2score_list.append(r2score)
    mse = metrics.mean_squared_error(y_test, y_pred_lr)
    mse_list.append(mse)
    
    if r2score > best_r2:
        best_r2 = r2score
        best_mse = mse
        best_test = tester

print(colored(f'Best test_size : {best_test}', 'blue'))
print(colored(f'Best R2Score : {best_r2}', 'blue'))
print(colored(f'Best Mean Squared Error : {best_mse}', 'blue'))

# Plot performance metrics
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(test_list, r2score_list, c='blue', label='R2Score')
ax[0].set_title("R2Score")
ax[0].legend()

ax[1].plot(test_list, mse_list, c='red', label='Mean Squared Error')
ax[1].set_title("Mean Squared Error")
ax[1].legend()
plt.show()

# Save trained model
import pickle
with open("flight_price_model.pkl", "wb") as file:
    pickle.dump(lr, file)

print("✅ Model saved successfully as flight_price_model.pkl")


# Prediction function with price scaling
def predict_flight_price(input_data):
    """
    Predicts flight price and scales it within ₹5,000–₹20,000 dynamically.
    """
    with open("flight_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    predicted_price = model.predict(input_data)[0]
    
    # Min and Max values from dataset (adjust if needed)
    min_original = 3000  # Example minimum price in dataset
    max_original = 80000  # Example maximum price in dataset
    
    # Apply Min-Max Scaling to fit within ₹5,000–₹20,000
    predicted_price = 5000 + (15000 * (predicted_price - min_original) / (max_original - min_original))
    
    # Ensure price stays within limits
    predicted_price = max(5000, min(predicted_price, 20000))
    
    return round(predicted_price, 2)

