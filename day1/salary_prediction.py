import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("salary_data.csv")

# Input and Output
X = data[['Experience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict salary
exp = 5
predicted_salary = model.predict([[exp]])

print("Predicted Salary:", predicted_salary[0])

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.show()
