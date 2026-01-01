import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("pass_fail.csv")

# Encode Pass/Fail
le = LabelEncoder()
data['Result'] = le.fit_transform(data['Result'])

# Inputs and Output
X = data[['Hours', 'Attendance']]
y = data['Result']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict
prediction = model.predict([[7, 78]])

result = le.inverse_transform(prediction)
print("Prediction:", result[0])
