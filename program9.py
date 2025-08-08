#(9).Write a program to implement Naive Bayes algorithm to classify the iris data set. Print both correct and wrong predictions. 

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Print correct and wrong predictions
print("Predictions on Test Set:")
for i in range(len(y_test)):
    predicted = y_pred[i]
    actual = y_test[i]
    status = "✔ Correct" if predicted == actual else "✘ Wrong"
    print(f"Sample {i+1}: Predicted = {class_names[predicted]}, Actual = {class_names[actual]} --> {status}")