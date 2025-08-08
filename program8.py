'''
(8).Write a program to demonstrate the working of the decision tree. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample 

'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Plot tree
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree - Iris")
plt.show()

# Predict new sample
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(sample)
print("Prediction for", sample, "is", iris.target_names[pred[0]])