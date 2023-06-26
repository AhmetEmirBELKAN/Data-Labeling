import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


digits = load_digits()
X = digits.images
y = digits.target

X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


initial_indices = np.random.choice(X_train.shape[0], size=10, replace=False)
initial_X = X_train[initial_indices]
initial_y = y_train[initial_indices]


labeled_X = initial_X.copy()
labeled_y = initial_y.copy()
unlabeled_X = np.delete(X_train, initial_indices, axis=0)

max_show_examples = 10  
shown_examples = 0  

while len(unlabeled_X) > 0 and shown_examples < max_show_examples:
  
    classifier = RandomForestClassifier()
    classifier.fit(labeled_X, labeled_y)
    confidence_scores = classifier.predict_proba(unlabeled_X)
    query_index = np.argmin(np.max(confidence_scores, axis=1))
    query_X = unlabeled_X[query_index]
    
   
    updated_X = np.vstack((labeled_X, query_X))
    updated_y = np.concatenate((labeled_y, [-1]))  
  
    query_image = query_X.reshape(8, 8)
    plt.imshow(query_image, cmap='gray')
    plt.title("Select the label for the query image")
    plt.show()
    query_label = int(input("Enter the label for the query image: "))
    updated_y[-1] = query_label
    
    
    labeled_X = updated_X
    labeled_y = updated_y
    
  
    unlabeled_X = np.delete(unlabeled_X, query_index, axis=0)
   
    shown_examples += 1

X_train = labeled_X
y_train = labeled_y

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


accuracy = classifier.score(X_test, y_test)
print("Classification accuracy: {:.2f}".format(accuracy))
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


predictions = classifier.predict(X_test)


for i in range(len(X_test)):
    query_image = X_test[i].reshape(8, 8)
    plt.imshow(query_image, cmap='gray')
    predicted_label = predictions[i]
    true_label = y_test[i]
    plt.title(f"Predicted Label: {predicted_label}, True Label: {true_label}")
    plt.show()