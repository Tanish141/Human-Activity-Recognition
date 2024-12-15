# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
df = pd.read_csv(r"D:\Udemy Python Projects\Hand Written Recognition\Dataset\hr_data.csv")

# Preprocess the dataset
X = df.drop('Activity',axis=1)
y = df['Activity']

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2,random_state=42)

# Train the Random forest Classifier
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Predictions of Test set
y_pred = model.predict(X_test)

# Evualte the model using accuracy, precsion, recall, f1-score
accuracy = accuracy_score(y_test,y_pred,)
precision = precision_score(y_test,y_pred,average='weighted')
recall = recall_score(y_test,y_pred,average='weighted')
f1 = f1_score(y_test,y_pred,average='weighted')

# Print scores 
print(f"Accuracy: {accuracy * 100: .2f}")
print(f"Precision : {precision * 100: .2f}")
print(f"Recall: {recall * 100: .2f}")
print(f"F1 Score : {f1 * 100: .2f}")

# Visulaize the confusion matric using Seabornsn Heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

