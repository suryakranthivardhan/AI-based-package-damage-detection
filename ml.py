import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pickle  

# Set your dataset path
data_dir = "originaldata"
categories = ["damaged", "non damaged"]

# Function to extract features (color histogram in this case)
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize for consistency
    hist = cv2.calcHist([image], [0, 1, 2], None, [8,8,8], [0,256, 0,256, 0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Prepare dataset
X = []
y = []

for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        features = extract_features(img_path)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model (SVM)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Save the trained model to a pickle file
model_filename = 'goods_damage_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")
