import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 128
DATASET_PATH = "Data/train"
CATEGORIES = ["cat", "dog"]
MAX_IMAGES_PER_CLASS = 500  # 500 cats + 500 dogs = 1000 images

print("[INFO] Script started.")
print("[INFO] Loading data...")

X = []
y = []

print("[INFO] Loading images from dataset...")

for label, category in enumerate(CATEGORIES):
    category_path = os.path.join(DATASET_PATH)
    count = 0
    for filename in os.listdir(category_path):
        if filename.startswith(category):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                X.append(img.flatten())  # Flatten 64x64 to 4096
                y.append(label)
                count += 1
                if count >= MAX_IMAGES_PER_CLASS:
                    print(f"[INFO] Loaded {count} images for class '{category}'; stopping early.")
                    break
    print(f"[INFO] Finished processing category: {category}")

X = np.array(X)
y = np.array(y)

print(f"[INFO] Data loaded. Shape of X: {X.shape}")
print("[INFO] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Training SVM model...")
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

print("[INFO] Predicting on test data...")
y_pred = model.predict(X_test)

print("[INFO] Classification report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

print("[INFO] Showing sample predictions...")

for i in range(5):
    img = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {CATEGORIES[y_pred[i]]}, True: {CATEGORIES[y_test[i]]}")
    plt.axis('off')
    plt.show()
