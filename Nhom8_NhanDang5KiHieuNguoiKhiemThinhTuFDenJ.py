# -*- coding: utf-8 -*-
"""### 1. Load Libraries"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

"""### 2. Load Data"""

# Paths
GESTURE_IMAGE_PATH = "Data/Gesture Image Data"
PREPROCESSED_IMAGE_PATH = "Data/Gesture Image Pre-Processed Data"

# Định nghĩa nhãn
CLASSES = ["F", "G", "H", "I", "J"]
IMG_SIZE = 50  # Kích thước ảnh

def load_data(image_path, preprocessed_path):
    images, labels, pred_masks = [], [], []

    for class_name in CLASSES:  # Duyệt đúng thứ tự F -> J
        class_dir = os.path.join(image_path, class_name)
        preprocessed_dir = os.path.join(preprocessed_path, class_name)

        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(class_dir) or not os.path.exists(preprocessed_dir):
            print(f"Folder không tồn tại: {class_dir} hoặc {preprocessed_dir}")
            continue

        for img_name in os.listdir(class_dir):  # Lặp qua từng ảnh
            img_path = os.path.join(class_dir, img_name)
            preprocessed_img_path = os.path.join(preprocessed_dir, img_name)

            # Đọc ảnh gốc
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Chuẩn hóa

            # Đọc ảnh đã xử lý trước nếu có
            if os.path.exists(preprocessed_img_path):
                pred_mask = cv2.imread(preprocessed_img_path, cv2.IMREAD_GRAYSCALE)
                pred_mask = cv2.resize(pred_mask, (IMG_SIZE, IMG_SIZE))
                pred_mask = pred_mask / 255.0  # Chuẩn hóa mask
            else:
                pred_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

            images.append(img)
            labels.append(CLASSES.index(class_name))  # Chuyển thành số 0-4
            pred_masks.append(pred_mask)

    # Chuyển thành numpy array
    images = np.array(images)
    labels = np.array(labels)
    pred_masks = np.array(pred_masks)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(CLASSES))

    return images, labels, pred_masks

# Load data
X, y, pred_masks = load_data(GESTURE_IMAGE_PATH, PREPROCESSED_IMAGE_PATH)

print("X shape:", X.shape)

print("y shape:", y.shape)

print("pred_masks shape:", pred_masks.shape)

"""### 3. Train-Validation-Test Split"""

X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(X, y, pred_masks, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test, mask_valid, mask_test = train_test_split(X_temp, y_temp, mask_temp, test_size=0.5, random_state=42)

"""### 4. Data Visualization"""

# Chuyển one-hot encoding về nhãn số (0, 1, 2, 3, 4)
y_train_labels = np.argmax(y_train, axis=1)

plt.figure(figsize=(10, 4))

# Scatter plot
plt.subplot(1, 3, 1)
plt.scatter(range(len(y_train_labels)), y_train_labels, alpha=0.5)
plt.title("Scatter Plot of Training Labels")
plt.yticks([0, 1, 2, 3, 4], ["F", "G", "H", "I", "J"])

# Bar plot (đếm số lượng từng nhãn)
plt.subplot(1, 3, 2)
unique_labels, counts = np.unique(y_train_labels, return_counts=True)
plt.bar(unique_labels, counts)
plt.xticks(unique_labels, ["F", "G", "H", "I", "J"])
plt.title("Class Distribution in Training Set")

# Pie chart (tương tự bar plot)
plt.subplot(1, 3, 3)
plt.pie(counts, labels=["F", "G", "H", "I", "J"], autopct='%1.1f%%')
plt.title("Class Distribution Pie Chart")

plt.tight_layout()
plt.show()

"""### 5. Feature Extraction with Transfer Learning"""

def build_model(base_model):
    base_model.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(y.shape[1], activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

vgg16_model = build_model(VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3)))
print("VGG16 Model Summary:")
vgg16_model.summary()

resnet50_model = build_model(ResNet50(weights='imagenet', include_top=False, input_shape=(50, 50, 3)))
print("ResNet50 Model Summary:")
resnet50_model.summary()

"""### 6. Train Models"""

print("Traning VGG16 model")
vgg16_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=32)
vgg16_model.save("Weights/vgg16_model.h5")

print("Training Resnet50 model")
resnet50_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=32)
resnet50_model.save("Weights/resnet50_model.h5")

"""### 7. Evaluate Models"""

from tensorflow.keras.models import load_model

# Load saved models
vgg16_model = load_model("Weights/vgg16_model.h5")
resnet50_model = load_model("Weights/resnet50_model.h5")

y_pred_vgg16 = np.argmax(vgg16_model.predict(X_test), axis=1)
y_pred_resnet50 = np.argmax(resnet50_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
class_labels = ["F", "G", "H", "I", "J"]

print("VGG16 Classification Report:")
print(classification_report(y_true, y_pred_vgg16, target_names=class_labels))

print("ResNet50 Classification Report:")
print(classification_report(y_true, y_pred_resnet50, target_names=class_labels))

# Danh sách nhãn chữ
class_labels = ["F", "G", "H", "I", "J"]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_true, y_pred_vgg16), annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("VGG16 Confusion Matrix")
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_true, y_pred_resnet50), annot=True, fmt='d', cmap='Reds', xticklabels=class_labels, yticklabels=class_labels)
plt.title("ResNet50 Confusion Matrix")
plt.show()

"""### 8. Test with Video without MediaPipe"""

# Danh sách nhãn chữ
class_labels = ["F", "G", "H", "I", "J"]

def test_video(model, name):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (50, 50)) / 255.0
        img = np.expand_dims(img, axis=0)

        pred_index = np.argmax(model.predict(img))
        pred_label = class_labels[pred_index]  # Lấy nhãn chữ từ danh sách

        cv2.putText(frame, f"{name}: {pred_label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy thử với VGG16 và ResNet50
test_video(vgg16_model, "VGG16")
test_video(resnet50_model, "ResNet50")

"""### 9. Test with Video with MediaPipe"""

import mediapipe as mp

class_labels = ["F", "G", "H", "I", "J"]

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def test_video_mediapipe(model, name):
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Resize ảnh về (50,50) và chuẩn hóa
            img = cv2.resize(frame, (50, 50)) / 255.0
            img = np.expand_dims(img, axis=0)

            # Lấy nhãn dự đoán
            pred_index = np.argmax(model.predict(img))
            pred_label = class_labels[pred_index]

            # Hiển thị nhãn dự đoán
            cv2.putText(frame, f"{name} + MediaPipe: {pred_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Vẽ landmark bàn tay
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy thử với VGG16 và ResNet50
test_video_mediapipe(vgg16_model, "VGG16")
test_video_mediapipe(resnet50_model, "ResNet50")

"""### 10. Convert into voice (still in development mode)"""

import pyttsx3

# Khởi tạo Text-to-Speech
engine = pyttsx3.init()
labels = ["F", "G", "H", "I", "J"]

# MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def test_video_speech(model, name):
    cap = cv2.VideoCapture(0)
    last_pred = None  # Tránh lặp lại cùng một âm thanh liên tục

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            img_resized = cv2.resize(frame, (50, 50)) / 255.0
            img_expanded = np.expand_dims(img_resized, axis=0)

            pred_index = np.argmax(model.predict(img_expanded))
            pred_label = labels[pred_index]

            # Hiển thị kết quả lên màn hình
            cv2.putText(frame, f"{name} + TTS: {pred_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Đọc nhãn bằng giọng nói nếu khác lần trước
            if pred_label != last_pred:
                engine.say(pred_label)
                engine.runAndWait()
                last_pred = pred_label

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy thử nghiệm với mô hình
test_video_speech(vgg16_model, "VGG16")
test_video_speech(resnet50_model, "ResNet50")