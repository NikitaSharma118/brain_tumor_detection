import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, EfficientNetB3
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import random 
from keras.preprocessing import image
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.models import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# from google.colab import drive
# drive.mount('/content/drive')

# """mounting the drive

# """

base_dir="dataset"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

"""checking for category of image"""

classes1 = os.listdir(train_dir)
print("Tumor Categories:", classes1)
classes2 = os.listdir(test_dir)
print("Tumor Categories:", classes2)

"""random maping of image from each category"""

import random
import matplotlib.image as mpimg

plt.figure(figsize=(12, 8))

for i, category in enumerate(classes1):
    folder = os.path.join(train_dir, category)
    file = random.choice(os.listdir(folder))   # pick random image
    img_path = os.path.join(folder, file)

    img = mpimg.imread(img_path)

    plt.subplot(2, 2, i+1)
    plt.imshow(img, cmap="gray")
    plt.title(category)
    plt.axis("off")

plt.show()

"""count of image per category"""

train_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in classes1}
test_counts = {cls: len(os.listdir(os.path.join(test_dir, cls))) for cls in classes2}

print("\nTrain Image Counts:", train_counts)
print("Test Image Counts:", test_counts)

df = pd.DataFrame({
    "Class": list(train_counts.keys()),
    "Train": list(train_counts.values()),
    "Test": list(test_counts.values())
})
df.plot(x="Class", kind="bar", figsize=(8,5))
plt.title("Number of Images per Class (Train vs Test)")
plt.ylabel("Image Count")
plt.show()

"""finding shapes(train data)"""

shapes = []
for cls in classes1:
    sample_imgs = os.listdir(os.path.join(train_dir, cls))[:20]  # check first 20 images
    for img_name in sample_imgs:
        img = cv2.imread(os.path.join(train_dir, cls, img_name))
        shapes.append(img.shape)
unique_shapes = np.unique(shapes, axis=0)
print("\nUnique image sizes in dataset:", unique_shapes)

"""Check if images are grayscale or RGB"""

img_sample_path = os.path.join(train_dir, classes1[0], os.listdir(os.path.join(train_dir, classes1[0]))[0])
img_sample = cv2.imread(img_sample_path)
print("Sample image shape:", img_sample.shape)

"""Pixel intensity distribution for one random image"""

img_gray = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)
plt.hist(img_gray.ravel(), bins=50, color='purple')
plt.title("Pixel Intensity Distribution (Sample Image)")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.show()

"""Average image per class (to visualize general features)"""

plt.figure(figsize=(12, 6))
for idx, cls in enumerate(classes1):
    folder = os.path.join(train_dir, cls)
    imgs = []
    for file in os.listdir(folder)[:30]:  # take first 30 images per class
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # resize for averaging
        imgs.append(img)
    avg_img = np.mean(imgs, axis=0)

    plt.subplot(1, len(classes1), idx+1)
    plt.imshow(avg_img, cmap="gray")
    plt.title(f"Avg {cls}")
    plt.axis("off")

plt.suptitle("Average Image per Class", fontsize=16)
plt.show()

"""preprocessing"""

IMG_SIZE = 224  # Updated size
RANDOM_STATE = 42

# ------------------------
# Function to load & preprocess images
# ------------------------
def load_images(data_dir, classes, img_size=IMG_SIZE):
    X, y = [], []
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)      # RGB
            img = cv2.resize(img, (img_size, img_size))  # Resize to 255x255
            img = img / 255.0               # Normalize
            X.append(img)
            y.append(cls)
    return np.array(X), np.array(y)


# Set global seed
SEED = 42
np.random.seed(SEED)       # for numpy ops
random.seed(SEED)          # for python ops
tf.random.set_seed(SEED)   # for tensorflow ops

# Params
IMG_SIZE = 224
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2

)

# Only rescale for validation & test
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED
)

# Validation generator
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED
)

# Test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

"""after preprocessing eda"""

train_counts = train_generator.classes
val_counts = val_generator.classes
test_counts = test_generator.classes

plt.figure(figsize=(12,5))

plt.hist(train_counts, bins=np.arange(len(train_generator.class_indices)+1)-0.5, rwidth=0.6)
plt.title("Training Set Class Distribution")
plt.xticks(range(len(train_generator.class_indices)), list(train_generator.class_indices.keys()))
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,5))
plt.hist(val_counts, bins=np.arange(len(val_generator.class_indices)+1)-0.5, rwidth=0.6)
plt.title("Validation Set Class Distribution")
plt.xticks(range(len(val_generator.class_indices)), list(val_generator.class_indices.keys()))
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,5))
plt.hist(test_counts, bins=np.arange(len(test_generator.class_indices)+1)-0.5, rwidth=0.6)
plt.title("Test Set Class Distribution")
plt.xticks(range(len(test_generator.class_indices)), list(test_generator.class_indices.keys()))
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

"""sample of augmented image"""

images, labels = next(train_generator)  # get one batch
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.title(list(train_generator.class_indices.keys())[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle("Sample Augmented Training Images", fontsize=16)
plt.show()

# 3. Check shape of batches
print("Train batch shape:", images.shape)

plt.figure(figsize=(10,5))
plt.hist(images.ravel(), bins=50, color="blue", alpha=0.7)
plt.title("Pixel Intensity Distribution (After Rescaling)")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# 5. Check min/max values
print("Pixel value range in this batch:", images.min(), "to", images.max())


"""VGG16 with early stopping and drop out and unfreezing 8 layers"""

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_accuracy',  # or 'val_loss'
    patience=5,              # wait 5 epochs before stopping
    restore_best_weights=True,
    verbose=1
)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze initial layers
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),           # first dropout
    Dense(128, activation='relu'),
    Dropout(0.3),           # second dropout
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model with callbacks
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,               # longer because early stopping will stop earlier if needed
    callbacks=[early_stop]
)

# Fine-tuning Stage


# Unfreeze the last 8 layers of the base model
for layer in base_model.layers[-8:]:
    layer.trainable = True

# Compile again with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # smaller LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with early stopping
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,                 # fine-tuning usually converges faster
    callbacks=[early_stop]
)

# Combine both histories
acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

"""evaluating test accuracy"""

# Evaluate test accuracy
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

"""confusion matrix"""

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Get predictions (probabilities)
y_pred_prob = model.predict(test_generator)

# Convert probabilities to class indices
y_pred = np.argmax(y_pred_prob, axis=1)

# True class indices
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report using actual class names from test folder
report = classification_report(
    y_true,
    y_pred,
    target_names=classes2  # using the folder names directly
)
print("Classification Report:\n", report)

"""Roc curve"""

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# True labels
y_true = test_generator.classes

# Binarize the labels for multi-class ROC
y_true_bin = label_binarize(y_true, classes=np.arange(len(classes2)))

# Get predicted probabilities
y_pred_prob = model.predict(test_generator)

# Plot ROC curve for each class
plt.figure(figsize=(8,6))

for i, class_name in enumerate(classes2):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

# Random guessing line
plt.plot([0,1], [0,1], 'k--', label='Random')

plt.title("ROC Curve - Multi-class")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()





"""ResNet50"""

# Step 1: Load Pretrained ResNet50
base_model1 = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze all layers for initial training
for layer in base_model1.layers:
    layer.trainable = False

# Step 2: Add Custom Classification Head
x = base_model1.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)
model3 = Model(inputs=base_model1.input, outputs=output)

# Step 3: Compile (Stage 1)
model3.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

checkpoint_stage1 = ModelCheckpoint("resnet_stage1_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, min_lr=1e-7, verbose=1)

# Step 4: Train Top Layers
history_stage1 = model3.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint_stage1, earlystop, reduce_lr],
    verbose=1
)

# Step 5: Fine-tuning Stage (Unfreeze deeper layers)
for layer in base_model1.layers:
    if 'conv3' in layer.name or 'conv4' in layer.name or 'conv5' in layer.name:
        layer.trainable = True

model3.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

checkpoint_stage2 = ModelCheckpoint("resnet50_finetuned_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
callbacks_stage2 = [checkpoint_stage2, earlystop, reduce_lr]

history_stage2 = model3.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks_stage2,
    verbose=1
)

# Step 6: Evaluate on Test Set
test_loss, test_acc = model3.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Step 7: Save Final Model
model3.save("brain_tumor_resnet50_final.h5")
print("\n Model saved as brain_tumor_resnet50_final.h5")

"""confusion matrix for ResNet50"""

y_true = test_generator.classes
# Use the full model3 for prediction, not just the base model
y_pred_probs = model3.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Step 2: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = list(test_generator.class_indices.keys())

# Step 3: Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix - ResNet50 Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Step 4: Print Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

"""plotting accuracy and loss curves (ResNet50)"""

# Plot Accuracy
plt.figure(figsize=(8,6))
plt.plot(history_stage2.history['accuracy'], label='Training Accuracy')
plt.plot(history_stage2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(8,6))
plt.plot(history_stage2.history['loss'], label='Training Loss')
plt.plot(history_stage2.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

"""ROC curve for ResNet50"""

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

# Assuming you already have these:
y_true = test_generator.classes  # true class indices
y_pred_probs = model3.predict(test_generator, verbose=1)  # softmax probabilities
y_pred = np.argmax(y_pred_probs, axis=1)

# Binarize the output for multi-class ROC
num_classes = len(test_generator.class_indices)
class_labels = list(test_generator.class_indices.keys())
y_test_bin = label_binarize(y_true, classes=np.arange(num_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

for i, color in zip(range(num_classes), colors[:num_classes]):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for ResNet50 Brain Tumor Detection')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()





"""EfficientNetB3 with early stopping and fine tuning"""

base_model2 = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers initially
for layer in base_model2.layers:
    layer.trainable = False


model4 = Sequential([
    base_model2,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile
model4.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
checkpoint = ModelCheckpoint('best_model3.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# initial training
history3 = model4.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# 7️ Fine-Tuning: Unfreeze Top Layers
for layer in base_model2.layers[-100:]:
    layer.trainable = True

model4.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history3_finetune = model4.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

"""confusion matrix for EfficientNetB3"""

y_true = test_generator.classes
# Use the full model4 for prediction, not just the base model
y_pred_probs = model4.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Step 2: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_labels = list(test_generator.class_indices.keys())

# Step 3: Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix - ResNet50 Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Step 4: Print Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))


#ROC Curve for efficientNetB3
y_true = test_generator.classes  # true class indices
y_pred_probs = model4.predict(test_generator, verbose=1)  # softmax probabilities
y_pred = np.argmax(y_pred_probs, axis=1)

# Binarize the output for multi-class ROC
num_classes = len(test_generator.class_indices)
class_labels = list(test_generator.class_indices.keys())
y_test_bin = label_binarize(y_true, classes=np.arange(num_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

for i, color in zip(range(num_classes), colors[:num_classes]):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for ResNet50 Brain Tumor Detection')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()





"""Explainable Ai (Grad-CAM for ResNet50 )"""

# 1. Load an image from test set
img_path = test_generator.filepaths[300]   # change index to test other images
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# 2. Make prediction
preds = model3.predict(img_array)
predicted_class = np.argmax(preds[0])
class_labels = list(test_generator.class_indices.keys())

print(f"Predicted class: {class_labels[predicted_class]} (confidence: {preds[0][predicted_class]:.2f})")

# 3. Identify last convolutional layer
# For ResNet50, last conv layer is typically 'conv5_block3_out'
last_conv_layer_name = "conv5_block3_out"
last_conv_layer = model3.get_layer(last_conv_layer_name)

# 4. Create Grad-CAM model
grad_model = Model(inputs=model3.inputs, outputs=[last_conv_layer.output, model3.output])

# 5. Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, predicted_class]

# Gradient of the top predicted class with respect to conv outputs
grads = tape.gradient(loss, conv_outputs)

# Global average pooling to get importance weights
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# 6. Weight the channels
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

# Normalize the heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap)
heatmap = heatmap.numpy()

# 7. Superimpose heatmap on original image
img_orig = cv2.imread(img_path)
img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay heatmap on original image (0.4 controls transparency)
superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

# 8. Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_orig)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.title(f"Overlay - {class_labels[predicted_class]}")
plt.axis('off')

plt.tight_layout()
plt.show()

