import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define paths to the training and testing data
train_dir = "/Users/dipeshdhamecha/Desktop/T&T/Train"
#train_rotten_apples_dir = "/Users/dipeshdhamecha/Desktop/Training and Test/dataset/Train/rottenapples"
test_dir = "/Users/dipeshdhamecha/Desktop/T&T/Test"
#test_rotten_apples_dir = "/Users/dipeshdhamecha/Desktop/Training and Test/dataset/Test/rottenapples"

# Define the main directory paths
#train_dir = os.path.dirname(train_fresh_apples_dir)
#test_dir = os.path.dirname(test_fresh_apples_dir)

# Define image size and batch size
img_height = 150
img_width = 150
batch_size = 32

# Create an ImageDataGenerator for data augmentation (optional)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # binary labels for fresh/stale classification
)

# Load and preprocess the testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Define the CNN model
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    #MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    Flatten(),
    Dense(5000, activation='relu'),
    Dense(1000, activation='relu'),
    #Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])
print(model.summary())

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.2f}")