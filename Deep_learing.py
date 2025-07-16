# Step 1: Import libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names for reference
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

               # Step 3: Build the CNN model
model = models.Sequential([
                layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation='relu'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10)])

# Step 4: Compile the model
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
# Step 5: Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
# Step 6: Visualize training accuracy and loss
plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
# Step 7: Make predictions
probability_model = tf.keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(x_test)
# Step 8: Visualize predictions
def plot_image(i, predictions_array, true_label, img):
    true_label = int(true_label[i])
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% (True: {class_names[true_label]})",color=color)

# Plot the first 9 test images with predicted labels
plt.figure(figsize=(10,10))
for i in range(9):
     plt.subplot(3, 3, i + 1)
    plot_image(i, predictions[i], y_test, x_test)
plt.tight_layout()
plt.show()