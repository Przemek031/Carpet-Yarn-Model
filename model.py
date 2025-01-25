import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
import os

# Function to check the current working directory
def get_current_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current working directory: {current_dir}")
    return current_dir

# Load data
try:
    current_dir = get_current_directory()
    data_path = os.path.join(current_dir, 'carpets.csv')
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file 'carpets.csv' was not found in the directory {current_dir}. Please ensure the file exists.")
except pd.errors.EmptyDataError:
    raise ValueError(f"The file 'carpets.csv' is empty. Please ensure the file contains data.")
except pd.errors.ParserError:
    raise ValueError(f"The file 'carpets.csv' contains parsing errors. Please ensure the file is correctly formatted.")

# Function to load images
def load_image(path):
    image_path = os.path.join(current_dir, path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file '{path}' does not exist.")
    img = load_img(image_path, target_size=(256, 256))  # Resize images to 256x256 pixels
    img = img_to_array(img)
    img /= 255.0  # Normalize
    print(f"Loaded image with shape: {img.shape}")  # Debugging
    return img

# Load images
try:
    data['image'] = data['image_path'].apply(load_image)
except Exception as e:
    raise ValueError(f"An issue occurred while loading images: {e}")

# Transform data into numpy arrays
try:
    def parse_size(size):
        try:
            # Automatically correct format, e.g., replace 'z' with 'x'
            size = size.replace('z', 'x').strip()
            dimensions = size.split('x')
            return [int(dimensions[0]), int(dimensions[1])]
        except (ValueError, IndexError):
            raise ValueError(f"Invalid size format: {size}")

    X_images = np.array(data['image'].tolist(), dtype=np.float32)
    X_sizes = np.array(data['size'].apply(parse_size).tolist(), dtype=np.float32)
    y = np.array(data['yarn_used'].to_numpy(), dtype=np.float32)
except KeyError as e:
    raise KeyError(f"Missing required key in the data: {e}")
except ValueError as e:
    raise ValueError(f"Error while processing the 'size' column: {e}")

# Model for image analysis
image_input = Input(shape=(256, 256, 3), name='image_input')
x = Conv2D(32, (3, 3), activation='relu', name='conv1')(image_input)
x = MaxPooling2D((2, 2), name='pool1')(x)
x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
x = MaxPooling2D((2, 2), name='pool2')(x)
x = Flatten(name='flatten')(x)

# Model for numerical data analysis
size_input = Input(shape=(2,), name='size_input')
y = Dense(64, activation='relu', name='dense1')(size_input)
y = Dense(32, activation='relu', name='dense2')(y)

# Combine both models
combined = Concatenate(name='concatenate')([x, y])
z = Dense(64, activation='relu', name='dense3')(combined)
z = Dense(1, activation='linear', name='output')(z)  # Output a single value (yarn amount)

# Create and compile the model
model = Model(inputs=[image_input, size_input], outputs=z, name='carpet_model')
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
print(type(X_images), type(X_sizes), type(y))  # Debugging data types
print(X_images.shape, X_sizes.shape, y.shape)  # Debugging data shapes

try:
    X_images = np.array(data['image'].tolist(), dtype=np.float32)
    X_sizes = np.array(data['size'].apply(parse_size).tolist(), dtype=np.float32)
    y = np.array(data['yarn_used'].to_numpy(), dtype=np.float32)

    print("Starting model training...")
    print(f"Data shapes: X_images={X_images.shape}, X_sizes={X_sizes.shape}, y={y.shape}")

    history = model.fit(
        [X_images, X_sizes], y, 
        epochs=250,
        batch_size=8,
        validation_split=0.2
    )
except Exception as e:
    raise RuntimeError(f"An issue occurred during model training: {e}")

# Visualize the training process
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Model Training Process')
plt.show()

# Example model usage
try:
    test_image_path = os.path.join(current_dir, 'test.jpg')
    new_image = load_image(test_image_path)
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension
    new_size = np.array([[81, 83]], dtype=np.float32)
    predicted_yarn = model.predict([new_image, new_size])
    print(f"Estimated amount of yarn needed: {round(predicted_yarn[0][0])}")
    print(f"Actual number of 50g skeins: {math.ceil((predicted_yarn[0][0]) / 50)}")
except Exception as e:
    raise RuntimeError(f"An issue occurred during prediction: {e}")
