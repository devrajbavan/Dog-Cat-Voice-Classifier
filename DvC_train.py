import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the seed for reproducibility
tf.random.set_seed(42)

# Set up the paths to your training and validation data
train_data_dir = 'train'  # Directory containing spectrogram images
valid_data_dir = 'valid'  # Directory containing spectrogram images

# Set up the parameters
batch_size = 32
image_height = 300
image_width = 300
num_epochs = 15

# Create the data generators for training and validation
train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
valid_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

valid_generator = valid_data_gen.flow_from_directory(
    valid_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size
)

# Save the final model
model.save('meow_vs_Bark.h5')

