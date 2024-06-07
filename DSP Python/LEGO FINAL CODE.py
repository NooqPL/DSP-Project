#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNet
import cv2

import pathlib

Main_image = "K:/LEGOs/Bricks/BRICKS_COLOR" # zmienić !!!!!!!!!!!!!!!!!!!!!!!!
photo_path_gray = "K:/LEGOs/Bricks/BRICKS_GRAY" # zmienić !!!!!!!!!!!!!!!!!!!!!!!!






data_dir_Brick = Main_image
data_dir_Gray = photo_path_gray

batch_size = 32 
img_height = 224
img_width = 224

print("done")


# In[10]:

#
#
# Create a dataset from a directory of images
#
#
# Create the training dataset from a directory of color images
train_ds_Brick = tf.keras.utils.image_dataset_from_directory(
  data_dir_Brick,                           # Path to the directory containing the image data
  validation_split=0.2,                     # Fraction of data to reserve for validation (20%)
  subset="training",                        # Specify that this subset is for training
  seed=123,                                 # Seed for shuffling and transformations to ensure reproducibility
  image_size=(img_height, img_width),       # Resize images to this size (height, width)
  batch_size=batch_size)                    # Number of images to be yielded from the dataset per batch


# Create the validation dataset from the same directory of images
val_ds_Brick = tf.keras.utils.image_dataset_from_directory(
  data_dir_Brick,
  validation_split=0.2,
  subset="validation",                      # Specify that this subset is for validation
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names_Brick = train_ds_Brick.class_names  # Get the class names (categories) from the training dataset
print(class_names_Brick)
#----------------------------------------------------------------#
# Create the training dataset from a directory of gray images
train_ds_Gray = tf.keras.utils.image_dataset_from_directory(
  data_dir_Gray,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds_Gray = tf.keras.utils.image_dataset_from_directory(
  data_dir_Gray,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names_Gray = train_ds_Gray.class_names
print(class_names_Gray)
print("done")


# In[11]:


AUTOTUNE = tf.data.AUTOTUNE   # Enable automatic tuning of performance options

train_ds_Brick = train_ds_Brick.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)      # Optimize the training dataset: cache it in memory, shuffle it, and prefetch batches
val_ds_Brick = val_ds_Brick.cache().prefetch(buffer_size=AUTOTUNE)                        # Optimize the validation dataset: cache it in memory and prefetch batches

normalization_layer_Brick = layers.Rescaling(1./255)           # Create a normalization layer to rescale pixel values from [0, 255] to [0, 1]

normalized_ds_Brick = train_ds_Brick.map(lambda x, y: (normalization_layer_Brick(x), y))  # Apply the normalization layer to the training dataset
image_batch_Brick, labels_batch_Brick = next(iter(normalized_ds_Brick))                   # Get a batch of images and labels from the normalized dataset
first_image_Brick = image_batch_Brick[0]        # Get the first image from the batch

print(np.min(first_image_Brick), np.max(first_image_Brick))
#----------------------------------------------------------------#
train_ds_Gray = train_ds_Gray.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_Gray = val_ds_Gray.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer_Gray = layers.Rescaling(1./255)

normalized_ds_Gray = train_ds_Gray.map(lambda x, y: (normalization_layer_Gray(x), y))
image_batch_Gray, labels_batch_Gray = next(iter(normalized_ds_Gray))
first_image_Gray = image_batch_Gray[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image_Gray), np.max(first_image_Gray))

print("done")


# In[12]:


num_classes_Brick = len(class_names_Brick)    # Define the number of classes based on the length of the class names list

input_shape_Brick = (224,224,3)               # Specify the input shape for the model

# Load the pre-trained MobileNet model without the top classification layer
base_model_Brick = MobileNet(input_shape_Brick,include_top=False, weights='imagenet') 

# Define a preprocessing function for the MobileNet model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Freeze the base model to prevent its weights from being updated during training
base_model_Brick.trainable = False

# Define the input layer with the specified input shape
inputs_Brick = tf.keras.Input(shape = input_shape_Brick)  # Apply the MobileNet preprocessing function to the inputs
x_Brick = preprocess_input(inputs_Brick)                  # Apply normalization to the preprocessed inputs
x_Brick = layers.Normalization()(x_Brick)                 # Print the tensor after preprocessing and normalization
print(x_Brick)

x_Brick = base_model_Brick(x_Brick, training = False)           # Pass the normalized inputs through the base MobileNet model without training
print(x_Brick)
#----------------------------------------------------------------#
num_classes_Gray = len(class_names_Gray)

input_shape_Gray = (224,224,3)

base_model_Gray = MobileNet(input_shape_Gray,include_top=False, weights='imagenet')

preprocess_input_Gray = tf.keras.applications.mobilenet_v2.preprocess_input

base_model_Gray.trainable = False

inputs_Gray = tf.keras.Input(shape = input_shape_Gray)
x_Gray = preprocess_input_Gray(inputs_Gray)
x_Gray = layers.Normalization()(x_Gray)
print(x_Gray)

x_Gray = base_model_Gray(x_Gray, training = False)
print(x_Gray)
print("done")


# In[13]:



# Add a Conv2D layer with 16 filters, 3x3 kernel size, 'same' padding, and ReLU activation
x_Brick = layers.Conv2D(16, 3, padding='same', activation='relu')(x_Brick)
x_Brick = layers.Dropout(0.2)(x_Brick)                             # for increasing accuracy !!!! <== Important!!!!
# Add a Dropout layer with a dropout rate of 0.2 to reduce overfitting and increase accuracy

print(x_Brick)

# Add a Conv2D layer with 32 filters, 3x3 kernel size, 'same' padding, and ReLU activation
x_Brick = layers.Conv2D(32, 3, padding='same', activation='relu')(x_Brick)
x_Brick = layers.Dropout(0.2)(x_Brick)                            # for increasing accuracy !!!! <== Important!!!!
# Add a Dropout layer with a dropout rate of 0.2 to reduce overfitting and increase accuracy


# Add a Conv2D layer with 64 filters, 3x3 kernel size, 'same' padding, and ReLU activation
x_Brick = layers.Conv2D(64, 3, padding='same', activation='relu')(x_Brick)
x_Brick = layers.Dropout(0.2)(x_Brick)                            # for increasing accuracy !!!! <== Important!!!!
# Add a Dropout layer with a dropout rate of 0.2 to reduce overfitting and increase accuracy

# Add a GlobalAveragePooling2D layer to reduce the spatial dimensions to a single value per filter
x_Brick = layers.GlobalAveragePooling2D()(x_Brick)
x_Brick = layers.Flatten()(x_Brick)   # Flatten the output from the previous layer
x_Brick = layers.Dense(128, activation='relu')(x_Brick)   # Add a Dense (fully connected) layer with 128 units and ReLU activation
output_Brick = layers.Dense(num_classes_Brick, activation = 'softmax')(x_Brick)
# Add a final Dense layer with 'num_classes_Brick' units and softmax activation for classification


# Create a Keras model with the specified inputs and outputs
model_Brick = tf.keras.Model(inputs_Brick, output_Brick )
#----------------------------------------------------------------#

x_Gray = layers.Conv2D(16, 3, padding='same', activation='relu')(x_Gray)
x_Gray = layers.Dropout(0.1)(x_Gray) # for increasing accuracy !!!! <== Important!!!!


print(x_Gray)
x_Gray = layers.Conv2D(32, 3, padding='same', activation='relu')(x_Gray)
x_Gray = layers.Dropout(0.1)(x_Gray) # for increasing accuracy !!!! <== Important!!!!



x_Gray = layers.Conv2D(64, 3, padding='same', activation='relu')(x_Gray)
x_Gray = layers.Dropout(0.1)(x_Gray) # for increasing accuracy !!!! <== Important!!!!



x_Gray = layers.GlobalAveragePooling2D()(x_Gray)
x_Gray = layers.Flatten()(x_Gray)
x_Gray = layers.Dense(128, activation='relu')(x_Gray)
output_Gray = layers.Dense(num_classes_Gray, activation = 'softmax')(x_Gray)

model_Gray = tf.keras.Model(inputs_Gray, output_Gray )

print("done")


# In[14]:

# Define the base learning rate for the optimizer
base_learning_rate = 0.001

# Compile the model with the specified optimizer, loss function, and metrics
model_Brick.compile(
  optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),  # Use the Adam optimizer with the base learning rate
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Use sparse categorical crossentropy loss function
  metrics=['accuracy']) # Monitor accuracy during training and evaluation

model_Brick.summary() # Print a summary of the model architecture

#----------------------------------------------------------------#

base_learning_rate = 0.001
model_Gray.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_Gray.summary()

print("done")


# In[15]:


epochs_Brick=1    # Set the number of epochs for training

# Train the model on the training dataset and validate on the validation dataset
history_Brick = model_Brick.fit(
  train_ds_Brick,                 # Training dataset
  validation_data=val_ds_Brick,   # Validation dataset
  epochs=epochs_Brick             # Number of epochs to train the model
)

#----------------------------------------------------------------#

epochs_Gray=1
history_Gray = model_Gray.fit(
  train_ds_Gray,
  validation_data=val_ds_Gray,
  epochs=epochs_Gray
)

print("done")


# In[ ]:

# Extract training and validation accuracy from the training history
acc_Brick = history_Brick.history['accuracy_Brick']
val_acc_Brick = history_Brick.history['val_accuracy_Brick']

# Extract training and validation loss from the training history
loss_Brick = history_Brick.history['loss_Brick']
val_loss_Brick = history_Brick.history['val_loss_Brick']

# Define the range of epochs for plotting
epochs_range_Brick = range(epochs_Brick)

# Create a new figure with a specified size
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range_Brick, acc_Brick, label='Training Accuracy_Brick')
plt.plot(epochs_range_Brick, val_acc_Brick, label='Validation Accuracy_Brick')
plt.legend(loc='lower right_Brick')
plt.title('Training and Validation Accuracy_Brick')

plt.subplot(1, 2, 2)
plt.plot(epochs_range_Brick, loss_Brick, label='Training Loss_Brick')
plt.plot(epochs_range_Brick, val_loss_Brick, label='Validation Loss_Brick')
plt.legend(loc='upper right_Brick')
plt.title('Training and Validation Loss_Brick')
plt.show()
#----------------------------------------------------------------#

acc_Gray = history_Gray.history['accuracy']
val_acc_Gray = history_Gray.history['val_accuracy']

loss_Gray = history_Gray.history['loss']
val_loss_Gray = history_Gray.history['val_loss']

epochs_range_Gray = range(epochs_Gray)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range_Gray, acc_Gray, label='Training Accuracy')
plt.plot(epochs_range_Gray, val_acc_Gray, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range_Gray, loss_Gray, label='Training Loss')
plt.plot(epochs_range_Gray, val_loss_Gray, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[16]:

# Save the trained model to a file with the specified filename
model_Brick.save('lego_classifier_Brick_Mobile_Shape_dropout10.keras')

print('saved_Brick')

#----------------------------------------------------------------#

model_Gray.save('lego_classifier_Gray_Mobile.keras')

print('saved_Gray')


# In[ ]:

# Define a sequential model for data augmentation
data_augmentation_Brick = keras.Sequential(
  [
    # Randomly flip images horizontally
    layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    # Randomly rotate images by a factor within [-0.1, 0.1] radians
    layers.RandomRotation(0.1),
    # Randomly zoom into images by a factor within [0.9, 1.1]
    layers.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images_Brick, _ in train_ds_Brick.take(1):
  for i_Brick in range(9):
    augmented_images_Brick = data_augmentation_Brick(images_Brick)
    ax = plt.subplot(3, 3, i_Brick + 1)
    plt.imshow(augmented_images_Brick[0].numpy().astype("uint8"))
    plt.axis("off")

#----------------------------------------------------------------#

data_augmentation_Gray = keras.Sequential(
  [
    layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images_Gray, _ in train_ds_Gray.take(1):
  for i_Gray in range(9):
    augmented_images_Gray = data_augmentation_Gray(images_Gray)
    ax_Gray = plt.subplot(3, 3, i_Gray + 1)
    plt.imshow(augmented_images_Gray[0].numpy().astype("uint8"))
    plt.axis("off")





# In[19]:

import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time



cam = cv2.VideoCapture(0)






#model_Brick_model = tf.keras.models.load_model('lego_classifier_Brick_Mobile_Shape_dropout10.keras')
#model_Gray_model = tf.keras.models.load_model('lego_classifier_Gray_Mobile.keras')

model_Brick_model = tf.keras.models.load_model('lego_classifier_Mobile.keras')
model_Gray_model = tf.keras.models.load_model('lego_classifier_Mobile_Shape_dropout10.keras')



test_photo_normal = "C:/Users/krzys/Desktop/frames/frame1.png"
gray_path_save = "K:/LEGOs/Bricks/gray.png"
brick_path = test_photo_normal






fig = plt.figure()










while(1):
    
    # Capture frame from webcam
    result, img_Brick = cam.read() 
     
    #Save frame on the test_photo_normal path from cv2 format to png
    cv2.imwrite(test_photo_normal, img_Brick)

    # Load frame in TensorFlow format
    img_Brick = tf.keras.utils.load_img(
    test_photo_normal, target_size=(img_height, img_width)
    )  #load frame from the test_photo_normal path in tf format
    img_array_Brick = tf.keras.utils.img_to_array(img_Brick)
    img_array_Brick = tf.expand_dims(img_array_Brick, 0) # Create a batch

    # Predictions on the color model
    predictions_Brick = model_Brick_model.predict(img_array_Brick)
    score_Brick = predictions_Brick[0]


    fig.add_subplot(2, 2, 1)   
    plt.imshow(test_photo_normal) 
    plt.axis('off') 
    plt.title("CAM View") 

    # Print prediction result for color
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names_Brick[np.argmax(score_Brick)], 100 * np.max(score_Brick))
    )
    
    
    
    
    
    
    image_gray = cv2.imread(test_photo_normal) #load frame from the same test_photo_normal path in cv2 format
    grayscale = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY) #change into grayscale
    cv2.imwrite(gray_path_save, grayscale) #save grayscale frame on the gray_path_save path
    print("done changing to gray")


    fig.add_subplot(2, 2, 2)   
    plt.imshow(gray_path_save) 
    plt.axis('off') 
    plt.title("Cam Pic to Gray")
    
    
    
    
    
    img_Gray = tf.keras.utils.load_img(
        gray_path_save, target_size=(img_height, img_width)
    ) #load grayscale frame from the same test_photo_normal path in tf format
    img_array_Gray = tf.keras.utils.img_to_array(img_Gray)
    img_array_Gray = tf.expand_dims(img_array_Gray, 0) # Create a batch

    predictions_Gray = model_Gray_model.predict(img_array_Gray)
    score_Gray = predictions_Gray[0]

    fig.add_subplot(2, 2, 3)  
    plt.figtext(0, 0, "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names_Gray[np.argmax(score_Gray)], 100 * np.max(score_Gray), fontsize = 20))
    fig.add_subplot(2, 2, 4) 
    plt.figtext(0, 0, "This image most likely belongs to {} with a {:.2f} percent confidence.%d ".format(class_names_Brick[np.argmax(score_Brick)], 100 * np.max(score_Brick)), fontsize = 20)



    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names_Gray[np.argmax(score_Gray)], 100 * np.max(score_Gray))
        )
    print(
            "-----------------------------------"
            
        )

    time.sleep(0.1)