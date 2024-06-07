#!/usr/bin/env python
# coding: utf-8

# In[22]:


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

Main_image = "K:/LEGOs/Bricks/BRICKS_GRAY"# zmienić !!!!!!!!!!!!!!!!!!!!!!!!

photo_path_gray = "K:/LEGOs/Bricks/gray2.png" # zmienić !!!!!!!!!!!!!!!!!!!!!!!!



image = cv2.imread(Main_image) 
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(photo_path_gray, grayscale)
print("done changing to gray")


data_dir_Brick = Main_image
data_dir_Gray = photo_path_gray

batch_size = 32
img_height = 224
img_width = 224


#----------------------------------------------------------------#
train_ds_Brick = tf.keras.utils.image_dataset_from_directory(
  data_dir_Brick,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds_Brick = tf.keras.utils.image_dataset_from_directory(
  data_dir_Brick,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names_Brick = train_ds_Brick.class_names
print(class_names_Brick)
#----------------------------------------------------------------#
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
#----------------------------------------------------------------#



















































# In[7]:

#----------------------------------------------------------------#
AUTOTUNE = tf.data.AUTOTUNE

train_ds_Brick = train_ds_Brick.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_Brick = val_ds_Brick.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer_Brick = layers.Rescaling(1./255)

normalized_ds_Brick = train_ds_Brick.map(lambda x, y: (normalization_layer_Brick(x), y))
image_batch_Brick, labels_batch_Brick = next(iter(normalized_ds_Brick))
first_image_Brick = image_batch_Brick[0]
# Notice the pixel values are now in `[0,1]`.
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
#----------------------------------------------------------------#






















# In[8]:

#----------------------------------------------------------------#
num_classes_Brick = len(class_names_Brick)

input_shape_Brick = (224,224,3)

base_model_Brick = MobileNet(input_shape_Brick,include_top=False, weights='imagenet')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model_Brick.trainable = False

inputs_Brick = tf.keras.Input(shape = input_shape_Brick)
x_Brick = preprocess_input(inputs_Brick)
x_Brick = layers.Normalization()(x_Brick)
print(x_Brick)
#x = layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
x = base_model_Brick(x_Brick, training = False)
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
#x = layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
x_Gray = base_model_Gray(x_Gray, training = False)
print(x_Gray)
#----------------------------------------------------------------#
















#----------------------------------------------------------------#
x_Brick = layers.Conv2D(16, 3, padding='same', activation='relu')(x_Brick)
x_Brick = layers.Dropout(0.2)(x_Brick) # for increasing accuracy !!!! <== Important!!!!

#x = layers.GlobalAveragePooling2D()(x)
print(x_Brick)
x_Brick = layers.Conv2D(32, 3, padding='same', activation='relu')(x_Brick)
x_Brick = layers.Dropout(0.2)(x_Brick) # for increasing accuracy !!!! <== Important!!!!
#x = layers.GlobalAveragePooling2D()(x)
x_Brick = layers.Conv2D(64, 3, padding='same', activation='relu')(x_Brick)
x_Brick = layers.Dropout(0.2)(x_Brick) # for increasing accuracy !!!! <== Important!!!!

x_Brick = layers.GlobalAveragePooling2D()(x_Brick)
x_Brick = layers.Flatten()(x_Brick)
x_Brick = layers.Dense(128, activation='relu')(x_Brick)
output_Brick = layers.Dense(num_classes_Brick, activation = 'softmax')(x_Brick)

model_Brick = tf.keras.Model(inputs_Brick, output_Brick )
#----------------------------------------------------------------#

x_Gray = layers.Conv2D(16, 3, padding='same', activation='relu')(x_Gray)
x_Gray = layers.Dropout(0.1)(x_Gray) # for increasing accuracy !!!! <== Important!!!!

#x = layers.GlobalAveragePooling2D()(x)
print(x_Gray)
x_Gray = layers.Conv2D(32, 3, padding='same', activation='relu')(x_Gray)
x_Gray = layers.Dropout(0.1)(x_Gray) # for increasing accuracy !!!! <== Important!!!!
#x = layers.GlobalAveragePooling2D()(x)
x_Gray = layers.Conv2D(64, 3, padding='same', activation='relu')(x_Gray)
x_Gray = layers.Dropout(0.1)(x_Gray) # for increasing accuracy !!!! <== Important!!!!

x_Gray = layers.GlobalAveragePooling2D()(x_Gray)
x_Gray = layers.Flatten()(x_Gray)
x_Gray = layers.Dense(128, activation='relu')(x_Gray)
output_Gray = layers.Dense(num_classes_Gray, activation = 'softmax')(x_Gray)

model_Gray = tf.keras.Model(inputs_Gray, output_Gray )
#----------------------------------------------------------------#









#----------------------------------------------------------------#
base_learning_rate = 0.001
model_Brick.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_Brick.summary()

#----------------------------------------------------------------#

base_learning_rate = 0.001
model_Gray.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_Gray.summary()

#----------------------------------------------------------------#










#----------------------------------------------------------------#
epochs_Brick=1
history_Brick = model_Brick.fit(
  train_ds_Brick,
  validation_data=val_ds_Brick,
  epochs=epochs_Brick
)

#----------------------------------------------------------------#

epochs_Gray=1
history_Gray = model_Gray.fit(
  train_ds_Gray,
  validation_data=val_ds_Gray,
  epochs=epochs_Gray
)
#----------------------------------------------------------------#





















#----------------------------------------------------------------#

acc_Brick = history_Brick.history['accuracy_Brick']
val_acc_Brick = history_Brick.history['val_accuracy_Brick']

loss_Brick = history_Brick.history['loss_Brick']
val_loss_Brick = history_Brick.history['val_loss_Brick']

epochs_range_Brick = range(epochs_Brick)

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

#----------------------------------------------------------------#










#----------------------------------------------------------------#
model_Brick.save('lego_classifier_Brick_Mobile_Shape_dropout10.keras')

print('saved_Brick')

#----------------------------------------------------------------#

model_Gray.save('lego_classifier_Gray_Mobile.keras')

print('saved_Gray')
#----------------------------------------------------------------#






























#----------------------------------------------------------------#
data_augmentation_Brick = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
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
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
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

#----------------------------------------------------------------#


# brick_path = "C:/Users/krzys/Downloads/red.jpg"

# img = tf.keras.utils.load_img(
#     brick_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )



#----------------------------------------------------------------#



model_Brick_model = tf.keras.models.load_model('lego_classifier_Brick_Mobile_Shape_dropout10.keras')
model_Gray_model = tf.keras.models.load_model('lego_classifier_Gray_Mobile_Shape_dropout10.keras')



img_Brick = tf.keras.utils.load_img(
    Main_image, target_size=(img_height, img_width)
)
img_array_Brick = tf.keras.utils.img_to_array(img_Brick)
img_array_Brick = tf.expand_dims(img_array_Brick, 0) # Create a batch

predictions_Brick = model_Brick.predict(img_array_Brick)
score_Brick = predictions_Brick[0]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names_Brick[np.argmax(score_Brick)], 100 * np.max(score_Brick))
)






img_Gray = tf.keras.utils.load_img(
    photo_path_gray, target_size=(img_height, img_width)
)
img_array_Gray = tf.keras.utils.img_to_array(img_Gray)
img_array_Gray = tf.expand_dims(img_array_Gray, 0) # Create a batch

predictions_Gray = model_Gray.predict(img_array_Gray)
score_Gray = predictions_Gray[0]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names_Gray[np.argmax(score_Gray)], 100 * np.max(score_Gray))
)


#----------------------------------------------------------------#
