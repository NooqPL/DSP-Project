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

import pathlib

data_dir = "K:\LEGOs\Bricks"

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


# In[23]:


train_ds


# In[6]:


import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")


# In[7]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# In[8]:


num_classes = len(class_names)

input_shape = (224,224,3)

base_model = MobileNet(input_shape,include_top=False, weights='imagenet')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model.trainable = False

inputs = tf.keras.Input(shape = input_shape)
x = preprocess_input(inputs)
x = layers.Normalization()(x)
print(x)
#x = layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
x = base_model(x, training = False)
print(x)





# code added here

# x = layers.GlobalAveragePooling2D()(x) 
#     #include dropout with probability of 0.2 to avoid overfitting
# x = layers.Dropout(0.2)(x)
#     # create a prediction layer with one neuron (as a classifier only needs one)
# prediction_layer = layers.Dense(1)

# outputs = prediction_layer(x)
# model = tf.keras.Model(inputs, outputs)

# end








x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
x = layers.Dropout(0.2)(x) # for increasing accuracy !!!! <== Important!!!!

#x = layers.GlobalAveragePooling2D()(x)
print(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.Dropout(0.2)(x) # for increasing accuracy !!!! <== Important!!!!
#x = layers.GlobalAveragePooling2D()(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Dropout(0.2)(x) # for increasing accuracy !!!! <== Important!!!!

x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes, activation = 'softmax')(x)

model = tf.keras.Model(inputs, output )

"""model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])"""


"""model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])"""

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[ ]:





# In[9]:


epochs=1
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[10]:


model.save('lego_classifier_Mobile.keras')

print('saved')


# In[11]:


data_augmentation = keras.Sequential(
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
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


# In[24]:


brick_path = "C:/Users/krzys/Downloads/blue.jpg"

img = tf.keras.utils.load_img(
    brick_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[26]:


max(predictions[0])


# In[13]:


brick_path = "C:/Users/krzys/Downloads/red.jpg"

img = tf.keras.utils.load_img(
    brick_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[14]:


brick_path = "C:/Users/krzys/Downloads/cylinder.jpg"

img = tf.keras.utils.load_img(
    brick_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[15]:


brick_path = "C:/Users/krzys/Downloads/tyre.jpg"

img = tf.keras.utils.load_img(
    brick_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[27]:


brick_path = "C:/Users/krzys/Downloads/banana.jpg"

img = tf.keras.utils.load_img(
    brick_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[28]:


predictions[0]


# In[29]:


brick_path = "K:/LEGOs/Bricks/Red_Brick/red_100.png"

img = tf.keras.utils.load_img(
    brick_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[30]:


predictions[0]


# In[20]:


train_ds


# In[ ]:


import numpy
from sklearn import metrics



confusion_matrix = metrics.confusion_matrix(actual, predicted)

