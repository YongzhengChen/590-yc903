#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:37:01 2021

@author: chenyongzheng
"""

from keras import layers
from keras import models
from keras.models import load_model
import numpy as np

## Parameters:
data = 'mnist'
data_augmentation = True
epochs=30 
batch_size=64
model_type = 'DFF'



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from scipy.ndimage.interpolation import shift


## determine which dataset to use
if data == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if data == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
if data == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    

# Visualize an image in the dataset.
import matplotlib.pyplot as plt
num = 10
images = train_images[:num]
labels = train_labels[:num]
num_row = 2
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()

# define a function to do data aucmentation.
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

if data_augmentation:
    train_images = [image for image in train_images]
    train_labels = [image for image in train_labels]
    for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
        for image, label in zip(train_images, train_labels):
            train_images.append(shift_image(image, dx, dy))
            train_labels.append(label)
    train_images = train_images.reshape((300000, 28, 28, 1))
else:
    train_images = train_images.reshape((60000, 28, 28, 1))
    
print('1')


train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Do 80-20 split of the “training” data into (train/validation)
from sklearn.model_selection import train_test_split
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2)
history = model.fit(train_images, train_labels, epochs=5, batch_size=64,validation_data=(validation_images, validation_labels))

print('2')

## Include a training/validation history plot at the end and print (train/test/val) metrics
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)

## Save model and parameters
model.save("my_h5_model.h5")

## Read model 
model = load_model("my_h5_model.h5")

## Have a function(s) that visualizes what the CNN is doing inside
img_path = 'img_1.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_images[0])

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')



