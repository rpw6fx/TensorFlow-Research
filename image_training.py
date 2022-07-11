# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:48:27 2022

@author: rpwoo
"""

# some code adapted from coursera's DeepLearning.AI TensorFlow Developer 
# Professional Certificate taught/created by Andrew Ng and Laurence Moroney


# import statements
import os
import string
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import csv
from shutil import copyfile
from numpy import hstack

# print(os.getcwd())

# Directory Variables
root_path = os.path.join(os.getcwd(), 'Images')
ZERO_SOURCE_DIR = os.path.join(root_path, '0 People')
# print(ZERO_SOURCE_DIR)
ONE_SOURCE_DIR = os.path.join(root_path, '1 Person')
# print(ONE_SOURCE_DIR)
TWO_SOURCE_DIR = os.path.join(root_path, '2 People')
# print(TWO_SOURCE_DIR)
THREE_SOURCE_DIR = os.path.join(root_path, '3 People')
# print(THREE_SOURCE_DIR)
FOUR_SOURCE_DIR = os.path.join(root_path, '4 People')
# print(FOUR_SOURCE_DIR)

testing_root = os.path.join(os.getcwd(), 'Testing')
TRAINING_DIR = os.path.join(testing_root, 'Training')
VALIDATION_DIR = os.path.join(testing_root, 'Validation')
# print(TRAINING_DIR)
# print(VALIDATION_DIR)
TRAINING_ZERO_DIR = os.path.join(TRAINING_DIR, 'zero')
VALIDATION_ZERO_DIR = os.path.join(VALIDATION_DIR, 'zero')
# print(TRAINING_ZERO_DIR)
# print(VALIDATION_ZERO_DIR)
TRAINING_ONE_DIR = os.path.join(TRAINING_DIR, 'one')
VALIDATION_ONE_DIR = os.path.join(VALIDATION_DIR, 'one')
TRAINING_TWO_DIR = os.path.join(TRAINING_DIR, 'two')
VALIDATION_TWO_DIR = os.path.join(VALIDATION_DIR, 'two')
TRAINING_THREE_DIR = os.path.join(TRAINING_DIR, 'three')
VALIDATION_THREE_DIR = os.path.join(VALIDATION_DIR, 'three')
TRAINING_FOUR_DIR = os.path.join(TRAINING_DIR, 'four')
VALIDATION_FOUR_DIR = os.path.join(VALIDATION_DIR, 'four')


# Empty directories in case you run this multiple times
if len(os.listdir(TRAINING_ZERO_DIR)) > 0:
  for file in os.scandir(TRAINING_ZERO_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_ONE_DIR)) > 0:
  for file in os.scandir(TRAINING_ONE_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_TWO_DIR)) > 0:
  for file in os.scandir(TRAINING_TWO_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_THREE_DIR)) > 0:
  for file in os.scandir(TRAINING_THREE_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_FOUR_DIR)) > 0:
  for file in os.scandir(TRAINING_FOUR_DIR):
    os.remove(file.path)
    
if len(os.listdir(VALIDATION_ZERO_DIR)) > 0:
  for file in os.scandir(VALIDATION_ZERO_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_ONE_DIR)) > 0:
  for file in os.scandir(VALIDATION_ONE_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_TWO_DIR)) > 0:
  for file in os.scandir(VALIDATION_TWO_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_THREE_DIR)) > 0:
  for file in os.scandir(VALIDATION_THREE_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_FOUR_DIR)) > 0:
  for file in os.scandir(VALIDATION_FOUR_DIR):
    os.remove(file.path)

# Other Variables
split_size = 0.8           # was 0.8 initially, 0.75 had batch size of 50


# ----------------------------- FUNCTION DEFINITIONS -------------------------

# code adapted from:
# https://intellipaat.com/community/7530/how-to-read-pgm-p2-image-in-python
def readpgm(name):
    with open(name) as f:
        lines = f.readlines()
        
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
            
    # here,it makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 
    
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
        
    # returns 
    # (a) the data 1xn array
    # (b) the length and width of the image
    # (c) the number of shades of gray of the image
    return (np.array(data[3:]),(data[1],data[0]),data[2])


def parse_data_from_input(filename):
  """
  Parses the images and labels from a CSV file
  
  Args:
    filename (string): path to the CSV file
    
  Returns:
    images, labels: tuple of numpy arrays containing the images and labels
  """
  with open(filename) as file:
    # Use csv.reader, passing in the appropriate delimiter
    # Remember that csv.reader can be iterated and returns one line in each iteration
    csv_reader = csv.reader(file, delimiter=',')
    # next(csv_reader)
    
    labels = []
    images = []
    count = 0

    for row in csv_reader:
      count = count + 1
      labels.append(row[0])
      images.append(row[1:])
    
    labels = np.asarray(labels, dtype = np.float64)
    images = np.asarray(images, dtype = np.float64)
    images = np.reshape(images, (count, 80, 60))

    return images, labels


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    """
    Splits the data into train and test sets
      
    Args:
        SOURCE_DIR (string): directory path containing the images
        TRAINING_DIR (string): directory path to be used for training
        VALIDATION_DIR (string): directory path to be used for validation
        SPLIT_SIZE (float): proportion of the dataset to be used for training
        
    Returns:
        None
      """
    
    # source_directory = os.listdir(SOURCE_DIR)
    new_dir = []
    for name in os.listdir(SOURCE_DIR):
        path = SOURCE_DIR + '\\' + name
        size = os.path.getsize(path)
        if size == 0 or size < 0:
            print(str(name) + " is zero length, so ignoring.")
        else:
            new_dir.append(name)
    
    num_files = len(new_dir)
    split = int(num_files * SPLIT_SIZE)
    
    randomized = random.sample(new_dir, num_files)
    training_images = randomized[:split]
    validation_images = randomized[split:]
    
    for filename in training_images:
        copyfile(SOURCE_DIR + '\\' + filename, TRAINING_DIR + '\\' + filename)
    for filename in validation_images:
        copyfile(SOURCE_DIR + '\\' + filename, VALIDATION_DIR + '\\' + filename)
        
        
# Plot a sample of 10 images from the training set
def plot_categories(training_images, training_labels):
  fig, axes = plt.subplots(1, 10, figsize=(80, 60))
  axes = axes.flatten()
  letters = list(string.ascii_lowercase)

  for k in range(10):
    img = training_images[k]
    img = np.expand_dims(img, axis=-1)
    img = array_to_img(img)
    ax = axes[k]
    ax.imshow(img, cmap="Greys_r")
    ax.set_title(f"{letters[int(training_labels[k])]}")
    ax.set_axis_off()

  plt.tight_layout()
  plt.show()
  
  
# GRADED FUNCTION: train_val_generators
def train_val_generators(training_images, training_labels, validation_images, validation_labels):
  """
  Creates the training and validation data generators
  
  Args:
    training_images (array): parsed images from the train CSV file
    training_labels (array): parsed labels from the train CSV file
    validation_images (array): parsed images from the test CSV file
    validation_labels (array): parsed labels from the test CSV file
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """

  # In this section you will have to add another dimension to the data
  # So, for example, if your array is (10000, 28, 28)
  # You will need to make it (10000, 28, 28, 1)
  # Hint: np.expand_dims
  training_images = np.expand_dims(training_images, axis = 3)
  validation_images = np.expand_dims(validation_images, axis = 3)

  # Instantiate the ImageDataGenerator class 
  # Don't forget to normalize pixel values 
  # and set arguments to augment the images (if desired)
  train_datagen = ImageDataGenerator(
      rescale = 1./65535,
      rotation_range=0,            # was 40, then 20 (maybe switch back???)
      width_shift_range=0.1,        # was 0.2
      height_shift_range=0.2,       # was 0.2
      shear_range=0,              # was 0.2
      zoom_range=0,               # was 0.2
      horizontal_flip=True,         # was True
      fill_mode='nearest')


  # Pass in the appropriate arguments to the flow method
  train_generator = train_datagen.flow(x=training_images,
                                       y=training_labels,
                                       batch_size=40)       # was 40 w/ 0.8

  
  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  # Remember that validation data should not be augmented
  # rescale by 2^16 because the pgm images are 16 bit
  validation_datagen = ImageDataGenerator(rescale = 1./65535.)

  # Pass in the appropriate arguments to the flow method
  validation_generator = validation_datagen.flow(x=validation_images,
                                                 y=validation_labels,
                                                 batch_size=40)     # was 32

  return train_generator, validation_generator

def create_model():   
    
    # NOTE: This is all inconsistent at times, so these might not model
        # the data the best. Try changing some things up to see if the model
        # can be created better
    # have found most success w/ 3 Convolutions, each 64, (3,3)
        # with pooling (2,2), batch size 40 for 200 epochs
    # also success w/ 3 conv, 64/64/128 (3,3), pooling (2,2), batch size 40/50,
        # for 100 epochs

    # Define the model
    # Began with 2 conv/pooling layers
    model = tf.keras.models.Sequential([
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(80, 60, 1)),
        tf.keras.layers.MaxPooling2D(2,2),                     # was(2,2)
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),   # was 64, (3,3)
        tf.keras.layers.MaxPooling2D(2,2),                      # was (2,2)
        # third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  # was 64, (3,3)
        tf.keras.layers.MaxPooling2D(2,2),                      # was (2,2)
        # adding another conv for fun
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),     # 512 best, then 256
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.summary()
    
    # optimizer was 'adam'
    # learning_rate = 1e-5
    optimizer = 'adam'
    
    model.compile(optimizer = optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])      
  
    return model

# ----------------------------------------------------------------------------

# Splitting data into training and validation sets
split_data(ZERO_SOURCE_DIR, TRAINING_ZERO_DIR, VALIDATION_ZERO_DIR, split_size)
split_data(ONE_SOURCE_DIR, TRAINING_ONE_DIR, VALIDATION_ONE_DIR, split_size)
split_data(TWO_SOURCE_DIR, TRAINING_TWO_DIR, VALIDATION_TWO_DIR, split_size)
split_data(THREE_SOURCE_DIR, TRAINING_THREE_DIR, VALIDATION_THREE_DIR, split_size)
split_data(FOUR_SOURCE_DIR, TRAINING_FOUR_DIR, VALIDATION_FOUR_DIR, split_size)

# print("There are " + str(len(os.listdir(TRAINING_ZERO_DIR))) + " images with zero people for training")
# print("There are " + str(len(os.listdir(VALIDATION_ZERO_DIR))) + " images with zero people for validation")
# print("There are " + str(len(os.listdir(TRAINING_ONE_DIR))) + " images with one person for training")
# print("There are " + str(len(os.listdir(VALIDATION_ONE_DIR))) + " images with one person for validation")
# print("There are " + str(len(os.listdir(TRAINING_TWO_DIR))) + " images with two people for training")
# print("There are " + str(len(os.listdir(VALIDATION_TWO_DIR))) + " images with two people for validation")
# print("There are " + str(len(os.listdir(TRAINING_THREE_DIR))) + " images with three people for training")
# print("There are " + str(len(os.listdir(VALIDATION_THREE_DIR))) + " images with three people for validation")
# print("There are " + str(len(os.listdir(TRAINING_FOUR_DIR))) + " images with four people for training")
# print("There are " + str(len(os.listdir(VALIDATION_FOUR_DIR))) + " images with four people for validation")

# -- Move the random selection of images into training and vaildation csv's --
# print(os.listdir(TRAINING_DIR))
# print(os.listdir(VALIDATION_DIR))

# ---------- creating csv files to use in training/testing process -----------
training = []
# print(os.listdir(TRAINING_DIR))
for directory in os.listdir(TRAINING_DIR):
    # print(directory)
    if directory == 'zero':
        dir_zero = os.path.join(TRAINING_DIR, 'zero')
        zero_val = [0]
        zero_val = np.array(zero_val)
        for img_name in os.listdir(dir_zero):
            # print(img_name)
            file = os.path.join(dir_zero, img_name)
            zero_data = readpgm(file)
            total_arr = hstack((zero_val, zero_data[0]))
            training.append(total_arr)
    if directory == 'one':
        dir_one = os.path.join(TRAINING_DIR, 'one')
        one_val = [1]
        one_val = np.array(one_val)
        for img_name in os.listdir(dir_one):
            # print(img_name)
            file = os.path.join(dir_one, img_name)
            one_data = readpgm(file)
            total_arr = hstack((one_val, one_data[0]))
            training.append(total_arr)
    if directory == 'two':
        dir_two = os.path.join(TRAINING_DIR, 'two')
        two_val = [2]
        two_val = np.array(two_val)
        for img_name in os.listdir(dir_two):
            # print(img_name)
            file = os.path.join(dir_two, img_name)
            two_data = readpgm(file)
            total_arr = hstack((two_val, two_data[0]))
            training.append(total_arr)
    if directory == 'three':
        dir_three = os.path.join(TRAINING_DIR, 'three')
        three_val = [3]
        three_val = np.array(three_val)
        for img_name in os.listdir(dir_three):
            # print(img_name)
            file = os.path.join(dir_three, img_name)
            three_data = readpgm(file)
            total_arr = hstack((three_val, three_data[0]))
            training.append(total_arr)
    if directory == 'four':
        dir_four = os.path.join(TRAINING_DIR, 'four')
        four_val = [4]
        four_val = np.array(four_val)
        for img_name in os.listdir(dir_four):
            # print(img_name)
            file = os.path.join(dir_four, img_name)
            four_data = readpgm(file)
            total_arr = hstack((four_val, four_data[0]))
            training.append(total_arr)
                
np.savetxt("training_data.csv", training, delimiter = ',')

validation = []
# print(os.listdir(VALIDATION_DIR))
for directory in os.listdir(VALIDATION_DIR):
    # print(directory)
    if directory == 'zero':
        dir_zero = os.path.join(VALIDATION_DIR, 'zero')
        zero_val = [0]
        zero_val = np.array(zero_val)
        for img_name in os.listdir(dir_zero):
            # print(img_name)
            file = os.path.join(dir_zero, img_name)
            zero_data = readpgm(file)
            total_arr = hstack((zero_val, zero_data[0]))
            validation.append(total_arr)
    if directory == 'one':
        dir_one = os.path.join(VALIDATION_DIR, 'one')
        one_val = [1]
        one_val = np.array(one_val)
        for img_name in os.listdir(dir_one):
            # print(img_name)
            file = os.path.join(dir_one, img_name)
            one_data = readpgm(file)
            total_arr = hstack((one_val, one_data[0]))
            validation.append(total_arr)
    if directory == 'two':
        dir_two = os.path.join(VALIDATION_DIR, 'two')
        two_val = [2]
        two_val = np.array(two_val)
        for img_name in os.listdir(dir_two):
            # print(img_name)
            file = os.path.join(dir_two, img_name)
            two_data = readpgm(file)
            total_arr = hstack((two_val, two_data[0]))
            validation.append(total_arr)
    if directory == 'three':
        dir_three = os.path.join(VALIDATION_DIR, 'three')
        three_val = [3]
        three_val = np.array(three_val)
        for img_name in os.listdir(dir_three):
            # print(img_name)
            file = os.path.join(dir_three, img_name)
            three_data = readpgm(file)
            total_arr = hstack((three_val, three_data[0]))
            validation.append(total_arr)
    if directory == 'four':
        dir_four = os.path.join(VALIDATION_DIR, 'four')
        four_val = [4]
        four_val = np.array(four_val)
        for img_name in os.listdir(dir_four):
            # print(img_name)
            file = os.path.join(dir_four, img_name)
            four_data = readpgm(file)
            total_arr = hstack((four_val, four_data[0]))
            validation.append(total_arr)
                
np.savetxt("validation_data.csv", validation, delimiter = ',')

# ----------------------------------------------------------------------------

TRAINING_FILE = os.path.join(os.getcwd(), 'training_data.csv')
VALIDATION_FILE = os.path.join(os.getcwd(), 'validation_data.csv')

# ----------------- test parse_data_from_input function ----------------------
training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

# print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
# print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
# print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
# print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")

# ------------- plotting random sample of 10 training images -----------------
# plot_categories(training_images, training_labels)

# ------------------------ Test your generators ------------------------------
train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)

# print(f"Images of training generator have shape: {train_generator.x.shape}")
# print(f"Labels of training generator have shape: {train_generator.y.shape}")
# print(f"Images of validation generator have shape: {validation_generator.x.shape}")
# print(f"Labels of validation generator have shape: {validation_generator.y.shape}")

# --------------------------- Save your model --------------------------------
model = create_model()

# -------------------------- Train your model --------------------------------
history = model.fit(train_generator,
                    epochs=200,             # overfit w/ 100, maybe try around 60
                    validation_data=validation_generator)

# --- Plot the chart for accuracy and loss on both training and validation ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# -------------------------- CODE THAT IS UNUSED ------------------------------
# # Just hardcoding it to make it work, still trying to get it to work other way
# training = []
# # training images
# dir_zero = os.path.join(TRAINING_DIR, 'zero')
# zero_val = [0]
# zero_val = np.array(zero_val)
# for img_name in os.listdir(dir_zero):
#     file = os.path.join(dir_zero, img_name)
#     zero_data = readpgm(file)
#     total_arr = hstack((zero_val, zero_data[0]))
#     training.append(total_arr)
# dir_one = os.path.join(TRAINING_DIR, 'one')
# one_val = [1]
# one_val = np.array(one_val)
# for img_name in os.listdir(dir_one):
#     file = os.path.join(dir_one, img_name)
#     one_data = readpgm(file)
#     total_arr = hstack((one_val, one_data[0]))
#     training.append(total_arr)
# dir_two = os.path.join(TRAINING_DIR, 'two')
# two_val = [2]
# two_val = np.array(two_val)
# for img_name in os.listdir(dir_two):
#     file = os.path.join(dir_two, img_name)
#     two_data = readpgm(file)
#     total_arr = hstack((two_val, two_data[0]))
#     training.append(total_arr)
# dir_three = os.path.join(TRAINING_DIR, 'three')
# three_val = [3]
# three_val = np.array(three_val)
# for img_name in os.listdir(dir_three):
#     file = os.path.join(dir_three, img_name)
#     three_data = readpgm(file)
#     total_arr = hstack((three_val, three_data[0]))
#     training.append(total_arr)
# dir_four = os.path.join(TRAINING_DIR, 'four')
# four_val = [4]
# four_val = np.array(four_val)
# for img_name in os.listdir(dir_four):
#     file = os.path.join(dir_four, img_name)
#     four_data = readpgm(file)
#     total_arr = hstack((four_val, four_data[0]))
#     training.append(total_arr)
    
# np.savetxt("training_data_hardcoded.csv", training, delimiter = ',')

# validation = []
# # validation images
# dir_zero = os.path.join(VALIDATION_DIR, 'zero')
# zero_val = [0]
# zero_val = np.array(zero_val)
# for img_name in os.listdir(dir_zero):
#     file = os.path.join(dir_zero, img_name)
#     zero_data = readpgm(file)
#     total_arr = hstack((zero_val, zero_data[0]))
#     validation.append(total_arr)
# dir_one = os.path.join(VALIDATION_DIR, 'one')
# one_val = [1]
# one_val = np.array(one_val)
# for img_name in os.listdir(dir_one):
#     file = os.path.join(dir_one, img_name)
#     one_data = readpgm(file)
#     total_arr = hstack((one_val, one_data[0]))
#     validation.append(total_arr)
# dir_two = os.path.join(VALIDATION_DIR, 'two')
# two_val = [2]
# two_val = np.array(two_val)
# for img_name in os.listdir(dir_two):
#     file = os.path.join(dir_two, img_name)
#     two_data = readpgm(file)
#     total_arr = hstack((two_val, two_data[0]))
#     validation.append(total_arr)
# dir_three = os.path.join(VALIDATION_DIR, 'three')
# three_val = [3]
# three_val = np.array(three_val)
# for img_name in os.listdir(dir_three):
#     file = os.path.join(dir_three, img_name)
#     three_data = readpgm(file)
#     total_arr = hstack((three_val, three_data[0]))
#     validation.append(total_arr)
# dir_four = os.path.join(VALIDATION_DIR, 'four')
# four_val = [4]
# four_val = np.array(four_val)
# for img_name in os.listdir(dir_four):
#     file = os.path.join(dir_four, img_name)
#     four_data = readpgm(file)
#     total_arr = hstack((four_val, four_data[0]))
#     validation.append(total_arr)
    
# np.savetxt("validation_data_hardcoded.csv", validation, delimiter = ',')