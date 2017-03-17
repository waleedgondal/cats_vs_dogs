# -*- coding: utf-8 -*-
"""
Utility functions.

Created on Fri Oct 11 10:24:31 2016
@author: M. Waleed Gondal
"""
import numpy as np
import tensorflow as tf
import cv2
import os
import csv
import random
    
def make_batches(path_to_csv, n_epochs ,height , width, batch_size, shuffle = True, training = True):
    
    """Make shuffled batches of images and their corresponding labels.
    
    Parameters
    ----------
    path_to_csv : string
        The path of csv file to be read.    
    n_epochs : int 
        The number of iterations for which the complete dataset is to be iterated.   
    height: int
        Height of an image
    width: int
        Width of an image
    batch_size: int
        The number of images to be stacked in one batch.
    training : Bool [default: True]
        
    Returns
    --------
    label_batch: Tensor
        A tensor of labels with shape (batchsize, label)
    image_batch: Tensor
        A tensor of image shape(batchsize, height, width, channels)"""
    
    image, label = read_data_from_csv(path_to_csv, n_epochs, height, width, training)
    if shuffle == True:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size, num_threads = 5,
                                                          capacity = 1000 + 3*batch_size, min_after_dequeue = 1000)
    else:
        image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads =5)
    return image_batch, label_batch


def read_data_from_csv(path_to_csv, n_epochs, height, width, training = True):
    
    """Read filenames and labels from a csv file. The csv file has to contain
    one file name with complete path and one integer label, separated by comma. The
    implementation follows Tensorflow input pipeline.
    
    Parameters
    ----------
    path_to_csv : string
        The path of csv file to be read.
    
    n_epochs : int 
        The number of iterations for which the complete dataset is to be iterated.
    
    height: int
        Height of an image
    width: int
        Width of an image
    training : Bool [default: True]
        
    Returns
    --------
    tf_label: Tensor
        A tensor of labels with shape (batchsize, label)
    tf_image: Tensor
        A tensor of image shape(batchsize, height, width, channels)"""
    
    csv_path = tf.train.string_input_producer([path_to_csv], num_epochs=n_epochs)
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(csv_path)
    im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])
    im_content = tf.read_file(im_name)
    tf_image = tf.image.decode_jpeg(im_content, channels=3)
    tf_image = tf.cast(tf_image, tf.float32) / 255.
    if training == True:
        tf_image = augment(tf_image)
    else:
        tf_image = tf.image.central_crop(tf_image, 0.95)
        tf_image = tf.image.per_image_whitening(tf_image)
    size = tf.cast([height, width], tf.int32)
    tf_image = tf.image.resize_images(tf_image, size)
    tf_label = tf.cast(im_label, tf.int32)
    
    return tf_image, tf_label

def count_images(file_path):
    """ Counts the number of rows (and hence the items) in a csv file"""
    file = open(file_path)
    reader = csv.reader(file)
    count = sum(1 for row in reader)
    file.close()
    return count    
    
    
def write_submission_csv(csv_path, fnames, labels, num_iter, batch_size):
    """
    csv_path: string
        Path for the csv file to be written
    fnames: ndarray
        Containing image_ids of size (num_iter, batch_size)
    image_labels: ndarray
        Containig predicted image labels of size (num_iter, batch_size)
    num_iter: int
        The number of iteration over the complete dataset. Equal to total number of batches per dataset (dataset_size/batch_size)
    batch_size: int
        Batch size for each iteration"""
    
    wfile = open(csv_path, 'wb')
    writer = csv.writer(wfile, delimiter=',')
    writer.writerow(['id','label'])
    for i in range(num_iter): # Num of batches
        for j in range(batch_size): # Iteration per batch
            prob = round(((labels[i])[j,1]), 4) #str((labels[i])[j])
            writer.writerow([(fnames[i])[j],prob])
            wfile.flush()
    wfile.close()    
    

def augment(image):
    """ Runtime Augmentations while training the network
    
    Parameters
    ----------
    image : Tensor
        A tensor of shape (height, width, channels).

    Yields
    --------
    image : A tensor
        A tensor of shape (height, width, channels)."""

    # Randomly flip the image
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    # Because these operations are not commutative, consider randomizing the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.05)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.6, upper=0.8)
    #distorted_image = tf.image.central_crop(distorted_image, 0.95)
    distorted_image = tf.image.per_image_whitening(distorted_image)
    
    return distorted_image
    



def make_split_csv(csv_path):
    """Make a CSV file with defined classes for cat and dogs"""
    wfile = open(csv_path, 'wb')
    writer = csv.writer(wfile, delimiter=',')
    for file in os.listdir('./train'):
        if file.split('.')[0] == 'cat':        
            writer.writerow(['./train/'+file,0])
        elif file.split('.')[0] == 'dog':
            writer.writerow(['./train/'+file,1])  
            
def randomize_csv(csv_path):
    """ Randomize the CSV file"""
    with open(csv_path, "rb") as source:
        reader = csv.reader(source)
        data = [(random.random(), line) for line in reader]
    data.sort()
    with open(csv_path, 'wb') as target:
        writer = csv.writer(target, delimiter=',')
        for _, line in data:
            writer.writerow(line)

"""  
# For splitting train set into 90% and 10% (2500) portions
count =0
path = 'random_complete.csv'
trainfile = open(path, "rb")
reader = csv.reader(trainfile)

wpath = 'train.csv'
wfile = open(wpath, 'wb')
writer = csv.writer(wfile, delimiter=',')

wpath1 = 'val.csv'
wfile1 = open(wpath1, 'wb')
writer1 = csv.writer(wfile1, delimiter=',')

for row in reader:
    if count <=22500:
        writer.writerow(row)
    else:
        writer1.writerow(row)
    count+=1
"""