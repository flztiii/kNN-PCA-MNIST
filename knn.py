# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:34:03 2019

@author: flztiii
"""

import os
import struct
import numpy as np
import time
import random
from pca import PCA

def load_mnist():
    root_path = './mnist'

    train_labels_path = os.path.join(root_path, 'train-labels.idx1-ubyte')
    train_images_path = os.path.join(root_path, 'train-images.idx3-ubyte')

    test_labels_path = os.path.join(root_path, 't10k-labels.idx1-ubyte')
    test_images_path = os.path.join(root_path, 't10k-images.idx3-ubyte')

    with open(train_labels_path, 'rb') as lpath:
        # '>' denotes bigedian
        # 'I' denotes unsigned char
        magic, n = struct.unpack('>II', lpath.read(8))
        #loaded = np.fromfile(lpath, dtype = np.uint8)
        train_labels = np.fromfile(lpath, dtype = np.uint8).astype(np.float)

    with open(train_images_path, 'rb') as ipath:
        magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(train_images_path, dtype = np.uint8)
        # images start from the 16th bytes
        train_images = loaded[16:].reshape(len(train_labels), 784).astype(np.float)

    with open(test_labels_path, 'rb') as lpath:
        # '>' denotes bigedian
        # 'I' denotes unsigned char
        magic, n = struct.unpack('>II', lpath.read(8))
        #loaded = np.fromfile(lpath, dtype = np.uint8)
        test_labels = np.fromfile(lpath, dtype = np.uint8).astype(np.float)

    with open(test_images_path, 'rb') as ipath:
        magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(test_images_path, dtype = np.uint8)
        # images start from the 16th bytes
        test_images = loaded[16:].reshape(len(test_labels), 784)

    return train_images, train_labels, test_images, test_labels

def test_mnist_data():
    train_images, train_labels, test_images, test_labels = load_mnist()
    fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
    ax =ax.flatten()
    for i in range(10):
        img = train_images[i][:].reshape(28, 28)
        ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
        print('corresponding labels = %d' %train_labels[i])
        
class KNNClassifier:
    def __init__(self, train_data, train_label):
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
        #print(self.train_data.shape)
        #print(self.train_label.shape)
    
    def distance(self, x):
        points = np.zeros_like(self.train_data)
        points[:] = x
        minusSquare = (self.train_data - points)**2
        EuclideanDistances = np.sqrt(minusSquare.sum(axis=1))
        return EuclideanDistances
    
    def test(self, input_data, k):
        distances = self.distance(input_data)
        sorted_index = np.argsort(distances)
        k_nearest_label = []
        for i in range(0, k):
            k_nearest_label.append(self.train_label[sorted_index[i]])
        label_count = [0]*10
        for label in k_nearest_label:
            label_count[int(label)] = label_count[int(label)] + 1
        result = np.argmax(label_count)
        return result
    
    def evaluate(self, test_data, test_label, k):
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        correct_count = 0
        all_count = len(test_label)
        for i in range(0, all_count):
            result = self.test(test_data[i], k)
            if result == test_label[i]:
                correct_count = correct_count + 1
            #print(result,"done")
        correct_rate = float(correct_count)/float(all_count)
        return correct_rate

if __name__ == '__main__':
    root_path = "./results/"
    train_images, train_labels, test_images, test_labels = load_mnist()
    repeat_times = 10
    test_num = 300
    # knn without using pca
    for eigen_len in range(100, train_images.shape[1], 100):
        cut_train_images = train_images[:,:eigen_len]
        cut_test_images = test_images[:,:eigen_len]
        classifier = KNNClassifier(cut_train_images, train_labels)
        record_file = open(root_path+"raw/"+str(eigen_len)+".txt", 'a+')
        for k in range(2, 20, 2):
            start = time.clock()
            avg_correct_rate = 0.0
            for j in range(0, repeat_times):
                test_data = []
                test_label = []
                while len(test_data) < test_num:
                    i = random.randint(0, len(cut_test_images)-1)
                    test_data.append(cut_test_images[i])
                    test_label.append(test_labels[i])
                correct_rate = classifier.evaluate(test_data, test_label, k)
                avg_correct_rate = avg_correct_rate + correct_rate
            avg_correct_rate = float(avg_correct_rate)/float(repeat_times)
            end = time.clock()
            print("When k is", k,", correct rate is", avg_correct_rate, "time consuming is", end-start, "egien length is", eigen_len)
            msg = str(k)+","+str(avg_correct_rate)+","+str(end-start)+"\n"
            record_file.write(msg)
        record_file.close()

    # knn using pca
    for eigen_len in range(100, train_images.shape[1], 100):
        pca_train_images,trans = PCA(train_images, eigen_len)
        pca_test_images = np.dot(test_images, trans)
        pca_classifier = KNNClassifier(pca_train_images, train_labels)
        record_file = open(root_path+"pca/"+str(eigen_len)+".txt", 'a+')
        for k in range(2, 20, 2):
            start = time.clock()
            avg_correct_rate = 0.0
            for j in range(0, repeat_times):
                test_data = []
                test_label = []
                while len(test_data) < test_num:
                    i = random.randint(0, len(pca_test_images)-1)
                    test_data.append(pca_test_images[i])
                    test_label.append(test_labels[i])
                correct_rate = pca_classifier.evaluate(test_data, test_label, k)
                avg_correct_rate = avg_correct_rate + correct_rate
            avg_correct_rate = float(avg_correct_rate)/float(repeat_times)
            end = time.clock()
            print("When k is", k,", correct rate is", avg_correct_rate, "time consuming is", end-start, "pca egien length is", pca_train_images.shape[1])
            msg = str(k)+","+str(avg_correct_rate)+","+str(end-start)+"\n"
            record_file.write(msg)
        record_file.close()