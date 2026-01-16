import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def load_and_process_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train, axis=(0, 1, 2))
    std = np.std(x_train, axis=(0, 1, 2))
    
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    num_classes = 100
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def load_and_process_data_mobilenet():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = mobilenet_preprocess(x_train)
    x_test = mobilenet_preprocess(x_test)

    num_classes = 100
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def load_and_process_data_densenet():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = densenet_preprocess(x_train)
    x_test = densenet_preprocess(x_test)

    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    return (x_train, y_train), (x_test, y_test)