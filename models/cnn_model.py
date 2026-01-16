from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, RandomFlip, RandomRotation, 
    RandomTranslation, RandomZoom
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

OPTIMIZER_MAP = {
    'Adam': Adam,
    'SGD': SGD,
    'RMSprop': RMSprop,
}

def create_cnn(
    model_name,
    input_shape,
    num_classes,
    num_conv_blocks=2,
    initial_filters=32,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    dense_units=128,
    dropout_rate=0.2,
    augmentation=False
):
    model = Sequential(name=model_name)
    model.add(Input(shape=input_shape, name='input_layer'))

    print("!!!!!!!!!!!AAAAAAAAAAAAAAAAAAAAAAA!!!!!!!!!!")
    print(num_conv_blocks, dropout_rate)
    
    # augmentation layer
    if augmentation:
        model.add(RandomFlip("horizontal", name="aug_flip"))
        model.add(RandomRotation(0.1, name="aug_rotation"))
        model.add(RandomTranslation(height_factor=0.1, width_factor=0.1, name="aug_translation"))
        model.add(RandomZoom(0.1, name="aug_zoom"))

    current_filters = initial_filters

    # convolutional layer
    for i in range(num_conv_blocks):
        model.add(Conv2D(filters=current_filters, 
                         kernel_size=kernel_size, 
                         activation='relu', 
                         padding='same',
                         use_bias=False,
                         name=f'conv_{i+1}_{current_filters}f'))
        
        model.add(BatchNormalization(name=f'bn_{i+1}'))
        model.add(MaxPooling2D(pool_size=pool_size, name=f'pool_{i+1}'))
        current_filters *= 2
        
    # fully connected layer
    model.add(Flatten(name='flatten'))
    model.add(Dense(dense_units, activation='relu', name='dense_hidden'))
    model.add(Dropout(dropout_rate, name='dropout'))
    model.add(Dense(num_classes, activation='softmax', name='output_softmax'))

    return model