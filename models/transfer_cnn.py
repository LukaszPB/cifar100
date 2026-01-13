from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_mobilenet_v2(model_name, input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # augmentation layer
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    # up sampling
    x = layers.UpSampling2D(size=(2, 2))(x)

    # mobilenet_v2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=x,
        pooling='avg'
    )

    # weight freezing
    base_model.trainable = False 

    # fully connected layer
    x = base_model.output
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name=model_name)