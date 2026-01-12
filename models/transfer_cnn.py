import tensorflow as tf
from tensorflow.keras import layers, models

def create_transfer_cnn(input_shape=(32, 32, 3), num_classes=100, fine_tune_at=None):
    """
    Tworzy model do Transfer Learningu na zbiorze CIFAR-100.
    
    Args:
        input_shape: Kształt obrazu wejściowego (domyślnie 32x32x3).
        num_classes: Liczba klas (100 dla CIFAR-100).
        fine_tune_at: Indeks warstwy, od której zaczynamy odmrażanie wag (opcjonalnie).
    """
    
    # 1. Warstwa wejściowa
    inputs = layers.Input(shape=input_shape)
    
    # 2. Resizing - Modele pre-trenowane na ImageNet (jak EfficientNet) 
    # najlepiej działają na większych obrazach niż 32x32.
    # Skalujemy je np. do 72x72 lub 224x224.
    x = layers.Resizing(72, 72)(inputs)
    
    # 3. Model bazowy (Pre-trained)
    # Wybieramy EfficientNetB0 bez "topu" (warstw klasyfikacyjnych)
    base_model = tf.keras.applications.EfficientNetB0(
        input_tensor=x,
        include_top=False,
        weights='imagenet'
    )
    
    # Zamrażamy model bazowy na początek
    base_model.trainable = False
    
    # Opcjonalne: Odmrażanie konkretnych warstw do Fine-tuningu
    if fine_tune_at is not None:
        base_model.trainable = True
        # Zamrażamy wszystkie warstwy PRZED indeksem fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # 4. Budowa nowej głowy (Top) modelu
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Redukcja overfittingu
    
    # Warstwa wyjściowa - 100 neuronów z aktywacją Softmax
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    return model