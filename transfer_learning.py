import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report

from data.cifar100 import load_and_process_data_mobilenet
from models import create_mobilenet_v2

def main():
    MODEL_NAME = "MobileNetV2_Transfer"
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.0005
    PATIENCE = 10

    (X_train, Y_train), (X_test, Y_test) = load_and_process_data_mobilenet()

    model = create_mobilenet_v2(MODEL_NAME, X_train.shape[1:], 100)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )

    os.makedirs("results/models", exist_ok=True)
    model_save_path = "results/models/MobileNet_v2.keras"
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(Y_test, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    summary = {
        MODEL_NAME: {
            "config": {
                "name": MODEL_NAME,
                "dense_units": 256,
                "dropout_rate": 0.4,
                "optimizer_type": "Adam",
                "learning_rate": LR,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "augmentation": True,
                "base_model": "MobileNetV2 (frozen)"
            },
            "metrics": {
                "accuracy": float(max(history.history['val_accuracy'])),
                "top_5_accuracy": float(max(history.history['val_top_5_accuracy'])),
                "f1_macro": float(report['macro avg']['f1-score']),
                "precision_macro": float(report['macro avg']['precision']),
                "recall_macro": float(report['macro avg']['recall']),
                "epochs_run": int(len(history.epoch))
            }
        }
    }

    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    file_name = f"results/mobilenet_v2_summary_{ts}.json"
    
    with open(file_name, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n>>> Gotowe. Model: {model_save_path}, Wyniki: {file_name}")

if __name__ == "__main__":
    main()