import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report

from data.cifar100 import load_and_process_data_densenet

from models import create_densenet121

def main():
    MODEL_NAME = "DenseNet121_Transfer"
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    PATIENCE = 6

    (X_train, Y_train), (X_test, Y_test) = load_and_process_data_densenet()

    model = create_densenet121(MODEL_NAME, X_train.shape[1:], 100)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )

    os.makedirs("results/models", exist_ok=True)
    model_save_path = "results/models/DenseNet121.keras"
    
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

    eval_results = model.evaluate(X_test, Y_test, verbose=0)
    test_loss = eval_results[0]
    test_acc = eval_results[1]
    test_top5 = eval_results[2]

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    summary = {
        MODEL_NAME: {
            "config": {
                "name": MODEL_NAME,
                "dense_units": 256,
                "dropout_rate": 0.3,
                "optimizer_type": "Adam",
                "learning_rate": LR,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "augmentation": True,
                "base_model": "DenseNet121"
            },
            "metrics": {
                "accuracy": float(test_acc),
                "top_5_accuracy": float(test_top5),
                "f1_macro": float(report['macro avg']['f1-score']),
                "precision_macro": float(report['macro avg']['precision']),
                "recall_macro": float(report['macro avg']['recall']),
                "test_loss": float(test_loss),
                "epochs_run": int(len(history.epoch))
            }
        }
    }

    ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    file_name = f"results/densenet121_summary_{ts}.json"
    
    with open(file_name, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n>>> Gotowe. Model: {model_save_path}, Wyniki: {file_name}")

if __name__ == "__main__":
    main()