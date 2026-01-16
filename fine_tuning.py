import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report

from data.cifar100 import load_and_process_data_densenet

def main():
    MODEL_NAME = "DenseNet121_FineTuning"
    INPUT_MODEL_PATH = "results\models\DenseNet121.keras"
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.00005
    PATIENCE = 5

    (X_train, Y_train), (X_test, Y_test) = load_and_process_data_densenet()

    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"Błąd: Nie znaleziono modelu {INPUT_MODEL_PATH}")
        return

    model = tf.keras.models.load_model(INPUT_MODEL_PATH)
    model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )

    final_model_path = "results/models/DenseNet121_fine-tune.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(final_model_path, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    test_metrics = model.evaluate(X_test, Y_test, verbose=0)
    test_acc = test_metrics[1]
    test_top5 = test_metrics[2]

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    summary = {
        MODEL_NAME: {
            "config": {
                "name": MODEL_NAME,
                "learning_rate": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "base_model": "DenseNet121 (fine-tuned)",
                "stage1_source": INPUT_MODEL_PATH
            },
            "metrics": {
                "accuracy": float(test_acc),
                "top_5_accuracy": float(test_top5),
                "f1_macro": float(report['macro avg']['f1-score']),
                "precision_macro": float(report['macro avg']['precision']),
                "recall_macro": float(report['macro avg']['recall']),
                "epochs_run": int(len(history.epoch))
            }
        }
    }

    ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    file_name = f"results/densenet121_finetuning_summary_{ts}.json"
    
    with open(file_name, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n>>> Sukces! Finalny model: {final_model_path}")

if __name__ == "__main__":
    main()