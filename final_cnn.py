import os
import json
import yaml
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data.cifar100 import load_and_process_data
from models import create_cnn

# --- Konfiguracja Środowiska ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def train_and_evaluate(params, train_data, test_data):
    """Trenuje jeden model i przeprowadza ewaluację na zbiorze testowym."""
    X_train_full, Y_train_full = train_data
    X_test, Y_test = test_data
    
    # 1. Budowa modelu
    model = create_cnn(
        params['name'], 
        X_train_full.shape[1:], 
        Y_train_full.shape[1], 
        **params.get('model_arch', {})
    )
    
    optimizer = tf.keras.optimizers.get(params.get('optimizer_type', 'Adam'))
    optimizer.learning_rate = params.get('learning_rate', 0.001)
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    # 2. Przygotowanie ścieżek i Callbacks
    checkpoint_dir = os.path.join("results", "models", params['name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

    callbacks = [
        # Monitorujemy stratę walidacyjną lub celność walidacyjną
        EarlyStopping(monitor='val_accuracy', patience=params.get('patience', 10), restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    ]

    # 3. Trening (z wydzieleniem 10% danych na walidację dla EarlyStopping)
    print(f"\n>>> Rozpoczynanie treningu modelu: {params['name']}")
    history = model.fit(
        X_train_full, Y_train_full,
        validation_split=0.1,  # Wydzielenie części danych na walidację
        epochs=params.get('epochs', 50),
        batch_size=params.get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )

    # 4. Ewaluacja końcowa na zbiorze TESTOWYM
    print("\n>>> Ewaluacja na zbiorze testowym...")
    test_preds = model.predict(X_test, verbose=0)
    test_labels_pred = np.argmax(test_preds, axis=1)
    test_labels_true = np.argmax(Y_test, axis=1)

    report = classification_report(test_labels_true, test_labels_pred, output_dict=True, zero_division=0)

    # Zbieranie metryk końcowych
    final_metrics = {
        "test_accuracy": float(report['accuracy']),
        "test_f1_macro": float(report['macro avg']['f1-score']),
        "test_precision_macro": float(report['macro avg']['precision']),
        "test_recall_macro": float(report['macro avg']['recall']),
        "epochs_completed": len(history.epoch),
        "best_val_accuracy": float(max(history.history['val_accuracy']))
    }

    print("-" * 30)
    print(f"WYNIKI TESTOWE ({params['name']}):")
    print(f"Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"F1-Macro: {final_metrics['test_f1_macro']:.4f}")
    print("-" * 30)

    return final_metrics

def main():
    # 1. Załaduj konfigurację (zakładamy, że YAML zawiera parametry bezpośrednio)
    config_path = "config/final_experiments.yaml"
    if not os.path.exists(config_path):
        print(f"Błąd: Nie znaleziono pliku {config_path}")
        return

    with open(config_path, 'r') as f:
        # Jeśli plik ma strukturę: { experiments: [ {params} ] }, bierzemy pierwszy element
        # Jeśli plik to po prostu słownik z parametrami, bierzemy go bezpośrednio
        config_data = yaml.safe_load(f)
        if 'experiments' in config_data:
            params = config_data['experiments'][0]
        else:
            params = config_data

    # 2. Załaduj dane
    (X_train, Y_train), (X_test, Y_test) = load_and_process_data()

    # 3. Uruchom proces
    results = train_and_evaluate(params, (X_train, Y_train), (X_test, Y_test))

    # 4. Zapisz wyniki
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    file_name = f"results/final_cnn_summary_{ts}.json"
    
    output = {
        "experiment_name": params['name'],
        "config": params,
        "results": results,
        "timestamp": ts
    }

    with open(file_name, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"\nProces zakończony. Wyniki testowe zapisano w: {file_name}")

if __name__ == '__main__':
    main()