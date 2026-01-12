import os
import sys
import json
import yaml
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
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

# --- Funkcje Logiki Treningowej ---

def train_fold(fold_idx, data, params, paths):
    """Przeprowadza trening dla jednego folda i zwraca rozszerzone metryki."""
    X_train, Y_train, X_val, Y_val = data
    
    # Budowa modelu
    model = create_cnn(
        params['name'], 
        X_train.shape[1:], 
        Y_train.shape[1], 
        **params['model_arch']
    )
    
    optimizer = tf.keras.optimizers.get(params['optimizer_type'])
    optimizer.learning_rate = params['learning_rate']
    
    # Dodanie Top-5 Accuracy do kompilacji
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy', 
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    # Callbacks
    checkpoint_path = os.path.join(paths['checkpoint_dir'], f"fold_{fold_idx}.weights.h5")
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=params['patience'], restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    ]

    # Trening
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # --- Ewaluacja po treningu (na najlepszych wagach) ---
    # Pobieramy predykcje, aby wyliczyć Precision/Recall/F1
    val_preds = model.predict(X_val, verbose=0)
    val_labels_pred = np.argmax(val_preds, axis=1)
    val_labels_true = np.argmax(Y_val, axis=1)
    epochs_done = len(history.epoch)

    # Generowanie raportu klasyfikacji (średnia macro jest kluczowa dla CIFAR-100)
    report = classification_report(val_labels_true, val_labels_pred, output_dict=True, zero_division=0)

    # Zbieranie metryk
    metrics = {
        "accuracy": float(max(history.history['val_accuracy'])),
        "top_5_accuracy": float(max(history.history['val_top_5_accuracy'])),
        "f1_macro": float(report['macro avg']['f1-score']),
        "precision_macro": float(report['macro avg']['precision']),
        "recall_macro": float(report['macro avg']['recall']),
        "epochs_run": int(epochs_done)
    }

    print(f"   Fold {fold_idx} zakończony | Acc: {metrics['accuracy']:.4f} | Top-5: {metrics['top_5_accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")
    
    tf.keras.backend.clear_session()
    return metrics

def run_experiment(config, X, Y):
    """Zarządza pełnym procesem K-Fold i agreguje rozszerzone wyniki."""
    name = config.get('name', 'exp')
    k = config.get('k_folds', 5)
    
    train_params = {
        'name': name,
        'epochs': config.get('epochs', 50),
        'patience': config.get('patience', 10),
        'batch_size': config.get('batch_size', 32),
        'optimizer_type': config.get('optimizer_type', 'Adam'),
        'learning_rate': config.get('learning_rate', 0.001),
        'model_arch': {k: v for k, v in config.items() if k not in 
                        ['name', 'k_folds', 'epochs', 'patience', 'batch_size', 'optimizer_type', 'learning_rate']}
    }

    paths = {'checkpoint_dir': os.path.join("results", "models", name)}
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    y_labels = np.argmax(Y, axis=1)
    
    fold_results = []
    print(f"\n>>> Eksperyment: {name} | K={k} | Params: {train_params['model_arch']}")

    for i, (t_idx, v_idx) in enumerate(skf.split(X, y_labels)):
        data = (X[t_idx], Y[t_idx], X[v_idx], Y[v_idx])
        res = train_fold(i + 1, data, train_params, paths)
        fold_results.append(res)

    # --- Agregacja wyników ze wszystkich foldów ---
    summary_metrics = {}
    metric_keys = fold_results[0].keys()
    
    for key in metric_keys:
        vals = [f[key] for f in fold_results]
        summary_metrics[f"mean_{key}"] = float(np.mean(vals))
        summary_metrics[f"std_{key}"] = float(np.std(vals))

    return summary_metrics

# --- Główny Nurt Programu ---

def main():
    # 1. Załaduj konfigurację
    config_path = "config/experiments.yaml"
    if not os.path.exists(config_path):
        print(f"Błąd: Nie znaleziono pliku {config_path}")
        return

    with open(config_path, 'r') as f:
        experiments = yaml.safe_load(f).get('experiments', [])

    # 2. Załaduj dane raz
    (X_train, Y_train), _ = load_and_process_data()

    all_summary = {}

    # 3. Pętla eksperymentów
    for exp_config in experiments:
        res = run_experiment(exp_config, X_train, Y_train)
        
        all_summary[exp_config['name']] = {
            "config": exp_config,
            "metrics": res
        }

    # 4. Zapisz wyniki (Poprawiony format daty dla Windows)
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S") # Usunięto dwukropki
    file_name = f"results/basic_cnn_summary_{ts}.json"
    
    with open(file_name, 'w') as f:
        json.dump(all_summary, f, indent=4)
    
    print(f"\nProces zakończony. Wyniki zapisano w: {file_name}")

if __name__ == '__main__':
    main()