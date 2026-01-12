# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# import os

# OPTIMIZER_MAP = {
#     'Adam': Adam,
#     'SGD': SGD,
#     'RMSprop': RMSprop,
# }

# def run_kfold_validation(X_train, Y_train, create_function, model_params):
#     # Wymiary wejścia i liczba klas
#     INPUT_SHAPE = X_train.shape[1:]
#     NUM_CLASSES = Y_train.shape[1]

#     Y_train_labels = np.argmax(Y_train, axis=1)

#     # Pobranie parametrów
#     k_folds = model_params.pop('k_folds')
#     epochs = model_params.pop('epochs')
#     batch_size = model_params.pop('batch_size')

#     optimizer_type = model_params.pop('optimizer_type', 'Adam')
#     learning_rate = model_params.pop('learning_rate', 0.001)

#     experiment_name = model_params.pop('name')

#     # K-Fold setup
#     skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

#     fold_accuracies = []
#     fold_losses = []

#     print(f"\n--- Rozpoczynam Walidację Krzyżową dla {experiment_name} (K={k_folds}) ---")

#     # --- Pętla K-Fold ---
#     for fold_n, (train_index, val_index) in enumerate(skf.split(X_train, Y_train_labels)):
#         print(f"\n[FOLD {fold_n+1}/{k_folds}]")

#         if optimizer_type in OPTIMIZER_MAP:
#             optimizer = OPTIMIZER_MAP[optimizer_type](learning_rate=learning_rate)
#         else:
#             raise ValueError(f"Nieznany optymalizator: {optimizer_type}")

#         # Utworzenie modelu
#         model = create_function(experiment_name, INPUT_SHAPE, NUM_CLASSES, **model_params)
#         model.build(input_shape=(None,) + INPUT_SHAPE)

#         model.compile(optimizer=optimizer,
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

#         # Podział danych folda
#         X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
#         Y_fold_train, Y_fold_val = Y_train[train_index], Y_train[val_index]

#         # Callbacks
#         log_dir = os.path.join("results", "tensorboard_logs", experiment_name, f"fold_{fold_n+1}")
#         checkpoint_dir = os.path.join("results","models",experiment_name)
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         checkpoint_path = os.path.join(checkpoint_dir,f"{experiment_name}_fold_{fold_n+1}_best.weights.h5")

#         callbacks_list = [
#             TensorBoard(log_dir=log_dir, histogram_freq=1),
#             EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1),
#             ModelCheckpoint(checkpoint_path,
#                             monitor='val_accuracy',
#                             save_best_only=True,
#                             save_weights_only=True,
#                             mode='max',
#                             verbose=0)
#         ]

#         # Trening folda
#         model.fit(
#             X_fold_train, Y_fold_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             validation_data=(X_fold_val, Y_fold_val),
#             callbacks=callbacks_list,
#             verbose=2
#         )

#         # Po treningu — Wczytaj najlepsze zapisane wagi
#         model.load_weights(checkpoint_path)

#         # Ocena na walidacji folda
#         loss, accuracy = model.evaluate(X_fold_val, Y_fold_val, verbose=0)

#         fold_losses.append(loss)
#         fold_accuracies.append(accuracy)

#         print(f"FOLD {fold_n+1} Zakończony. Najlepsza dokładność walidacyjna: {accuracy*100:.2f}%")

#         # Czyszczenie sesji TF po każdym foldzie
#         tf.keras.backend.clear_session()
#         del model

#     # Wyniki końcowe
#     avg_accuracy = np.mean(fold_accuracies)
#     avg_loss = np.mean(fold_losses)

#     print("\n----------------------------------------------------")
#     print(f"ŚREDNIA DOKŁADNOŚĆ DLA {k_folds} FOLDÓW: {avg_accuracy*100:.2f}%")
#     print(f"ŚREDNIA STRATA DLA {k_folds} FOLDÓW: {avg_loss:.4f}")
#     print("----------------------------------------------------")

#     return {
#         'avg_accuracy': avg_accuracy,
#         'avg_loss': avg_loss,
#         'fold_accuracies': fold_accuracies,
#         'fold_losses': fold_losses
#     }

import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

OPTIMIZER_MAP = {
    'Adam': Adam,
    'SGD': SGD,
    'RMSprop': RMSprop,
}

def run_kfold_validation(X_train, Y_train, create_function, model_params):
    INPUT_SHAPE = X_train.shape[1:]
    NUM_CLASSES = Y_train.shape[1]
    Y_train_labels = np.argmax(Y_train, axis=1)

    k_folds = model_params.pop('k_folds')
    epochs = model_params.pop('epochs')
    batch_size = model_params.pop('batch_size')
    optimizer_type = model_params.pop('optimizer_type', 'Adam')
    learning_rate = model_params.pop('learning_rate', 0.001)
    experiment_name = model_params.pop('name')

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Inicjalizacja tablicy na predykcje Out-of-Fold
    oof_predictions = np.zeros((X_train.shape[0], NUM_CLASSES))
    
    fold_accuracies = []
    fold_losses = []

    print(f"\n--- Rozpoczynam Walidację Krzyżową OOF dla {experiment_name} (K={k_folds}) ---")

    for fold_n, (train_index, val_index) in enumerate(skf.split(X_train, Y_train_labels)):
        print(f"\n[FOLD {fold_n+1}/{k_folds}]")

        optimizer = OPTIMIZER_MAP[optimizer_type](learning_rate=learning_rate)
        model = create_function(experiment_name, INPUT_SHAPE, NUM_CLASSES, **model_params)
        model.build(input_shape=(None,) + INPUT_SHAPE)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        Y_fold_train, Y_fold_val = Y_train[train_index], Y_train[val_index]

        log_dir = os.path.join("results", "tensorboard_logs", experiment_name, f"fold_{fold_n+1}")
        checkpoint_dir = os.path.join("results", "models", experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_fold_{fold_n+1}_best.weights.h5")

        callbacks_list = [
            TensorBoard(log_dir=log_dir, histogram_freq=1),
            EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, 
                            save_weights_only=True, mode='max', verbose=0)
        ]

        model.fit(
            X_fold_train, Y_fold_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_fold_val, Y_fold_val),
            callbacks=callbacks_list,
            verbose=2
        )

        model.load_weights(checkpoint_path)

        # Generowanie predykcji dla folda walidacyjnego
        fold_preds = model.predict(X_fold_val, verbose=0)
        oof_predictions[val_index] = fold_preds

        # Metryki pomocnicze dla bieżącego folda
        loss, accuracy = model.evaluate(X_fold_val, Y_fold_val, verbose=0)
        fold_losses.append(loss)
        fold_accuracies.append(accuracy)

        tf.keras.backend.clear_session()
        del model

    # Obliczenia końcowe na podstawie pełnego wektora OOF
    final_oof_labels = np.argmax(oof_predictions, axis=1)
    global_accuracy = accuracy_score(Y_train_labels, final_oof_labels)
    
    print("\n" + "="*50)
    print(f"WYNIKI KOŃCOWE OOF DLA: {experiment_name}")
    print(f"GLOBALNA DOKŁADNOŚĆ (OOF): {global_accuracy*100:.2f}%")
    print("\nRaport Klasyfikacji:")
    print(classification_report(Y_train_labels, final_oof_labels))
    print("\nMacierz Pomyłek:")
    print(confusion_matrix(Y_train_labels, final_oof_labels))
    print("="*50)

    return {
        # 'oof_predictions': oof_predictions,
        'global_accuracy': global_accuracy,
        'fold_accuracies': fold_accuracies,
        'fold_losses': fold_losses
    }