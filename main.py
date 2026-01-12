import numpy as np
import os
import sys
import json
from datetime import datetime
import yaml
import random
import tensorflow as tf

from data.cifar100 import load_and_process_data
from utils.kfold import run_kfold_validation
from models import create_cnn

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def load_config(filepath="config/experiments.yaml"):
    if not os.path.exists(filepath):
        print(f"Błąd: Nie znaleziono pliku konfiguracyjnego pod ścieżką: {filepath}")
        sys.exit(1)
        
    print(f"Ładowanie konfiguracji z {filepath}...")
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
        
    return config['experiments']


def run_single_experiment(experiment_config, X_train, Y_train):
    exp = experiment_config.copy()
    exp_name = exp.get("name")

    print(f"\nRozpoczynam Eksperyment: {exp_name}")

    k_folds = exp.pop("k_folds")
    epochs = exp.pop("epochs")
    batch_size = exp.pop("batch_size")
    optimizer_type = exp.pop("optimizer_type")
    learning_rate = exp.pop("learning_rate")

    model_params = exp

    results = run_kfold_validation(
        X_train=X_train,
        Y_train=Y_train,
        create_function=create_cnn,
        model_params={
            "name": exp_name,
            "k_folds": k_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            **model_params
        }
    )

    return {
        "exp_name": exp_name,
        "config": {
            "optimizer": optimizer_type,
            "learning_rate": learning_rate,
            "k_folds": k_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            **model_params
        },
        "results": results
    }


def run_all_experiments():
    config_list = load_config()

    (X_train, Y_train), (_, _) = load_and_process_data()

    all_results = {}

    for config in config_list:
        result = run_single_experiment(config, X_train, Y_train) 
        
        exp_name = result['exp_name']
        all_results[exp_name] = {
            "config": result['config'],
            "results": result['results']
        }

    if not os.path.exists("results"):
        os.makedirs("results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join("results", f"experiment_summary_{timestamp}.json")

    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nZapisano podsumowanie wszystkich eksperymentów do: {output_filename}")

if __name__ == '__main__':
    run_all_experiments()
