import json
import os

from clustering.kmeans_module import (
    DEFAULT_GA_EARLY_MUTATION_RATE,
    DEFAULT_GA_GENERATIONS,
    DEFAULT_GA_HYPERPARAM_SOURCE,
    DEFAULT_GA_LATE_MUTATION_RATE,
    DEFAULT_GA_MAX_STAGNANT,
    DEFAULT_GA_MID_MUTATION_RATE,
    DEFAULT_GA_MUTATION_RATE,
    DEFAULT_GA_POP_SIZE,
)


ACTIVE_TRAINING_CONFIG_PATH = os.path.join("configs", "active_training_config.json")


def get_builtin_training_config():
    return {
        "population_size": DEFAULT_GA_POP_SIZE,
        "generations": DEFAULT_GA_GENERATIONS,
        "mutation_rate": DEFAULT_GA_MUTATION_RATE,
        "early_mutation_rate": DEFAULT_GA_EARLY_MUTATION_RATE,
        "mid_mutation_rate": DEFAULT_GA_MID_MUTATION_RATE,
        "late_mutation_rate": DEFAULT_GA_LATE_MUTATION_RATE,
        "max_stagnant": DEFAULT_GA_MAX_STAGNANT,
        "hyperparameter_source": DEFAULT_GA_HYPERPARAM_SOURCE,
    }


def _normalize_training_config(raw_config):
    defaults = get_builtin_training_config()
    config = defaults.copy()
    if not isinstance(raw_config, dict):
        return config

    for key in config:
        if key not in raw_config or raw_config[key] in (None, ""):
            continue
        config[key] = raw_config[key]

    config["population_size"] = int(config["population_size"])
    config["generations"] = int(config["generations"])
    config["max_stagnant"] = int(config["max_stagnant"])
    config["mutation_rate"] = float(config["mutation_rate"])
    config["early_mutation_rate"] = float(config["early_mutation_rate"])
    config["mid_mutation_rate"] = float(config["mid_mutation_rate"])
    config["late_mutation_rate"] = float(config["late_mutation_rate"])
    config["hyperparameter_source"] = str(config["hyperparameter_source"])
    return config


def load_active_training_config():
    if not os.path.exists(ACTIVE_TRAINING_CONFIG_PATH):
        return get_builtin_training_config()

    try:
        with open(ACTIVE_TRAINING_CONFIG_PATH, encoding="utf-8") as f:
            raw_config = json.load(f)
        return _normalize_training_config(raw_config)
    except Exception:
        return get_builtin_training_config()


def save_active_training_config(config):
    normalized = _normalize_training_config(config)
    os.makedirs(os.path.dirname(ACTIVE_TRAINING_CONFIG_PATH), exist_ok=True)
    with open(ACTIVE_TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized
