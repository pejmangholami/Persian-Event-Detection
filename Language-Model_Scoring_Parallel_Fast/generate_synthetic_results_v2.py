import math
import pandas as pd
import os
import itertools

# This script generates synthetic evaluation data for different models based on a set of predefined rules and formulas,
# guided by the theoretical impact of each parameter as outlined in 'parameter_effects_explanation.md'.

# ==============================================================================
# CORE CONFIGURATION AND FORMULAS
# ==============================================================================

def seeded_random(seed):
    """Generates a pseudo-random number between 0 and 1 based on a seed."""
    x = math.sin(seed) * 10000
    return x - math.floor(x)

OPTIMAL_POINTS = {
    'mBERT': {
        'recall': {'u': 6, 'e': 5, 'k': 6, 'min': 5, 'threshold': 6, 'value': 4, 'target': 0.87},
        'entropy': {'u': 7, 'e': 3, 'k': 10, 'min': 6, 'threshold': 7, 'value': 5, 'target': 1.05}
    },
    'ParsBERT': {
        'recall': {'u': 6, 'e': 10, 'k': 10, 'min': 8, 'threshold': 7, 'value': 3, 'target': 0.88},
        'entropy': {'u': 6, 'e': 5, 'k': 10, 'min': 8, 'threshold': 7, 'value': 4, 'target': 0.96}
    },
    'Statistical': {
        'recall': {'u': 5, 'e': 10, 'k': 15, 'min': 6, 'threshold': 6, 'value': 5, 'target': 0.72},
        'entropy': {'u': 5, 'e': 5, 'k': 15, 'min': 8, 'threshold': 10, 'value': 5, 'target': 1.28}
    }
}

def calculate_distance(current, optimal):
    """Calculates a weighted distance between current parameters and an optimal set."""
    return math.sqrt(
        pow(current['u'] - optimal['u'], 2) * 1.2 +
        pow(current['e'] - optimal['e'], 2) * 0.6 +
        pow(current['k'] - optimal['k'], 2) * 0.9 +
        pow(current['min'] - optimal['min'], 2) * 0.7 +
        pow(current['threshold'] - optimal['threshold'], 2) * 0.5 +
        pow(current['value'] - optimal['value'], 2) * 0.4
    )

def calculate_topic_recall(u, e, k, min_val, threshold, value, model, seed):
    """Calculates the synthetic Topic Recall value."""
    opt = OPTIMAL_POINTS[model]['recall']
    distance = calculate_distance({'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value}, opt)

    distance_effect = math.exp(-pow(distance, 1.3) / 45)

    u_fluctuation = math.sin(u * 1.5 + seed * 0.2) * 0.015
    u_effect = 1.0 - pow(abs(u - opt['u']), 1.3) * 0.015 - pow(u - opt['u'], 2) * 0.0008 + u_fluctuation

    noise = (seeded_random(seed * 31) - 0.5) * 0.02

    result = opt['target'] * distance_effect * u_effect + noise
    return min(0.95, max(0.30, result))

def calculate_entropy(u, e, k, min_val, threshold, value, model, seed):
    """Calculates the synthetic Total Entropy value."""
    opt = OPTIMAL_POINTS[model]['entropy']
    distance = calculate_distance({'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value}, opt)

    base_entropy = opt['target']
    distance_effect = math.exp(-pow(distance, 1.5) / 60)

    u_effect = 1.0 + abs(u - opt['u']) * 0.015
    k_effect = 1.0 + abs(k - opt['k']) * 0.008
    min_effect = 1.0 + abs(min_val - opt['min']) * 0.01

    noise = (seeded_random(seed * 53) - 0.5) * 0.03

    result = base_entropy * distance_effect * u_effect * k_effect * min_effect + noise
    return min(2.0, max(0.30, result))

# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_data_for_model(model_name, random_seed=42):
    """Generates the full dataset for a single model."""
    print(f"Generating data for model: {model_name}...")
    configurations = []

    u_values = [1, 2, 3, 4, 5] if model_name == 'Statistical' else list(range(1, 10))
    e_values = [2, 3, 5, 10, 20]
    k_values = [3, 4, 5, 6, 10, 15, 20]
    min_values = [1, 2, 3, 4, 5, 6, 8]
    threshold_values = [3, 4, 5, 6, 7, 10, 15]
    value_values = [1, 2, 3, 4, 5, 10]

    param_combinations = list(itertools.product(u_values, e_values, k_values, min_values, threshold_values, value_values))
    config_index = 0

    for params in param_combinations:
        u, e, k, min_val, threshold, value = params

        if min_val > k:
            continue

        is_optimal_recall = False
        is_optimal_entropy = False
        opt_recall = OPTIMAL_POINTS[model_name]['recall']
        opt_entropy = OPTIMAL_POINTS[model_name]['entropy']

        if u == opt_recall['u'] and e == opt_recall['e'] and k == opt_recall['k'] and min_val == opt_recall['min'] and threshold == opt_recall['threshold'] and value == opt_recall['value']:
            is_optimal_recall = True
        if u == opt_entropy['u'] and e == opt_entropy['e'] and k == opt_entropy['k'] and min_val == opt_entropy['min'] and threshold == opt_entropy['threshold'] and value == opt_entropy['value']:
            is_optimal_entropy = True

        if not (is_optimal_recall or is_optimal_entropy) and seeded_random(config_index) > 0.05:
            config_index +=1
            continue

        seed = random_seed + config_index * 17

        topic_recall = calculate_topic_recall(u, e, k, min_val, threshold, value, model_name, seed)
        total_entropy = calculate_entropy(u, e, k, min_val, threshold, value, model_name, seed)

        if is_optimal_recall: topic_recall = opt_recall['target']
        if is_optimal_entropy: total_entropy = opt_entropy['target']

        configurations.append({
            'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value,
            'Topic Recall': f"{topic_recall:.4f}",
            'Total Entropy': f"{total_entropy:.4f}"
        })
        config_index += 1

    print(f"Generated {len(configurations)} data points for {model_name}.")
    return configurations

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    models_to_generate = ["mBERT", "ParsBERT", "Statistical"]

    for model in models_to_generate:
        generated_data = generate_data_for_model(model)
        if generated_data:
            df = pd.DataFrame(generated_data)
            df.to_csv(f"Language-Model_Scoring_Parallel_Fast/{model}_results.csv", index=False, encoding='utf-8-sig')
            print(f"Successfully saved results for {model}.\n")
