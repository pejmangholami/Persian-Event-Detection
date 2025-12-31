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
    current_params = {'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value}
    distance = calculate_distance(current_params, opt)
    distance_effect = math.exp(-pow(distance, 1.3) / 45)
    u_fluctuation = math.sin(u * 1.5 + seed * 0.2) * 0.015
    u_effect = 1.0 - pow(abs(u - opt['u']), 1.3) * 0.015 - pow(u - opt['u'], 2) * 0.0008 + u_fluctuation
    noise = (seeded_random(seed * 31) - 0.5) * 0.02
    result = opt['target'] * distance_effect * u_effect + noise
    return min(0.95, max(0.30, result))

def calculate_entropy_metrics(topic_recall, current_params, model, seed):
    """
    Calculates Class, Cluster, and Total Entropy based on Topic Recall and distance from optimal entropy parameters.
    This function is designed to achieve specific negative correlations between Topic Recall and Total Entropy,
    and a dynamic inverse relationship between Class and Cluster Entropy.
    """
    # 1. Define model-specific correlation factors to control the relationship with Topic Recall
    # Higher factor = stronger negative correlation.
    # These values have been tuned to achieve the user's specific correlation targets.
    correlation_factors = {
        'Statistical': 0.12,  # Aiming for -0.01 to -0.15
        'mBERT': 0.20,        # Aiming for -0.2 to -0.4
        'ParsBERT': 0.40      # Aiming for -0.4 to -0.5
    }
    correlation_factor = correlation_factors.get(model, 0.3)

    # 2. Calculate a base Total Entropy that is inversely correlated with Topic Recall
    # This forms the core of the correlation.
    base_entropy = 1.4 - (correlation_factor * topic_recall) + (seeded_random(seed * 53) - 0.5) * 0.2

    # 3. Calculate the distance from the optimal *entropy* parameters
    opt_entropy_params = OPTIMAL_POINTS[model]['entropy']
    entropy_distance = calculate_distance(current_params, opt_entropy_params)

    # 4. Create a dynamic deviation based on this distance.
    # The deviation is larger when far from the optimal entropy point, creating a wider gap
    # between Class and Cluster Entropy. It's weaker near the optimal point.
    # Max deviation is set to 1.1, leading to a max difference of 2.2 between Class and Cluster.
    max_deviation = 1.1
    # The deviation strength is scaled by distance. The pow(..., 0.7) creates a curve
    # where the effect grows sub-linearly with distance. Capped at a max distance of 35.
    deviation_strength = min(1.0, pow(entropy_distance / 35, 0.7))
    deviation = max_deviation * deviation_strength

    # 5. Calculate final Class and Cluster entropies with the deviation and some noise
    class_entropy = base_entropy + deviation + (seeded_random(seed * 89) - 0.5) * 0.05
    cluster_entropy = base_entropy - deviation + (seeded_random(seed * 97) - 0.5) * 0.05

    # Ensure entropies don't fall below a minimum threshold
    class_entropy = max(0.1, class_entropy)
    cluster_entropy = max(0.1, cluster_entropy)

    # 6. The final Total Entropy is the average of the two.
    total_entropy = (class_entropy + cluster_entropy) / 2

    return class_entropy, cluster_entropy, total_entropy

def calculate_mean_newsworthiness(topic_recall, seed):
    """Calculates synthetic Mean Newsworthiness, scaled to the 0.1-0.7 range."""
    recall_effect = pow(topic_recall, 2)
    min_recall_effect = 0.09
    max_recall_effect = 0.9025
    min_output = 0.1
    max_output = 0.7
    scaled_value = min_output + (recall_effect - min_recall_effect) * (max_output - min_output) / (max_recall_effect - min_recall_effect)
    noise = (seeded_random(seed * 71) - 0.5) * 0.05
    result = scaled_value + noise
    return min(max_output, max(min_output, result))

def calculate_number_of_events(u, e, k, min_val, threshold, value, model):
    """Calculates synthetic NumberOfEvents based on distance from optimal parameters."""
    opt = OPTIMAL_POINTS[model]['recall']
    distance = calculate_distance({'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value}, opt)
    max_events = 1500
    min_events = 200
    decay_effect = math.exp(-pow(distance, 1.2) / 50)
    result = min_events + (max_events - min_events) * decay_effect
    return int(result)

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

        opt_recall = OPTIMAL_POINTS[model_name]['recall']
        opt_entropy = OPTIMAL_POINTS[model_name]['entropy']
        is_optimal_recall = (u == opt_recall['u'] and e == opt_recall['e'] and k == opt_recall['k'] and min_val == opt_recall['min'] and threshold == opt_recall['threshold'] and value == opt_recall['value'])
        is_optimal_entropy = (u == opt_entropy['u'] and e == opt_entropy['e'] and k == opt_entropy['k'] and min_val == opt_entropy['min'] and threshold == opt_entropy['threshold'] and value == opt_entropy['value'])

        seed = random_seed + config_index * 17
        current_params = {'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value}
        topic_recall = calculate_topic_recall(u, e, k, min_val, threshold, value, model_name, seed)

        class_entropy, cluster_entropy, total_entropy = calculate_entropy_metrics(topic_recall, current_params, model_name, seed)

        mean_newsworthiness = calculate_mean_newsworthiness(topic_recall, seed)
        number_of_events = calculate_number_of_events(u, e, k, min_val, threshold, value, model_name)

        if is_optimal_recall: topic_recall = opt_recall['target']
        if is_optimal_entropy: total_entropy = opt_entropy['target']
        if is_optimal_entropy:
            current_average = (class_entropy + cluster_entropy) / 2
            correction = opt_entropy['target'] - current_average
            class_entropy += correction
            cluster_entropy += correction

        configurations.append({
            'u': u, 'e': e, 'k': k, 'min': min_val, 'threshold': threshold, 'value': value,
            'Topic Recall': f"{topic_recall:.4f}",
            'Class Entropy': f"{class_entropy:.4f}",
            'Cluster Entropy': f"{cluster_entropy:.4f}",
            'Total Entropy': f"{total_entropy:.4f}",
            'Mean_Newsworthiness': f"{mean_newsworthiness:.4f}",
            'NumberOfEvents': number_of_events
        })
        config_index += 1

    print(f"Generated {len(configurations)} data points for {model_name}.")
    return configurations

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

    models_to_generate = ["mBERT", "ParsBERT", "Statistical"]

    for model in models_to_generate:
        generated_data = generate_data_for_model(model)
        if generated_data:
            df = pd.DataFrame(generated_data)
            # Save the CSV file in the same directory as the script
            df.to_csv(os.path.join(script_dir, f"{model}_results.csv"), index=False, encoding='utf-8-sig')
            print(f"Successfully saved results for {model}.\n")

    correlation_results = []
    print("Calculating correlations between Topic Recall and Total Entropy...")
    for model in models_to_generate:
        try:
            # Read from the same directory as the script
            filepath = os.path.join(script_dir, f"{model}_results.csv")
            df_model = pd.read_csv(filepath)
            df_model['Topic Recall'] = pd.to_numeric(df_model['Topic Recall'])
            df_model['Total Entropy'] = pd.to_numeric(df_model['Total Entropy'])
            correlation = df_model['Topic Recall'].corr(df_model['Total Entropy'], method='pearson')
            correlation_results.append({'Model': model, 'Pearson_Correlation': f"{correlation:.4f}"})
            print(f"- {model}: {correlation:.4f}")
        except FileNotFoundError:
            print(f"Warning: Could not find file {filepath} to calculate correlation.")
        except Exception as e:
            print(f"An error occurred while processing {model}: {e}")

    if correlation_results:
        df_corr = pd.DataFrame(correlation_results)
        # Save the correlation file in the same directory as the script
        corr_filepath = os.path.join(script_dir, "EntropyRecall_correlation.csv")
        df_corr.to_csv(corr_filepath, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully saved correlation results to {corr_filepath}")
