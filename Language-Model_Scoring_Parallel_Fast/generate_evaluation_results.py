import math
import pandas as pd
import os

# This script generates synthetic evaluation data for different models based on a set of predefined rules and formulas.
# The core logic is a port of a TypeScript/React component (`evaluation_table_generator.tsx`).

# ==============================================================================
# CORE CONFIGURATION AND FORMULAS
# ==============================================================================

# A simple seeded random function to generate reproducible noise.
def seeded_random(seed):
    """Generates a pseudo-random number between 0 and 1 based on a seed."""
    x = math.sin(seed) * 10000
    return x - math.floor(x)

# Optimal points for Recall and Entropy for each model. These are the target values
# that the generation logic will gravitate towards.
OPTIMAL_POINTS = {
    'mBERT': {
        'recall': {'u': 6, 'e': 5, 'k': 6, 'kmin': 5, 'threshold': 6, 'value': 4, 'target': 0.87},
        'entropy': {'u': 7, 'e': 3, 'k': 10, 'kmin': 6, 'threshold': 7, 'value': 5, 'target': 1.05}
    },
    'ParsBERT': {
        'recall': {'u': 6, 'e': 10, 'k': 10, 'kmin': 8, 'threshold': 7, 'value': 3, 'target': 0.88},
        'entropy': {'u': 6, 'e': 5, 'k': 10, 'kmin': 8, 'threshold': 7, 'value': 4, 'target': 0.96}
    },
    'Statistical': {
        'recall': {'u': 5, 'e': 10, 'k': 15, 'kmin': 6, 'threshold': 6, 'value': 5, 'target': 0.72},
        'entropy': {'u': 5, 'e': 5, 'k': 15, 'kmin': 8, 'threshold': 10, 'value': 5, 'target': 1.28}
    }
}

def calculate_distance(current, optimal):
    """Calculates a weighted distance between current parameters and an optimal set."""
    return math.sqrt(
        pow(current['u'] - optimal['u'], 2) * 1.2 +
        pow(current['e'] - optimal['e'], 2) * 0.6 +
        pow(current['k'] - optimal['k'], 2) * 0.9 +
        pow(current['kmin'] - optimal['kmin'], 2) * 0.7 +
        pow(current['threshold'] - optimal['threshold'], 2) * 0.5 +
        pow(current['value'] - optimal['value'], 2) * 0.4
    )

def calculate_topic_recall(u, e, k, kmin, threshold, value, model, seed):
    """Calculates the synthetic Topic Recall value with more natural fluctuations."""
    opt = OPTIMAL_POINTS[model]['recall']

    distance = calculate_distance({'u': u, 'e': e, 'k': k, 'kmin': kmin, 'threshold': threshold, 'value': value}, opt)

    # The core effect is an exponential decay based on the overall distance from the optimal point.
    distance_effect = math.exp(-pow(distance, 1.5) / 35)

    # --- Parameter-specific effects with naturalistic fluctuations ---

    # U-parameter: A gentle curve around the optimum, with small, periodic dips.
    u_diff = u - opt['u']
    u_base_effect = 1.0 - pow(abs(u_diff), 1.4) * 0.017 - pow(u_diff, 2) * 0.001
    u_fluctuation = math.sin(u * 1.5 + seed * 0.2) * 0.009 # Small dips and bumps
    u_effect = u_base_effect + u_fluctuation

    # E-parameter: Similar gentle curve.
    e_diff = e - opt['e']
    e_effect = 1.0 - abs(e_diff) * 0.011 - pow(e_diff, 2) * 0.0007

    # K and Kmin parameters: Effect based on their ratio and distance.
    k_ratio = abs(k - opt['k']) / 10
    kmin_ratio = abs(kmin - opt['kmin']) / 5
    k_effect = 1.0 - (k_ratio + kmin_ratio) * 0.025

    # Threshold and Value: Linear effects.
    threshold_effect = 1.0 - abs(threshold - opt['threshold']) * 0.008
    value_effect = 1.0 - abs(value - opt['value']) * 0.006

    # --- Noise and final calculation ---
    correlation_factors = {'mBERT': 0.025, 'ParsBERT': 0.020, 'Statistical': 0.035} # Reduced noise for stability

    # Reproducible noise to ensure results are the same every time
    independent_noise = (seeded_random(seed * 31) - 0.5) * correlation_factors[model] + \
                        (seeded_random(seed * 41) - 0.5) * correlation_factors[model] * 0.6

    result = opt['target'] * distance_effect * u_effect * e_effect * k_effect * threshold_effect * value_effect + independent_noise

    # Clamp the result to a realistic range.
    return min(0.95, max(0.60, result))

def calculate_class_entropy(u, e, k, kmin, threshold, value, model, seed):
    """Calculates the synthetic Class Entropy value."""
    opt = OPTIMAL_POINTS[model]['entropy']

    distance = calculate_distance({'u': u, 'e': e, 'k': k, 'kmin': kmin, 'threshold': threshold, 'value': value}, opt)

    base_entropy = opt['target'] * 0.45
    distance_effect = 1.0 + (distance / 15)

    e_diff = abs(e - opt['e'])
    e_effect = 1.0 + e_diff * 0.022 - e_diff * 0.001

    k_diff = abs(k - opt['k'])
    k_effect = 1.0 + k_diff * 0.018

    u_effect = 1.0 + abs(u - opt['u']) * 0.015

    tv_interaction = 1.0 + (threshold - opt['threshold']) * (value - opt['value']) / 400

    correlation_factors = {'mBERT': 0.055, 'ParsBERT': 0.045, 'Statistical': 0.070}
    independent_noise = (seeded_random(seed * 53) - 0.5) * correlation_factors[model]

    result = base_entropy * distance_effect * e_effect * k_effect * u_effect * tv_interaction + independent_noise

    return min(1.8, max(0.25, result))

def calculate_cluster_entropy(u, e, k, kmin, threshold, value, class_entropy, model, seed):
    """Calculates the synthetic Cluster Entropy, with a more nuanced inverse relation to Class Entropy."""
    opt = OPTIMAL_POINTS[model]['entropy']

    base_entropy = opt['target'] * 0.55
    distance = calculate_distance({'u': u, 'e': e, 'k': k, 'kmin': kmin, 'threshold': threshold, 'value': value}, opt)

    # --- Inverse Relationship with Class Entropy ---
    # The relationship is primarily inverse, but its strength is moderated by other parameters (u, k)
    # to prevent a perfectly mirrored curve.
    normalized_class_entropy = class_entropy / 1.8
    base_inverse_effect = pow(1.0 - normalized_class_entropy, 0.6)

    # Moderating factor to make the inverse relationship less direct.
    moderation_strength = 0.6 + (seeded_random(u * k) - 0.5) * 0.4 # Varies between 0.4 and 0.8
    inverse_effect = (base_inverse_effect * moderation_strength) + ((1.0 - base_inverse_effect) * (1 - moderation_strength))
    inverse_effect = inverse_effect * 1.4 + 0.3


    distance_effect = 1.0 + (distance / 18)

    kmin_diff = opt['kmin'] - kmin
    kmin_effect = 1.0 + kmin_diff * 0.012

    u_effect = 1.0 + pow(abs(u - opt['u']), 1.2) * 0.01

    kt_diff = abs((k - opt['k']) - (threshold - opt['threshold']))
    kt_interaction = 1.0 + kt_diff * 0.008

    correlation_factors = {'mBERT': 0.060, 'ParsBERT': 0.048, 'Statistical': 0.075}
    independent_noise = (seeded_random(seed * 67) - 0.5) * correlation_factors[model]

    result = base_entropy * inverse_effect * distance_effect * kmin_effect * u_effect * kt_interaction + independent_noise

    return min(1.8, max(0.20, result))

def calculate_total_entropy(class_entropy, cluster_entropy, u, k):
    """Calculates the total entropy as a weighted average of class and cluster entropy."""
    class_weight = 0.48 + (u - 5) * 0.015
    cluster_weight = 1 - class_weight

    total = class_weight * class_entropy + cluster_weight * cluster_entropy

    # Special adjustment for ParsBERT to help it reach its higher optimal entropy target.
    # At its optimal entropy point, ParsBERT's naturally generated value is ~0.53, but the
    # user's target is ~0.96. This targeted multiplier lifts the value to the desired range
    # without disrupting the surrounding data points.
    if 'ParsBERT' in model and u == OPTIMAL_POINTS['ParsBERT']['entropy']['u']:
         total *= 1.8

    # Special adjustment for mBERT to help it reach its higher optimal entropy target.
    if 'mBERT' in model and u == OPTIMAL_POINTS['mBERT']['entropy']['u']:
        total *= 1.8


    return min(2.0, max(0.30, total))

def calculate_newsworthiness(topic_recall, total_entropy, u, e, k, model, seed):
    """Calculates the synthetic Newsworthiness score."""
    correlation_weights = {
        'mBERT': {'recall': 0.38, 'entropy': 0.28, 'independent': 0.34},
        'ParsBERT': {'recall': 0.52, 'entropy': 0.36, 'independent': 0.12},
        'Statistical': {'recall': 0.28, 'entropy': 0.18, 'independent': 0.54}
    }
    weights = correlation_weights[model]

    recall_score = topic_recall
    entropy_score = max(0, 1.0 - (total_entropy / 2.0))

    u_factor = math.sin((u / 9) * math.pi * 1.5) * 0.25
    e_factor = math.cos((e / 20) * math.pi * 1.8) * 0.20
    k_factor = math.exp(-abs(k - 10) / 12) * 0.22

    complex_interaction = math.tanh((u * e) / 100) * 0.15
    independent_score = 0.5 + u_factor + e_factor + k_factor + complex_interaction

    noise_strength = {'mBERT': 0.08, 'ParsBERT': 0.05, 'Statistical': 0.12}
    strong_noise = (seeded_random(seed * 79) - 0.5) * noise_strength[model] + \
                   (seeded_random(seed * 83) - 0.5) * noise_strength[model] * 0.7

    result = weights['recall'] * recall_score + \
             weights['entropy'] * entropy_score + \
             weights['independent'] * independent_score + \
             strong_noise

    return min(0.96, max(0.40, result))

def calculate_number_of_events(u, e, k, threshold, value, model, seed):
    """Calculates the synthetic NumberOfEvents."""
    base_events = {'mBERT': 160, 'ParsBERT': 165, 'Statistical': 140}
    base = base_events[model]

    threshold_effect = pow(threshold / 10, 1.3)
    value_effect = math.sqrt(value) / 2.5
    u_effect = 1.2 - abs(u - 5) * 0.06
    e_effect = math.log(1 + e) / 3
    k_effect = 1.0 - (k - 12) * 0.01

    noise = pow(seeded_random(seed * 13), 0.5) * 40 - 20

    result = base * threshold_effect * value_effect * u_effect * e_effect * k_effect + noise
    return max(80, round(result))

# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_data_for_model(model_name, random_seed=42):
    """Generates the full dataset for a single model."""
    print(f"Generating data for model: {model_name}...")
    configurations = []

    # Define parameter ranges
    u_values = [1, 2, 3, 4, 5] if model_name == 'Statistical' else [1, 2, 3, 4, 5, 6, 7, 8, 9]
    e_values = [2, 3, 5, 10, 20]
    k_values = [3, 4, 5, 6, 10, 15, 20]
    kmin_values = [1, 2, 3, 4, 5, 6, 8]
    threshold_values = [3, 4, 5, 6, 7, 10, 15]
    value_values = [1, 2, 3, 4, 5, 10]

    config_index = 0

    for u in u_values:
        for e in e_values:
            for k in k_values:
                for kmin in kmin_values:
                    if kmin > k:
                        continue

                    for threshold in threshold_values:
                        for value in value_values:
                            # This complex condition block replicates the `isValid` logic from the original JS code
                            # to generate a curated, sparse set of parameter combinations.
                            # --- Start of validity conditions ---
                            is_valid = False

                            # Rule 1: Optimal point for mBERT Entropy
                            if u == 7 and e == 3 and k == 10 and kmin == 6 and threshold == 7 and value == 5: is_valid = True
                            # Rule 2: Optimal point for ParsBERT Entropy
                            if u == 6 and e == 5 and k == 10 and kmin == 8 and threshold == 7 and value == 4: is_valid = True
                            # Rule 3: Optimal point for ParsBERT Recall
                            if u == 6 and e == 10 and k == 10 and kmin == 8 and threshold == 7 and value == 3: is_valid = True


                            is_valid = is_valid or (
                                (u == 1 and e == 10 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u == 1 and e == 20 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u in [1, 2] and e in [2, 3, 5] and k == 10 and kmin == 8 and threshold == 6 and value == 3) or
                                (u == 3 and e == 10 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u == 3 and e == 20 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u in [3, 5, 7] and e in [2, 3, 5] and k == 10 and kmin in [4, 8] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u in [4] and e in [2, 3, 5] and k == 10 and kmin == 8 and threshold == 6 and value == 3) or
                                (u == 5 and e == 10 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u == 5 and e == 20 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u == 6 and e in [2, 3, 5] and k == 10 and kmin == 8 and threshold in [3, 4, 5, 6, 7] and value in [1, 2, 3, 4]) or
                                (u == 6 and e == 5 and k in [3, 4, 5, 6] and kmin in [1, 2, 3, 4, 5, 6] and threshold == 6 and value == 3) or
                                (u == 7 and e == 10 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u == 7 and e == 20 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u in [8] and e in [2, 3, 5] and k == 10 and kmin == 8 and threshold == 6 and value == 3) or
                                (u == 9 and e == 10 and k in [10, 15, 20] and kmin in [4, 6] and threshold in [5, 10, 15] and value in [5, 10]) or
                                (u == 9 and e in [2, 3, 5] and k == 10 and kmin == 8 and threshold == 6 and value == 3) or
                                (u == 9 and e == 20 and k == 10 and kmin == 4 and threshold == 10 and value == 10) or
                                (u in [2, 4, 6, 8] and e in [10, 20] and k == 10 and kmin == 8 and threshold == 6 and value == 3)
                            )
                            # --- End of validity conditions ---

                            if not is_valid:
                                continue

                            seed = random_seed + config_index * 17

                            topic_recall = calculate_topic_recall(u, e, k, kmin, threshold, value, model_name, seed)
                            class_entropy = calculate_class_entropy(u, e, k, kmin, threshold, value, model_name, seed)
                            cluster_entropy = calculate_cluster_entropy(u, e, k, kmin, threshold, value, class_entropy, model_name, seed)
                            total_entropy = calculate_total_entropy(class_entropy, cluster_entropy, u, k)
                            newsworthiness = calculate_newsworthiness(topic_recall, total_entropy, u, e, k, model_name, seed)
                            num_events = calculate_number_of_events(u, e, k, threshold, value, model_name, seed)

                            configurations.append({
                                'u': u, 'e': e, 'k': k, 'min': kmin, 'threshold': threshold, 'value': value,
                                'Topic Recall': f"{topic_recall:.4f}",
                                'Class Entropy': f"{class_entropy:.4f}",
                                'Cluster Entropy': f"{cluster_entropy:.4f}",
                                'Total Entropy': f"{total_entropy:.4f}",
                                'Mean_Newsworthiness': f"{newsworthiness:.4f}",
                                'NumberOfEvents': num_events
                            })

                            config_index += 1
    print(f"Generated {len(configurations)} data points for {model_name}.")
    return configurations

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Ensure pandas is installed
    try:
        import pandas as pd
    except ImportError:
        print("Pandas is not installed. Please install it using: pip install pandas")
        exit()

    models_to_generate = ["mBERT", "ParsBERT", "Statistical"]
    all_dataframes = {}

    for model in models_to_generate:
        generated_data = generate_data_for_model(model)

        if generated_data:
            df = pd.DataFrame(generated_data)
            all_dataframes[model] = df # Store the dataframe for later use

            # Define output filename
            output_filename = f"{model}_results.csv"
            output_path = os.path.join(os.path.dirname(__file__), output_filename)

            # Save to CSV
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Successfully saved results to {output_path}\n")
        else:
            print(f"No data generated for {model}. Skipping file creation.")

    # ==============================================================================
    # CORRELATION ANALYSIS
    # ==============================================================================
    print("Calculating correlations between Topic Recall and Total Entropy...")
    correlation_results = []

    for model_name, df in all_dataframes.items():
        # Convert columns to numeric types for correlation calculation
        df['Topic Recall'] = pd.to_numeric(df['Topic Recall'])
        df['Total Entropy'] = pd.to_numeric(df['Total Entropy'])

        # Calculate Pearson correlation
        correlation = df['Topic Recall'].corr(df['Total Entropy'], method='pearson')

        correlation_results.append({
            'Model': model_name,
            'Pearson Correlation': f"{correlation:.4f}",
            'Correlation Percentage': f"{correlation:.2%}"
        })

    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        correlation_output_path = os.path.join(os.path.dirname(__file__), "EntropyRecall_correlation.csv")
        correlation_df.to_csv(correlation_output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully saved correlation results to {correlation_output_path}")
    else:
        print("No data was generated, so no correlations were calculated.")
