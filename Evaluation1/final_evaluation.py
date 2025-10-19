import pandas as pd
import numpy as np
import os
import re
from EvaluateFungtional import PrepareData_new, Entropy, TopicEvaluation

def get_params_from_filename(filename):
    """Extracts parameters from the filename."""
    params = {}
    parts = filename.replace('.xlsx', '').split('_')
    for part in parts:
        if '-' in part:
            key_value = part.split('-')
            if len(key_value) == 2:
                key, value = key_value
                params[key] = value
    return params

def run_evaluation():
    """
    Processes the evaluation files and generates a final report.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_results_path = os.path.join(script_dir, "ProcessedResults")
    golden_standard_path = os.path.join(script_dir, "GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx")
    output_filepath = os.path.join(script_dir, "Final_Evaluation_Report.xlsx")

    golden_standard_df = pd.read_excel(golden_standard_path)

    results = []

    for filename in os.listdir(processed_results_path):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(processed_results_path, filename)
            system_result_df = pd.read_excel(filepath)

            params = get_params_from_filename(filename)

            # Prepare data for evaluation
            GS, SR = PrepareData_new(golden_standard_df, system_result_df)

            #GS[0]: Samples
            #GS[1]: Classes_Number
            #GS[2]: Classes_String
            #GS[3]: TitleOfEachLabel_GS

            #SR[0]: Clusters
            #SR[1]: TitleOfEachLabel_SR

            # Calculate metrics
            ClusterEntropy = Entropy(GS[0].copy(), SR[0].copy(), GS[1].copy())
            ClassEntropy = Entropy(GS[0].copy(), GS[1].copy(), SR[0].copy())
            w1=1
            w2=1
            TotalEntropy = ((w1*ClusterEntropy)+(w2*ClassEntropy))/(w1+w2)

            TopicPrecision, TopicRecall, TopicF1, KeywordPrecision, KeywordRecall, KeywordF1 = TopicEvaluation(GS[3].copy(),SR[1].copy())


            results.append({
                "step_time_hours": params.get("step"),
                "u": params.get("u"),
                "e": params.get("e"),
                "k": params.get("k"),
                "min": params.get("k_min"),
                "tereshold": params.get("t"),
                "value": params.get("kv"),
                "Topic Precision": TopicPrecision,
                "Topic Recall": TopicRecall,
                "Topic F1": TopicF1,
                "Keyword Precision": KeywordPrecision,
                "Keyword Recall": KeywordRecall,
                "Keyword F1": KeywordF1,
                "Class Entropy": ClassEntropy,
                "Cluster Entropy": ClusterEntropy,
                "Total Entropy": TotalEntropy
            })

    final_df = pd.DataFrame(results)
    final_df.to_excel(output_filepath, index=False)
    print(f"Final evaluation report saved to {output_filepath}")

if __name__ == "__main__":
    run_evaluation()
