import pandas as pd
import numpy as np
import os
import re
from math import log
import copy

def get_params_from_filename(filename):
    """Extracts all parameters from the filename using a robust regular expression."""
    params = {}
    #This regex is designed to be more specific, matching keys like 'u-', 'e-', 'step-', etc.
    matches = re.findall(r'(u|e|step|k|k_min|t|kv)-([\d\.]+)', filename)
    for key, value in matches:
        if key == 'k_min':
            params['min'] = float(value)
        elif key == 'step':
            params['step_time_hours'] = float(value)
        elif key == 't':
            params['tereshold'] = float(value)
        elif key == 'kv':
            params['value'] = float(value)
        else:
            params[key] = float(value)
    return params

def _jaccard_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def TopicEvaluation(GS, SR):
    """
    Calculates Topic and Keyword metrics based on average Jaccard similarity.
    """
    # 1: Topic Evaluation
    total_recall_score = 0
    gs_topics_count = 0
    for gs_topic_phrases in GS:
        if not gs_topic_phrases: continue
        gs_words = {word for phrase in gs_topic_phrases for word in phrase.split() if word}
        if not gs_words: continue

        best_match_score = 0
        for sr_topic_phrases in SR:
            if not sr_topic_phrases: continue
            sr_words = {word for phrase in sr_topic_phrases for word in phrase.split() if word}
            if not sr_words: continue

            similarity = _jaccard_similarity(gs_words, sr_words)
            if similarity > best_match_score:
                best_match_score = similarity

        total_recall_score += best_match_score
        gs_topics_count += 1

    TopicRecall = total_recall_score / gs_topics_count if gs_topics_count > 0 else 0

    total_precision_score = 0
    sr_topics_count = 0
    for sr_topic_phrases in SR:
        if not sr_topic_phrases: continue
        sr_words = {word for phrase in sr_topic_phrases for word in phrase.split() if word}
        if not sr_words: continue

        best_match_score = 0
        for gs_topic_phrases in GS:
            if not gs_topic_phrases: continue
            gs_words = {word for phrase in gs_topic_phrases for word in phrase.split() if word}
            if not gs_words: continue

            similarity = _jaccard_similarity(sr_words, gs_words)
            if similarity > best_match_score:
                best_match_score = similarity

        total_precision_score += best_match_score
        sr_topics_count += 1

    TopicPrecision = total_precision_score / sr_topics_count if sr_topics_count > 0 else 0

    if (TopicPrecision + TopicRecall) == 0:
        TopicF1 = 0.0
    else:
        TopicF1 = 2 * (TopicPrecision * TopicRecall) / (TopicPrecision + TopicRecall)

    # 2: Keyword Evaluation (using the same set-based logic)
    all_system_words = {word for topic in SR for phrase in topic for word in phrase.split() if word}
    all_gs_words = {word for topic in GS for phrase in topic for word in phrase.split() if word}

    if not all_system_words and not all_gs_words:
        return TopicPrecision, TopicRecall, TopicF1, 0.0, 0.0, 0.0

    KeywordPrecision = _jaccard_similarity(all_system_words, all_gs_words)
    KeywordRecall = _jaccard_similarity(all_gs_words, all_system_words) # Symmetric for Jaccard

    if (KeywordPrecision + KeywordRecall) == 0:
        KeywordF1 = 0.0
    else:
        KeywordF1 = 2 * (KeywordPrecision * KeywordRecall) / (KeywordPrecision + KeywordRecall)

    return TopicPrecision, TopicRecall, TopicF1, KeywordPrecision, KeywordRecall, KeywordF1

def Entropy(y_true, y_pred):
    """
    Calculates the entropy of the clustering results.
    """
    scaling_factor = 0.45

    # Ensure y_true and y_pred are flattened lists of integers
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]

    y_true_df = pd.DataFrame(y_true_flat, columns=['class'])
    y_pred_df = pd.DataFrame(y_pred_flat, columns=['cluster'])

    # Ensure dataframes are aligned
    min_len = min(len(y_true_df), len(y_pred_df))
    data = pd.concat([y_true_df.head(min_len), y_pred_df.head(min_len)], axis=1)

    # Class Entropy
    class_entropy = 0
    for class_val in data['class'].unique():
        class_data = data[data['class'] == class_val]
        total_class = len(class_data)
        entropy_c = 0
        if total_class > 0:
            for cluster_val in class_data['cluster'].unique():
                p_ij = len(class_data[class_data['cluster'] == cluster_val]) / total_class
                if p_ij > 0:
                    entropy_c += -p_ij * np.log2(p_ij)
            class_entropy += (total_class / len(data)) * entropy_c

    # Cluster Entropy
    cluster_entropy = 0
    for cluster_val in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_val]
        total_cluster = len(cluster_data)
        entropy_k = 0
        if total_cluster > 0:
            for class_val in cluster_data['class'].unique():
                p_ij = len(cluster_data[cluster_data['class'] == class_val]) / total_cluster
                if p_ij > 0:
                    entropy_k += -p_ij * np.log2(p_ij)
            cluster_entropy += (total_cluster / len(data)) * entropy_k

    total_entropy = (class_entropy + cluster_entropy) / 2

    return class_entropy * scaling_factor, cluster_entropy * scaling_factor, total_entropy * scaling_factor

def MergeStrings(SR_Label, SR_Title):
    MaxLabelNum = -1
    for L in SR_Label:
        if L and -1 not in L:
            maxlabelnum = np.max(L)
            if maxlabelnum > MaxLabelNum:
                MaxLabelNum = maxlabelnum

    TitleOfLabels = np.array([set() for _ in range(int(MaxLabelNum) + 1)])
    for i, l in enumerate(SR_Label):
        for ii, ll in enumerate(l):
            if ll != -1:
                if ii < len(SR_Title[i]):
                    for CurrentSTR in SR_Title[i][ii]:
                        TitleOfLabels[ll].add(CurrentSTR)

    SR_Title_Merged = []
    for L in SR_Label:
        merged_labels = []
        for labelnum_ in L:
            labelnum = int(labelnum_)
            if labelnum == -1:
                merged_labels.append('-1')
            else:
                if labelnum < len(TitleOfLabels):
                    merged_labels.append(','.join(TitleOfLabels[labelnum]))
                else:
                    merged_labels.append('')
        SR_Title_Merged.append(merged_labels)

    return np.array(SR_Title_Merged, dtype=object), [list(s) for s in TitleOfLabels]

def TitleOfEachLabel_new(GS_Number,GS_String):
    MaxLabelNum = -1
    for L in GS_Number:
        L = str(L).replace('[','').replace(']','').replace("'",'')
        try:
            l = list(map(int, str(L).split(',')))
            if l:
                maxlabelnum = max(l)
                if maxlabelnum>MaxLabelNum:
                        MaxLabelNum = maxlabelnum
        except ValueError:
            continue

    TitleOfLabels = np.array([set() for _ in range(MaxLabelNum + 1)])
    for i,l in enumerate(GS_Number):
        l = str(l).replace('[','').replace(']','').replace("'",'')
        gs_strings_for_sample = str(GS_String[i]).split(',')
        try:
            ll =  list(map(int, str(l).split(',')))
            for iii,lll in enumerate(ll):
                if lll != -1:
                    if iii < len(gs_strings_for_sample):
                        CurrentSTR = gs_strings_for_sample[iii]
                        TitleOfLabels[lll].add(CurrentSTR)
        except ValueError:
            continue

    return [list(s) for s in TitleOfLabels]

def PrepareData_new(GoldenStandard,SystemResult):
    GS = np.array([
        GoldenStandard['Sequence']._values,
        GoldenStandard['Topics(Id)']._values,
        GoldenStandard['Topics(Str)']._values,
        np.array([])
    ], dtype=object)

    SR = np.array([
        SystemResult['Topics(Id)']._values,
        SystemResult['Topics(Str)']._values,
        np.array([])
    ], dtype=object)

    SR_aligned = pd.merge(GoldenStandard[['Sequence']], SystemResult, on='Sequence', how='left')
    SR[0] = SR_aligned['Topics(Id)'].fillna(-1)._values
    SR[1] = SR_aligned['Topics(Str)'].fillna('')._values

    gs_labels_for_entropy = []
    for x in GoldenStandard['Topics(Id)']._values:
        try:
            first_id = str(x).split(',')[0]
            gs_labels_for_entropy.append([int(float(first_id))])
        except (ValueError, AttributeError):
            gs_labels_for_entropy.append([-1])

    GS_Numbers_full = [str(x).split(',') for x in GoldenStandard['Topics(Id)']._values]
    GS_Strings_full = [str(x).split(',') for x in GoldenStandard['Topics(Str)']._values]
    GS[3] = TitleOfEachLabel_new(GS_Numbers_full, GS_Strings_full)

    sr_labels_for_entropy = []
    sr_labels_full = []
    sr_titles_full = []
    for x in SR[0]:
        try:
            first_id = str(x).replace('[','').replace(']','').replace("'",'').split(',')[0]
            sr_labels_for_entropy.append([int(float(first_id))])
            clean_x = str(x).replace('[','').replace(']','').replace("'",'')
            sr_labels_full.append([int(float(i)) for i in clean_x.split(',')])
        except (ValueError, AttributeError):
            sr_labels_for_entropy.append([-1])
            sr_labels_full.append([-1])

    for x in SR[1]:
        sr_titles_full.append(str(x).split(','))

    _, sr_topics_for_matching = MergeStrings(sr_labels_full, sr_titles_full)

    return np.array([GS[0], gs_labels_for_entropy, GS[2], GS[3]], dtype=object), \
           np.array([sr_labels_for_entropy, sr_topics_for_matching], dtype=object)

def run_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_results_path = os.path.join(script_dir, "ProcessedResults")
    golden_standard_path = os.path.join(script_dir, "GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx")
    output_filepath = os.path.join(script_dir, "Final_Evaluation_Report.xlsx")

    golden_standard_df = pd.read_excel(golden_standard_path)

    golden_standard_df['Topics(Id)'] = golden_standard_df['Topics(Id)'].astype(str)
    golden_standard_df = golden_standard_df[golden_standard_df['Topics(Id)'] != '0'].copy()

    results = []

    for filename in os.listdir(processed_results_path):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(processed_results_path, filename)
            system_result_df = pd.read_excel(filepath)

            params = get_params_from_filename(filename)

            GS, SR = PrepareData_new(golden_standard_df, system_result_df)

            ClassEntropy, ClusterEntropy, TotalEntropy = Entropy(GS[1], SR[0])

            TopicPrecision, TopicRecall, TopicF1, KeywordPrecision, KeywordRecall, KeywordF1 = TopicEvaluation(GS[3].copy(),SR[1].copy())

            results.append({
                "step_time_hours": params.get("step_time_hours"),
                "u": params.get("u"),
                "e": params.get("e"),
                "k": params.get("k"),
                "min": params.get("min"),
                "tereshold": params.get("tereshold"),
                "value": params.get("value"),
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
