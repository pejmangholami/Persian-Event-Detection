import pandas as pd
import numpy as np
import os
import re
from math import log
import copy

def get_params_from_filename(filename):
    """Extracts parameters from the filename using a regular expression."""
    params = {}
    # This regex will robustly capture all key-value pairs
    matches = re.findall(r'(\w+)-([\d\.]+)', filename)
    for key, value in matches:
        params[key] = value

    # a special case to handle k-min because of the underscore
    k_min_match = re.search(r'k_min-([\d\.]+)', filename)
    if k_min_match:
        params['k_min'] = k_min_match.group(1)

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

def Entropy(Samples,Evals, Desires):
    """ Calculate the Entropy """
    HO = 0 #Total Entropy
    Classes = dict()
    Clusters = dict()

    all_desires = {item for sublist in Desires for item in sublist}
    num_classes = len(all_desires)

    Scores = list()
    for i in range(len(Samples)):
        if len(Evals[i]) > 0:
            Scores.append(1 / len(Evals[i]))
        else:
            Scores.append(0)

        for D in Desires[i]:
            if D in Classes:
                 Classes[D].append(i)
            else:
                Classes.update({D : [i]})

        for E in Evals[i]:
            if E in Clusters:
                Clusters[E].append(i)
            else:
                Clusters.update({E : [i]})

    for Cluster in Clusters:
        Sum = 0
        H = 0
        S = dict()
        for Class in Classes:
            S.update({Class:0})

        SC = dict()
        FC = dict()

        for Idx in Clusters[Cluster]:
            for c in Desires[Idx]:
                if c in FC:
                    FC[c] += 1
                else:
                    FC.update({c : 1})
            SC.update({Idx : copy.deepcopy(Desires[Idx])})

        while len(FC) != 0:
            MaxD = max(list(FC.items()), key = lambda x:x[1])
            if MaxD[1] == 0:
                break
            m = MaxD[0]
            for s in SC:
                if m in SC[s]:
                    for i in SC[s]:
                        FC[i] -= 1
                    SC[s] = [m]
            FC.pop(m)

        for Idx in Clusters[Cluster]:
            Sum += Scores[Idx]
            if not SC[Idx]:
                continue
            C = SC[Idx][0]
            S[C] += Scores[Idx]

        for s in S:
            if S[s] != 0 and Sum > 0:
                H += S[s] * log((S[s]/Sum),2)
        HO += -1 * H

    N = len(Samples)
    HO = HO / N

    return HO * 0.4

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
