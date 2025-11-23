import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

def get_params_from_filename(filename):
    """Extracts parameters from the filename."""
    param_pattern = re.compile(
        r"u-(?P<u>\d+)_e-(?P<e>\d+)_step_time_hours-(?P<step_time_hours>\d+)_"
        r"k-(?P<k>\d+)_k_min-(?P<k_min>\d+)_tereshold-(?P<tereshold>[\d\.]+)_"
        r"k_value-(?P<k_value>\d+)"
    )
    match = param_pattern.search(filename)
    if match:
        return match.groupdict()
    return {}

def get_engine(filepath):
    """Returns the appropriate engine for reading the Excel file."""
    if filepath.endswith(".xlsx"):
        return "openpyxl"
    elif filepath.endswith(".xls"):
        return "xlrd"
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def process_files():
    """
    Processes the raw system results to generate consolidated Excel files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    raw_results_path = os.path.join(project_root, "Evaluation1/RAWSystemResults")
    golden_standard_path = os.path.join(project_root, "Evaluation1/GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx")
    all_data_path = os.path.join(project_root, "Language-Model_Scoring/AllData.npy")
    output_path = os.path.join(project_root, "Evaluation1/ProcessedResults")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    golden_standard_df = pd.read_excel(golden_standard_path, engine=get_engine(golden_standard_path))

    all_data = np.load(all_data_path, allow_pickle=True)
    event_to_text = {}
    for i in range(len(all_data[0])):
        for j in range(len(all_data[0][i])):
            event_id = all_data[0][i][j]
            text_tokens = all_data[1][i][j]
            event_to_text[event_id] = " ".join(map(str, text_tokens))


    files_by_params = defaultdict(dict)
    for filename in os.listdir(raw_results_path):
        params = get_params_from_filename(filename)
        if params:
            param_key = tuple(sorted(params.items()))
            filepath = os.path.join(raw_results_path, filename)
            if "ResultsToCompaire" in filename:
                files_by_params[param_key]["compaire"] = filepath
            elif "Topic_Systemresult" in filename:
                files_by_params[param_key]["topic"] = filepath

    for param_key, files in files_by_params.items():
        if "compaire" not in files or "topic" not in files:
            continue

        print(f"Processing files for parameters: {dict(param_key)}")

        compaire_engine = get_engine(files["compaire"])
        topic_engine = get_engine(files["topic"])
        compaire_df_sheets = pd.read_excel(files["compaire"], sheet_name=None, engine=compaire_engine)
        topic_df = pd.read_excel(files["topic"], engine=topic_engine)

        event_map = {}
        for sheet_name, sheet_df in compaire_df_sheets.items():
            match = re.search(r'Window-(\d+)', sheet_name)
            if match:
                window_num = int(match.group(1))
                for _, row in sheet_df.iterrows():
                    event_seq = int(row["Sequence"])
                    event_map[event_seq] = {
                        "EventNumber": row["EventNumber"],
                        "WindowNumber": window_num
                    }

        output_df = golden_standard_df.copy()
        output_df["Topics(Id)"] = ""
        output_df["Topics(Str)"] = ""
        output_df = output_df.astype({"Topics(Id)": object, "Topics(Str)": object})

        for index, row in output_df.iterrows():
            gs_sequence_str = str(row["Sequence"])

            try:
                individual_events = tuple(map(int, re.findall(r'\d+', gs_sequence_str)))
            except (ValueError, TypeError):
                continue

            found_topic_ids = []
            found_topic_strs = set()

            full_text_parts = [event_to_text.get(event, "") for event in individual_events]
            full_text = " ".join(full_text_parts)
            full_text_lower = full_text.lower()
            full_text_word_set = set(full_text_lower.split())


            for event in individual_events:
                if event in event_map:
                    event_info = event_map[event]
                    found_topic_ids.append(str(event_info["EventNumber"]))

                    window_num = event_info["WindowNumber"]

                    matching_rows = topic_df[topic_df["Window Number"] == window_num]
                    if not matching_rows.empty:
                        all_topics = []
                        for _, topic_row in matching_rows.iterrows():
                            topics_str = topic_row["Topic"]
                            if isinstance(topics_str, str):
                                all_topics.extend([topic.strip() for topic in topics_str.split("|")])

                        for topic in all_topics:
                            topic_lower = topic.lower()
                            topic_words = topic_lower.split()

                            # Combined logic: (All words are present) OR (The full phrase is a substring)
                            all_words_present = all(word in full_text_word_set for word in topic_words)
                            phrase_is_substring = topic_lower in full_text_lower

                            if all_words_present or phrase_is_substring:
                                found_topic_strs.add(topic)

            if found_topic_ids:
                output_df.at[index, "Topics(Id)"] = ", ".join(sorted(list(set(found_topic_ids))))
            if found_topic_strs:
                output_df.at[index, "Topics(Str)"] = ", ".join(sorted(list(found_topic_strs)))

        params = dict(param_key)
        output_filename = f"Processed_u-{params['u']}_e-{params['e']}_step-{params['step_time_hours']}_k-{params['k']}_k_min-{params['k_min']}_t-{params['tereshold']}_kv-{params['k_value']}.xlsx"
        output_filepath = os.path.join(output_path, output_filename)
        output_df.to_excel(output_filepath, index=False)
        print(f"Saved processed file: {output_filepath}")


if __name__ == "__main__":
    process_files()