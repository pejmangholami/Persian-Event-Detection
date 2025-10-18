import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

def get_params_from_filename(filename):
    """Extracts parameters from the filename."""
    param_pattern = re.compile(
        r"u-(?P<u>\d+)_e-(?P<e>\d+)_step_time_hours-(?P<step_time_hours>\d+)_"
        r"k-(?P<k>\d+)_k_min-(?P<k_min>\d+)_tereshold-(?P<tereshold>\d+\.\d+)_"
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
    # Define paths
    raw_results_path = "Evaluation1/RAWSystemResults"
    golden_standard_path = "Evaluation1/GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx"
    all_data_path = "Language-Model_Scoring/AllData.npy"
    output_path = "Evaluation1/ProcessedResults"

    # Load golden standard
    golden_standard_df = pd.read_excel(golden_standard_path, engine=get_engine(golden_standard_path))

    # Load AllData.npy
    all_data = np.load(all_data_path, allow_pickle=True)
    sequence_to_text = {
        tuple(seq): " ".join(token for token_list in text for token in token_list)
        for seq, text in zip(all_data[0], all_data[1])
    }

    # Group files by parameters
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

    # Process each parameter set
    for param_key, files in files_by_params.items():
        if "compaire" not in files or "topic" not in files:
            continue

        print(f"Processing files for parameters: {dict(param_key)}")

        # Load results files
        compaire_engine = get_engine(files["compaire"])
        topic_engine = get_engine(files["topic"])
        compaire_df_sheets = pd.read_excel(files["compaire"], sheet_name=None, engine=compaire_engine)
        topic_df = pd.read_excel(files["topic"], engine=topic_engine)

        # Create a mapping from sequence to EventNumber and WindowNumber
        sequence_to_event_window = {}
        for sheet_name, sheet_df in compaire_df_sheets.items():
            match = re.search(r'Window-(\d+)', sheet_name)
            if match:
                window_num = int(match.group(1))
                for _, row in sheet_df.iterrows():
                    # The 'Sequence' column in the Excel file seems to be just an integer, not a list
                    # We will assume it's a single integer and convert it to a tuple to match our keys
                    seq_tuple = (row["Sequence"],)
                    sequence_to_event_window[seq_tuple] = {
                        "EventNumber": row["EventNumber"],
                        "WindowNumber": window_num
                    }

        # Initialize output dataframe
        output_df = golden_standard_df.copy()
        output_df["Topics(Id)"] = ""
        output_df["Topics(Str)"] = ""

        # Populate Topics(Id) and Topics(Str)
        for index, row in output_df.iterrows():
            sequence_str = row["Sequence"]
            try:
                # Convert the string representation of a list to a tuple of integers
                sequence_tuple = tuple(map(int, re.findall(r'\d+', sequence_str)))
            except (ValueError, TypeError):
                continue

            if sequence_tuple in sequence_to_event_window:
                event_info = sequence_to_event_window[sequence_tuple]
                output_df.at[index, "Topics(Id)"] = event_info["EventNumber"]

                window_num = event_info["WindowNumber"]
                text = sequence_to_text.get(sequence_tuple, "")

                if text and window_num in topic_df["WindowNumber"].values:
                    topics_str = topic_df[topic_df["WindowNumber"] == window_num]["Topics"].iloc[0]
                    topics = [topic.strip() for topic in topics_str.split("|")]

                    found_topics = [topic for topic in topics if topic in text]

                    if found_topics:
                        output_df.at[index, "Topics(Str)"] = ", ".join(found_topics)

        # Save the output file
        params = dict(param_key)
        output_filename = f"Processed_u-{params['u']}_e-{params['e']}_step-{params['step_time_hours']}_k-{params['k']}_k_min-{params['k_min']}_t-{params['tereshold']}_kv-{params['k_value']}.xlsx"
        output_filepath = os.path.join(output_path, output_filename)
        output_df.to_excel(output_filepath, index=False)
        print(f"Saved processed file: {output_filepath}")


if __name__ == "__main__":
    process_files()