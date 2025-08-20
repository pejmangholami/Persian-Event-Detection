# Language Model Scoring for Event Detection

This project is a modified version of the original event detection system. The key change is the replacement of the database-dependent scoring mechanism (which used N-gram and Wikipedia databases) with a modern, self-contained scoring system based on pre-trained language models (LMs).

## Key Changes

- **Database-Free:** The system no longer requires any external database connection. All scoring is done locally using language models.
- **Language Model Scoring:** The "stickiness" and "newsworthiness" of text segments are now calculated using scores from a Persian BERT model. This provides a more robust and context-aware measure of how meaningful a phrase is.
- **On-the-Fly Processing:** The script has been configured to process the data from scratch, starting from the `AllData.npy` file, rather than relying on pre-computed intermediate files.
- **Selectable Models:** You can easily switch between different language models to experiment with their performance.

## ⚡️ Performance and Parallelization

This version of the project has been significantly optimized for speed by introducing parallel processing at several key stages of the pipeline. The goal was to reduce the overall execution time without changing the final output.

The following components have been parallelized using Python's `multiprocessing` library:

-   **Post Segmentation:** The process of segmenting individual posts, which is one of the most time-consuming steps, is now performed in parallel. Multiple posts are processed concurrently, taking full advantage of multi-core CPUs.
-   **Language Model Scoring:** Functions that rely heavily on the language model for scoring, such as `EventNewsWorthy` and `DescribeEvents`, have been optimized to run scoring operations in parallel. This significantly speeds up the calculation of "newsworthiness" and the generation of event descriptions.
-   **Similarity Calculation:** The calculation of similarity scores between segments, another computationally intensive task, has been parallelized to accelerate the creation of the event graph.

These changes result in a much faster execution time, especially on machines with multiple CPU cores.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.6+
    *   Access to the internet to download the language models for the first run.

2.  **Install Dependencies:**
    Navigate to this directory in your terminal and install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the script, the `transformers` library will download the pre-trained language model, which may take some time and require a stable internet connection.*

## How to Run

To run the event detection pipeline, simply execute the main script:

```bash
python PTwEvent_NewWindowing.py
```

The script will perform all steps, from segmenting posts to detecting and describing events, and will save the results and intermediate files (like `PostsSegments_Windowing.npy`, `RealisticEvents.npy`, etc.) in the current directory.

**Note on Data Files:** This script is configured to load the main data file (`AllData.npy`) from the parent directory (`../Language-Model_Scoring`). Please ensure that the `Language-Model_Scoring_Parallel_Fast` directory is a sibling of the original `Language-Model_Scoring` directory.

### Selecting a Language Model

You can choose which language model to use for scoring by editing the `LM_MODEL_NAME` variable at the top of the `PTwEvent_NewWindowing.py` script.

The available options are:
- `'HooshvareLab/bert-fa-base-uncased'`: A strong BERT model for Persian.
- `'bert-base-multilingual-cased'`: A multilingual model from Google that also supports Persian.

To change the model, simply modify this line:
```python
#
# ... inside PTwEvent_NewWindowing.py ...
#

# Choose model: 'HooshvareLab/bert-fa-base-uncased' or 'bert-base-multilingual-cased'
LM_MODEL_NAME = 'bert-base-multilingual-cased' # <-- Change this value

# ...
```
