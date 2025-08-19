# Language Model Scoring for Event Detection

This project is a modified version of the original event detection system. The key change is the replacement of the database-dependent scoring mechanism (which used N-gram and Wikipedia databases) with a modern, self-contained scoring system based on pre-trained language models (LMs).

## Key Changes

- **Database-Free:** The system no longer requires any external database connection. All scoring is done locally using language models.
- **Language Model Scoring:** The "stickiness" and "newsworthiness" of text segments are now calculated using scores from a Persian BERT model. This provides a more robust and context-aware measure of how meaningful a phrase is.
- **On-the-Fly Processing:** The script has been configured to process the data from scratch, starting from the `AllData.npy` file, rather than relying on pre-computed intermediate files.
- **Selectable Models:** You can easily switch between different language models to experiment with their performance.

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
