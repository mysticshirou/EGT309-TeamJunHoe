# EGT309-TeamJunHoe

## Section A - Contributors

### Foo Tun Wei Darren - 231725Z@mymail.nyp.edu.sg

### Harish Kanna - 230268R@mymail.nyp.edu.sg

## Section B - Folder Structure

```
# Comments highlight directories not in base kedro structure

EGT309-TEAMJUNHOE
├── conf
│   ├── base
│   └── local
├── data
│   ├── 01_raw
│   ├── 02_cleaned
│   ├── 06_models
│   │   └── trained_model
│   └── 08_reporting
├── saved_models                # Finalized models saved here
└── src
    └── egt309_teamjunhoe
        └── pipelines
            ├── data_preprocessing
            └── data_science
                └── models      # Contains model classes

17 directories
```
## Section C - Instructions

Run `run.sh` to run the finalized pipeline

Modify parameters in `conf/base/parameters.yml`

There are three categories of parameters:
1. `data_cleaning_parameters` - Parameters for data cleaning
2. `encoding_params` - Parameters for data encoding (e.g. OHE / LabelEncoding)
3. `model_params` - Parameters for the ML model pipeline

All editable parameters are included in the `parameters.yml` file. Possible parameter inputs are also included in the file.

## Section D - Pipeline

### Flowchart

```mermaid
graph TD
    start(START) --> a
    subgraph "kedro pipeline"
        subgraph data_preprocessing
            a[/DATASET: bmarket_data/] --> A[NODE: clean_dataset]
            A --> b[/DATASET: cleaned_data/]
            A --> B[NODE: feature_selection_dataset]
            B --> C[NODE: encode_dataset]
            C -->|"Selected encoding option (e.g. ''ohe'')"| c[/DATASET: encoded_cleaned_data/]
            C --> D[NODE: split_dataset]
            D --> X_train; D --> y_train; D --> X_test; D --> y_test
            b --> B
            subgraph datasets
                a; b; c
            end
        end
        subgraph data_science
            D --> E[NODE: model_choice]
            E -->|"Selected model option (e.g. ''cat_boost'')"| F[NODE: model_train]
            X_train --> F; y_train --> F; X_test --> F; y_test --> F
            F --> d[/PICKLE: trained_model/]
            F --> e[/PLOT: training_graphs/]
            d --> G[NODE: model_eval]
            G --> H[NODE: model_save]
            d --> H
            G --> g[/PLOT: evaluation_graphs/]
            G --> h[/JSON: evaluation_metrics/]
            subgraph reporting
                e; g; h
            end
        end
        H --> i[/PICKLE: final_model/]
    end
    i --> en(END)
```

### Explanation for each node in the kedro pipeline (in sequence)

| Node | Function | Input(s) | Output(s) |
| :---: | :--- | :---: | :---: |
| clean_dataset |  Cleans the dataset according to the methods and algorithms created in eda.ipynb. Some parameters can be configured from the parameters.yml | bmarket_data (SQLTableDataset) | cleaned_data (ParquetDataset) |
| feature_selection_dataset | Remove columns starting from least important based on the ranking defined inside parameters.yml. Number of columns to drop can be configured inside parameters.yml | cleaned_data (ParquetDataset) | feature_selected_data (InMemory) |
| encode_dataset | Encode the dataset based on the specified encoding strategy inside parameters.yml ("ohe", "label", "none") | feature_selected_data (InMemory) | encoded_cleaned_data (ParquetDataset) |
| split_dataset | Stratified split the dataset into training and testing splits. Has functionality for imbalance handling and can be configured from the parameters.yml | encoded_cleaned_data (ParquetDataset) | X_train, y_train, X_test, y_test (InMemory) |
| model_choice | Initialises the chosen model as specified within the parameters.yml. ("decision_tree", "cat_boost", "ada_boost", etc.) | No dataset input | model_choice (InMemory) |
| model_train | Trains the model according to the training strategy specified in parameters.yml. Training methods differ depending on the model choice. | X_train, y_train, model_choice (InMemory) | trained_model (PickleDataset), training_graphs (MatplotlibDataset) |
| model_eval | Evaluates the model and generates a classification / metric report and plots an evaluation graph | trained_model (PickleDataset), X_test, y_test (InMemory) | evaluation_metrics (JSONDataset), evaluation_graphs (MatplotlibDataset) |
| model_save | Saves the model to the saved_models directory | trained_model (PickleDataset) | None (Saves model to directory) |

## Section E - EDA Overview

## Section F - Data Processing Overview

This section primarily explains the (default) main data processing steps for each column of the dataset.

| Column Name | Processing Steps |
|:---:|---|
| AGE | Data processing done by changing outlier ages (150 yrs old) via imputation (configurable) |
| MARITAL STATUS | No data processing |
| OCCUPATION | No data processing, some categories changed to easier to read format |
| EDUCATION | No data processing, some categories changed to easier to read format |
| CONTACT METHOD | Combined overlapping values |
| CAMPAIGN CALLS | Changed all negative values to positive |
| PREVIOUS CONTACT DAYS | Created new boolean column to denote 999 as "not previously contact" |
| CREDIT DEFAULT | No data processing |
| HOUSING LOAN | Changed null values to "missing" |
| PERSONAL LOAN | Imputed null values via mode imputation (configurable) |

## Section G - Model Choice

## Section H - Model Evaluation

### Main metrics for model evaluation:
| Ranking | Metric | Definition | Reasoning |
| :---: | :---: | :--- | :--- |
| 1 | F4 Score | Weighted harmonic mean of precision and recall that favours recall | Maximize recall while ensuring precision is not fully abandoned |
| 2 | Recall | How many of total positives were correctly identified as positives | The cost of not selecting (false negative) a customer is greater than the cost of getting a false positive |
| 3 | Precision | How many of predicted positives were true positives | Precision needs to be reasonable such that the bank can filter more true negatives and thus improve resource efficiency |

## Section I - Other Considerations