# EGT309-TeamJunHoe

## Section A - Contributors

### Foo Tun Wei Darren - 231725Z@mymail.nyp.edu.sg
- Data cleaning & preprocessing: Age, Campaign Calls, Education Level, Contact Method, Occupation
- Feature analysis: Univariate analysis for above columns
- Kedro pipeline development: Overall data Preprocessing pipeline, Overall data Science pipeline, Configuration approach
- Model tuning methods: Grid search, Bayesian optimisation
- Model evaluation methods: Confusion matrix visualisation

### Harish Kanna - 230268R@mymail.nyp.edu.sg
- Data cleaning & preprocessing: Marital Status, Previous Contact Days, Credit Default, Personal Loan, Housing Loan
- Feature analysis: Bivariate analysis, Feature selection, Univariate analysis for above columns
- Kedro pipeline development: Model registry, Model interface (Structural Subtyping)
- Model tuning methods: Threshold tuning
- Model evaluation methods: Evaluation metrics decision-making, Weight geometric mean scoring (metric)

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

Make sure your working directory is always the base working directory of the en

### EDA
Run `docker build --no-cache -t eda -f Dockerfile-eda .` to build the jupyter server image
Run `docker run -d -p 8888:8888 eda` to start the jupyter server and use it as the jupyter kernel in your preferred IDE

### Kedro Pipeline
Run `run.sh` to run the finalized pipeline

Modify parameters used in the data_preprocessing pipeline in `conf/base/parameters_datapreprocessing.yml`

There are four categories of data preprocessing parameters:
1. `cleaning_params` - Parameters for data cleaning
2. `feature_selection_params` - Parameters for selecting top N features and ranking of features
2. `encoding_params` - Parameters for data encoding (e.g. OHE / LabelEncoding)
3. `splitting_params` - Parameters data splitting

Modify parameters used in the data_science pipeline in `conf/base/parameters_datascience.yml`

Under the top level model_params key, every type of model has their own section (e.g. cat_boost_settings)

The top of the model_params section has other global values to be selected such as model choice.

Possible parameters are shown as comments above parameters when helpful.

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
The following are some of the main insights and problems we found through Exploratory Data Analysis, and a brief overview of how we dealt with them.

### Many High-Cardinality Categorical Features
Over half of all columns are of a categorical type, and the number of unique classes in a column goes up to 12 at max. Multiple options were created for encoding these columns, which is not only for experimentation but also because different encoding types work better for different models. One-Hot encoding works best with XGBoost, Label Encoding for LightGBM, and no encoding for CatBoost (it uses it's own encoding internally). As such, the pipeline can dynamically change the type of encoding used with a configurable option in the parameters_datapreprocessing.yml file to cater to different models' needs.

### Very imbalanced dataset
The majority class comprises 88% of the entire dataset, which is a harsh imbalance that must be addressed in some way. While both undersampling and oversampling were implemented as choices in the pipeline, the final choice was just to stratify the classes while splitting the data. Oversampling methods like SMOTE would not work well on the complex data, while undersampling would get rid of important data that the model would benefit from having access to for learning.

### Outlier handling
One column featured many outlier values, the Age column (150 years), initially we decided to change all the outliers to -1, to signify that the age is unknown. However, the final choice was to impute the outlier values based on data from other columns (occupation, education level, marital status, credit default, and contact method). Which was successful as the distribution after imputation was able to precisely replicate the original distribution.

### Null value handling
Two of the columns, Housing Loan and Personal Loan, contained null values. Neither of these columns were correlated with any other column, making accurate imputation possible. Housing Loan especially had over 60% of missing data which will make any form of imputation very likely to be wrong, hence all missing values were filled with the value 'missing'. Personal Loan only had around 10% of missing data, and the mode of the column 'no' had a very high percentage, hence mode imputation was used for it.

### Inconsistent value handling
The Contact Method column had more unique values due to inconsistent entry of data than classes that actually existed. Both 'cellular' and 'telephone' were entered in two different ways, which was collapsed to one unique class each.

### Lack of values in certain classes for some columns
The Occupation column had a large number of unique classes, which resulted in some classes having a low number of data points. The Credit Default column only had 3 unique classes, but the 'yes' class only had 3 data points in the entire dataset. Models are prone to overfitting on these classes with few samples, hence feature engineering was used to combine some classes and increase the number of samples per class in each column. Occupation was generalized into 'employed' and 'not employed', while Credit Default was generalized into 'known' and 'unknown'.

### Data entry error handling
The Campaign Calls column featured many data entry errors, with many of its values being negative numbers. This was handled through statistical analysis with the Kolmogorov-Smirnov test (2 sample K-S test), which helped to prove that the negative and positive sample spaces were likely from the same distribution. Hence, the final decision of moving the negative numbers into the positive space throught the absolute function.

## Section F - Data Processing Overview

This section primarily explains the (default) main data processing steps for each column of the dataset.

| Column Name | Processing Steps |
|:---:|---|
| AGE | Data processing done by changing outlier ages (150 yrs old) via KNN imputation based on specific other columns |
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
### CatBoost
CatBoost was the first choice as it is specifically built to handle datasets with many categorical columns. It uses a special encoding type called 'Ordered Target Encoding'. It does not create many new columns like One-Hot Encoding, and assigns numeric values instead to each class like Label Encoding. However, Label Encoding only does a simple naive class mapping that has no real semantic meaning, while Target Encoding uses the relationship between each class of a column and the target column to assign numeric values that have real meaning the model can capitalize on using comparison operators like a normal numeric column.

### LightGBM
LightGBM was the second choice due to its efficiency and speed in training that enables much faster hyperparameter tuning, especially as we continuously attempt to improve the model performance by widening the search spaces. LightGBM achieves this by bundling features together into Exclusive Feature Bundles (EFBs) to reduce computation required, but uses complex algorithms to ensure that this compression of features is lossless and does not affect accuracy.

### XGBoost
Our final choice was XGBoost due to its baseline powerful performance and robustness across tabular dataset problems. XGBoost stands its ground by pushing more standard features to their max performance, such as having highly advanced regularization to prevent overfitting. XGBoost is commonly included as a comparison model for these purposes, and we wish to also use XGBoost to check if our strategies with CatBoost and LightGBM actually lead to better performance.

## Section H - Model Evaluation

### Main metrics for model evaluation:
| Ranking | Metric | Definition | Formula | Reasoning |
| :---: | :---: | :--- | :--- | :--- |
| 1 | Weighted Geometric Mean (GMS) | Weighted geometric mean of recall and specificity that favors recall | `GMS = Recall^0.6 * Specificity^0.4` | Maximize recall while ensuring specificity is not fully abandoned |
| 2 | Recall | How many of total positives were correctly identified as positives | `Recall = TP / (TP + FN)` | The cost of not selecting (false negative) a customer is greater than the cost of getting a false positive |
| 3 | Specificity | How many of total negatives were correctly identified as negatives | `Specificity = TN / (TN + FP)` | Specificity needs to be reasonable such that the bank can confidently filter out customers identified as unlikely to subscribe |



## Section I - Other Considerations

### Model Drift
Over time, predictions are likely to get less and less accurate on newer customers. It is important to monitor metrics to ensure that model retraining is performed when necessary.

### Changing features
While certain features like Age are independent to the banks' events, other features like Campaign Calls that were fully reliant on the one campaign that the bank used to collect data from may change drastically or be removed as a feature. Similarily, retraining will be necessary in this case.

### Interpreting results
While easy to set a clear threshold to separate customers between two distinct classes as 'subscribed' and 'not subscribed', it might be more useful to create multiple bins corresponding to their probability of subscribing, to add the possibility of contacting customers proportionally to their chance of subscribing and enhancing resource allocation while not automatically discarding a large amount of customers.

