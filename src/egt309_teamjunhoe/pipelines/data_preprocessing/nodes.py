import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler  

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Data Cleaning Helper Functions
#   All data cleaning functions will automatically encode categorical data into numerical data
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def _clean_initial (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Basic initial cleaning (drop ids & rename columns)
    column_renames = {name : name.lower().replace(" ", "_") for name in intermediate_data.columns}
    intermediate_data.rename(columns=column_renames, inplace=True)
    intermediate_data.drop("client_id", axis=1, inplace=True)

    return intermediate_data

def _clean_age_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Converting age from str to int
    intermediate_data.age = intermediate_data.age.map(lambda x: int(x.split(" ")[0]))
    # Set >90 years old to -1
    intermediate_data.age = intermediate_data.age.map(lambda x: x if x != 150 else -1)

    if params.get("age_knn_impute", None) == True:
        # Impute missing age data via KNN as per EDA
        # Create OHE data for KNN
        knn_ohe_data = intermediate_data.iloc[:, [0, 2, 3, 4]]
        knn_ohe_data = pd.get_dummies(knn_ohe_data, ["occupation", "marital_status", "education_level"])
        knn_ohe_data.age = knn_ohe_data.age.map(lambda x: x if x != -1 else np.nan)

        # Impute missing values in age column
        imputed_data = KNNImputer(**params.get("age_knn_impute_settings", {})).fit_transform(knn_ohe_data)
        intermediate_data.age = pd.Series([x[0] for x in imputed_data])

    elif params.get("create_age_unk", None) == True:
        # Create new unknown age column to show if age is unknown or not
        intermediate_data.insert(
            loc=intermediate_data.columns.get_loc("age") + 1, 
            column="age_unk",
            value=intermediate_data.age.map(lambda x: True if x == -1 else False)
        )

    # No label encoding needed
    return intermediate_data

def _clean_occupation_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    if params.get("generalize", None) == True:
        intermediate_data.occupation = intermediate_data.occupation.apply(lambda x: "no" if (x == "student") \
                                                                                        or (x == "retired") \
                                                                                            else "yes")
    return intermediate_data

def _clean_marital_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # No data cleaning needed
    return intermediate_data

def _clean_education_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # No data cleaning needed
    intermediate_data.education_level = intermediate_data.education_level.map(lambda cat: cat.lower().replace(".", "_"))
    return intermediate_data

def _clean_credit_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Combine yes and no into known
    intermediate_data.credit_default = intermediate_data.credit_default.map(lambda x: x if x == "unknown" else "known")
    return intermediate_data

def _clean_contact_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Combining Cell & cellular + Telephone & telephone
    intermediate_data.contact_method = intermediate_data.contact_method.map(
        lambda x: "cellular" if x[0].lower() == "c" else "telephone"
    )
    return intermediate_data

def _clean_campaign_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Converting all negative numbers to positive numbers
    intermediate_data.campaign_calls = intermediate_data.campaign_calls.abs()

    return intermediate_data

def _clean_pdays_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    if params.get("create_previously_contacted", None) == True:
        # Create previously contacted column (denotes if person has been contacted recently (boolean))
        intermediate_data.insert(
            loc=intermediate_data.columns.get_loc("previous_contact_days")+1, 
            column="previously_contacted", 
            value=intermediate_data.previous_contact_days.map(lambda x: False if x == 999 else True)
        )

    # No label encoding needed
    return intermediate_data

def _clean_housing_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Replace null values with missing
    intermediate_data.housing_loan = intermediate_data.housing_loan.fillna("missing")
    return intermediate_data

def _clean_personal_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Different methods based on what is chosen in parameters.yml
    method = params.get("personal_loan_cleaning")
    match method:
        case "fill": 
            intermediate_data.personal_loan = intermediate_data.personal_loan.fillna("missing")
        case "drop":
            intermediate_data.drop(labels=['personal_loan'], axis=1, inplace=True)
        case "impute":
            intermediate_data.personal_loan = intermediate_data.personal_loan.fillna(intermediate_data.personal_loan.mode()[0])
    return intermediate_data

def _clean_subscriber_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Convert to boolean
    intermediate_data.subscription_status = intermediate_data.subscription_status.map(lambda x: True if x == "yes" else False)

    return intermediate_data

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Feature selection helper function
# -----------------------------------------------------------------------------------------------------

def _feature_selection (intermediate_data: pd.DataFrame, params):
    ranking: list = params.get("ranking")
    top_n_features: int = params.get("top_n_features")
    selected_features = ranking[:top_n_features]
    cols_to_drop = [col for col in intermediate_data.columns if \
                    col not in selected_features and col != 'subscription_status']
    intermediate_data = intermediate_data.drop(columns=cols_to_drop)
    return intermediate_data

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Encoding Helper Function
# -----------------------------------------------------------------------------------------------------

def _one_hot_encode (intermediate_data: pd.DataFrame):
    return pd.get_dummies(
        intermediate_data, 
        columns=[col for col in intermediate_data.columns if intermediate_data[col].dtype == "object"]
    )

def _label_encode (intermediate_data: pd.DataFrame):
    le = LabelEncoder()
    for col in intermediate_data.columns:
        if intermediate_data[col].dtype == "object":
            intermediate_data[col] = le.fit_transform(intermediate_data[col])
    return intermediate_data

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Splitting Helper Function
# --------------------------------------------------------------------------------------------------------------------------------------------------------


def _undersampling_split(intermediate_data: pd.DataFrame, test_size):
    X = intermediate_data.drop(columns=["subscription_status"])
    y = intermediate_data.subscription_status

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y
    )

    # apply undersampling to training set only
    rus = RandomUnderSampler()
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test

def _stratified_split(intermediate_data: pd.DataFrame, test_size):
    X = intermediate_data.drop(columns=["subscription_status"])
    y = intermediate_data.subscription_status

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def _class_weighted_split(intermediate_data: pd.DataFrame): ...

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Data Cleaning Node
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def clean_dataset (dataset: pd.DataFrame, params) -> pd.DataFrame:
    # Run the data through all the data cleaning functions
    # Feature Columns
    intermediate_data = _clean_initial(dataset, params)
    intermediate_data = _clean_occupation_column(intermediate_data, params)
    intermediate_data = _clean_marital_column(intermediate_data, params)
    intermediate_data = _clean_education_column(intermediate_data, params)
    intermediate_data = _clean_age_column(intermediate_data, params)
    intermediate_data = _clean_credit_column(intermediate_data, params)
    intermediate_data = _clean_contact_column(intermediate_data, params)
    intermediate_data = _clean_campaign_column(intermediate_data, params)
    intermediate_data = _clean_pdays_column(intermediate_data, params)
    intermediate_data = _clean_housing_column(intermediate_data, params)
    intermediate_data = _clean_personal_column(intermediate_data, params)

    # Target Column
    cleaned_data = _clean_subscriber_column(intermediate_data, params)

    return cleaned_data

def feature_selection_dataset (dataset: pd.DataFrame, params):
    return _feature_selection(dataset, params)

def encode_dataset (dataset: pd.DataFrame, params):
    encoding_type = params.get("encode")
    match encoding_type:
        case "ohe":
            return _one_hot_encode(dataset)
        case "label":
            return _label_encode(dataset)
        case "none":
            return dataset
        case _:
            return _one_hot_encode(dataset)
    
def split_dataset (dataset: pd.DataFrame, params):
    test_size = params.get("test_size")
    method = params.get("imbalance_handling", None)
    match method:
        case "stratified":
            return _stratified_split(dataset, test_size)
        case "undersampling":
            return _undersampling_split(dataset, test_size)
        case _:
            raise ValueError(f"\"{params.get('imbalance_handling'), None}\" is not a valid model choice")
