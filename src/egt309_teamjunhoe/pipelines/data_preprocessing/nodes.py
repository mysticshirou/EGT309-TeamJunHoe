import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

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

    if params.get("create_age_unk", None) == True:
        # Create new unknown age column to show if age is unknown or not
        intermediate_data.insert(
            loc=intermediate_data.columns.get_loc("age")+1, 
            column="age_unk", 
            value=intermediate_data.age.map(lambda x: True if x == -1 else False)
        )

    # No label encoding needed
    return intermediate_data

def _clean_occupation_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # No data cleaning needed
    intermediate_data.occupation = intermediate_data.occupation.map(lambda cat: cat.lower().replace(".", ""))
    return intermediate_data

def _clean_marital_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # No data cleaning needed
    return intermediate_data

def _clean_education_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # No data cleaning needed
    intermediate_data.education_level = intermediate_data.education_level.map(lambda cat: cat.lower().replace(".", "_"))
    return intermediate_data

def _clean_credit_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # No data cleaning needed
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
    # Replace null values with -1
    intermediate_data.housing_loan = intermediate_data.housing_loan.fillna("missing")

    # Label encoding (set values)
    category_map = {"missing": -1, "no": 0, "yes": 1, "unknown": 2}
    intermediate_data.housing_loan = intermediate_data.housing_loan.map(category_map)
    return intermediate_data

def _clean_personal_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Not done

    return intermediate_data

def _clean_subscriber_column (intermediate_data: pd.DataFrame, params) -> pd.DataFrame:
    # Convert to boolean
    intermediate_data.subscription_status = intermediate_data.subscription_status.map(lambda x: True if x == "yes" else False)

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
        if col.dtype == "object":
            intermediate_data[col] = le.fit_transform(intermediate_data[col])
    return intermediate_data

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#   Data Cleaning Node
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def clean_dataset (dataset: pd.DataFrame, params) -> pd.DataFrame:
    # Run the data through all the data cleaning functions
    # Feature Columns
    intermediate_data = _clean_initial(dataset, params)
    intermediate_data = _clean_age_column(intermediate_data, params)
    intermediate_data = _clean_occupation_column(intermediate_data, params)
    intermediate_data = _clean_marital_column(intermediate_data, params)
    intermediate_data = _clean_education_column(intermediate_data, params)
    intermediate_data = _clean_credit_column(intermediate_data, params)
    intermediate_data = _clean_contact_column(intermediate_data, params)
    intermediate_data = _clean_campaign_column(intermediate_data, params)
    intermediate_data = _clean_pdays_column(intermediate_data, params)
    intermediate_data = _clean_housing_column(intermediate_data, params)
    intermediate_data = _clean_personal_column(intermediate_data, params)

    # Target Column
    cleaned_data = _clean_subscriber_column(intermediate_data, params)

    return cleaned_data

def encode_dataset (dataset: pd.DataFrame, params):
    if params.get("one_hot_encode", None) == True:
        return _one_hot_encode(dataset)
    elif params.get("label_encode", None) == True:
        return _label_encode(dataset)
    else:
        # Default to one hot encoding
        return _one_hot_encode(dataset)