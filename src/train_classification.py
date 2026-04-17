from data_preprocessing import get_feature_table
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler


def train_is_churned(df) :

    X = get_feature_table(df)
    y = df["is_churned_first"]

    cat_cols = X.select_dtypes(include=["object" , "string"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

    preprocess = ColumnTransformer([("cat" , OneHotEncoder(handle_unknown="ignore") , cat_cols) ,
                                    ("num" , StandardScaler() , num_cols)])    