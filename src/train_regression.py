from data_preprocessing import get_feature_table
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline


def train_lifetime_value(df) :

    X = get_feature_table(df)
    y = df["lifetime_value_max"]

    cat_cols = X.select_dtypes(include=["object" , "string"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

    preprocess = ColumnTransformer([("cat" , OneHotEncoder(handle_unknown="ignore") , cat_cols) ,
                                    ("num" , StandardScaler() , num_cols)])
    
    pipeline = Pipeline([
        ("preprocessing" , preprocess) ,
        ("xgbr" , XGBRegressor(
            n_jobs = -1 , 
            random_state = 42 , 
            subsample = 0.8 ,
            max_depth = 5 ,
            min_child_weight = 2 ,
            gamma = 0.5375 ,
            learning_rate = 0.0133 ,
            reg_alpha = 0.0136 ,
            reg_lambda = 2.229 ,
            n_estimators = 373 ,
        ))
    ])

    # The HyperParameter is done in HyperParameter Tunning notebook and found best values

    model = pipeline.fit(X_train , y_train)

    return model , X_test , y_test