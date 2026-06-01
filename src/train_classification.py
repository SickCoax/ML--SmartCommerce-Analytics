from data_preprocessing import get_feature_table
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


def train_is_churned(df) :

    X = get_feature_table(df)
    y = df["is_churned_first"]

    cat_cols = X.select_dtypes(include=["object" , "string"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

    preprocess = ColumnTransformer([("cat" , OneHotEncoder(handle_unknown="ignore") , cat_cols) ,
                                    ("num" , StandardScaler() , num_cols)])
    
    pipeline = Pipeline([
        ("preprocess" , preprocess),
        ("xgbc" , XGBClassifier(
            n_jobs = -1 ,
            random_state = 42 ,
            subsample = 0.8 ,
            colsample_bytree = 0.8,
            max_depth = 4,
            min_child_weight = 1,
            gamma = 1.2222,           
            learning_rate = 0.0219 ,
            reg_alpha = 0.0106 ,
            reg_lambda = 14 ,
            n_estimators = 355 ,
            scale_pos_weight = 5
        ))
    ])

    # The HyperParameter is done in HyperParameter Tunning notebook and found the best parameters

    model = pipeline.fit(X_train , y_train)

    return model , X_test , y_test