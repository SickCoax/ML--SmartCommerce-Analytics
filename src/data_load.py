import pandas as pd
from sklearn.preprocessing import OneHotEncoder

customers = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\customers.csv")
transactions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\transactions.csv")
products = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\products.csv")
sessions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\sessions.csv")

df = pd.merge(transactions , customers , on="customer_id")
df = pd.merge(df , products , on="product_id")

df = df.drop(["product_id" , "transaction_date" , "signup_date" , "product_name" , "stock_quantity" , "weight_kg" , "category" , "brand"] , axis=1)

ohe = OneHotEncoder(handle_unknown="ignore" , sparse_output=False)

# OHE FOR paymnet_method
array_encoded_payment_method = ohe.fit_transform(df[["payment_method"]])
encoded_payment_method = pd.DataFrame(array_encoded_payment_method , columns=ohe.get_feature_names_out(["payment_method"]))
df = df.drop(["payment_method"] , axis=1)
df = pd.concat([df , encoded_payment_method] , axis=1)

# OHE FOR status
array_encoded_status = ohe.fit_transform(df[["status"]])
encoded_status = pd.DataFrame(array_encoded_status , columns=ohe.get_feature_names_out(["status"]))
df = df.drop(["status"] , axis=1)
df = pd.concat([df , encoded_status] , axis=1)

df = df.groupby("customer_id").agg({
    "quantity" : "sum" ,
    "unit_price" : "mean" ,
    "total_amount" : "sum" ,
    "discount_applied" : ["sum" , "mean"] ,
    "shipping_cost" : ["sum" , "mean"] ,
    "age" : "first" ,
    "lifetime_value" : "max" ,
    "transaction_id" : "count" ,
    "email_opt_in" : "max",
    "has_app" : "max",
    "price" : "mean" ,
    "avg_rating" : "mean" ,
    "num_ratings" : "mean" , 
    "discount_pct" : "mean" ,
    "gender" : "first" ,
    "country" : "first" ,
    "segment" : "first" ,
    "payment_method_apple_pay" : "sum" ,
    "payment_method_bank_transfer" : "sum" , 
    "payment_method_credit_card" : "sum" , 
    "payment_method_debit_card" : "sum" , 
    "payment_method_google_pay" : "sum" , 
    "payment_method_paypal" : "sum" ,
    "status_cancelled" : "sum" , 
    "status_completed" : "sum" , 
    "status_pending" : "sum" , 
    "status_refunded" : "sum",
    "is_churned" : "first"
})

df = df.reset_index()

df.columns = ['_'.join(col).strip('_') for col in df.columns]
df = df.drop(["customer_id"] , axis=1)