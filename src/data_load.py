import pandas as pd

customers = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\customers.csv")
transactions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\transactions.csv")
products = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\products.csv")
sessions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\sessions.csv")

df = pd.merge(transactions , customers , on="customer_id")
df = pd.merge(df , products , on="product_id")

df = df.drop(["transaction_id" , "customer_id" , "product_id" , "transaction_date" , "signup_date" , "product_name" , "stock_quantity" , "weight_kg"] , axis=1)
