import pandas as pd
from data_preprocessing import processed_dataset

customers = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\customers.csv")
transactions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\transactions.csv")
products = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\products.csv")
# sessions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\sessions.csv")

df = processed_dataset(customers , transactions , products)
cat_cols = df.select_dtypes(include=["object" , "string"]).columns
num_cols = df.select_dtypes(include=["number"]).columns

# while True :
#     choice = int(input("Enter Your Choice : "))

#     match choice :
#         case 1 :
#             pass