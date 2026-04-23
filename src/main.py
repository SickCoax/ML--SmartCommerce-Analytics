import pandas as pd
from data_preprocessing import processed_dataset
from train_regression import train_lifetime_value

customers = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\customers.csv")
transactions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\transactions.csv")
products = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\products.csv")
# sessions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\sessions.csv")

df = processed_dataset(customers , transactions , products)

model = train_lifetime_value(df)
print(model)


# while True :
#     choice = int(input("Enter Your Choice : "))

#     match choice :
#         case 1 :
#             pass