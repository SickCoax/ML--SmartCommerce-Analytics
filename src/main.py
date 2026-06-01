import pandas as pd
from data_preprocessing import processed_dataset
from train_regression import train_lifetime_value
from train_classification import train_is_churned
from evaluation import evaluate_reg , evaluate_cls

customers = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\customers.csv")
transactions = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\transactions.csv")
products = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\SmartCommerce Analytics\dataset\products.csv")

df = processed_dataset(customers , transactions , products)

# model , X_test , y_test = train_lifetime_value(df)

# y_pred , MAE = evaluate_reg(model , X_test , y_test)

# print("Model Predictions : ")
# print(y_pred)
# print()
# print(f"Mean Absolute Error : {MAE}")

# while True :
#     choice = int(input("Enter Your Choice : "))


model , X_test , y_test = train_is_churned(df)

ypred , f1 = evaluate_cls(model , X_test , y_test)

print(ypred)
print()
print(f1)

