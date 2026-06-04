import pandas as pd
from data_preprocessing import processed_dataset
from train_regression import train_lifetime_value
from train_classification import train_is_churned
from evaluation import evaluate_reg , evaluate_cls

customers = pd.read_csv(r"dataset\customers.csv")
transactions = pd.read_csv(r"dataset\transactions.csv")
products = pd.read_csv(r"dataset\products.csv")

df = processed_dataset(customers , transactions , products)

while True :

    print("Choice :     Task")
    print(" {1}   : Lifetime Value")
    print(" {2}   :   Is Churned")
    print(" {3}   :     EXIT")
    print()

    try :

        choice = int(input("Enter Your Choice : "))
        print()    

        match choice :

            case 1 :
                model , X_test , y_test = train_lifetime_value(df)
                y_pred , MAE = evaluate_reg(model , X_test , y_test)

                print("----------------------------------------------")
                print("Model Predictions : ")
                print(y_pred)
                print()
                print(f"Mean Absolute Error : {MAE}")
                print()
                print("SUCCESFULLY DONE")
                print("----------------------------------------------")
                print()
            
            case 2 :
                model , X_test , y_test = train_is_churned(df)
                ypred , f1 = evaluate_cls(model , X_test , y_test)

                print("----------------------------------------------")
                print("Model Predictions : ")
                print(ypred)
                print()
                print(f"F1 Score : {f1}")
                print()
                print("SUCCESFULLY DONE")
                print("----------------------------------------------")
                print()
            
            case 3 :
                print("----------------------------------------------")
                print("EXITED")
                print("----------------------------------------------")
                break

            case _ :
                print("----------------------------------------------")
                print("INVALID OPTION")
                print("----------------------------------------------")
                print()

    except ValueError :
        print("----------------------------------------------")
        print("INVALID OPTION")
        print("----------------------------------------------")
        print()
