from train_regression import train_lifetime_value
from sklearn.metrics import mean_absolute_error

def evaluate_reg(model , X_test , y_test) :

    ypred = model.predict(X_test)

    MAE = mean_absolute_error(y_test , ypred)

    return ypred , MAE