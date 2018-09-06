from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from myTools.pdTools.auto_ml_tool.amt_2_2 import Amtl



data_path = 'train.csv'
train_data = pd.read_csv(data_path)
list_org = ['PassengerId', 'Survived', 'Pclass', "Sex", "Age", "SibSp", "Fare", "Cabin", "Embarked"]

gender_map_a = {np.NaN: 1}
train_data["Cabin"] = train_data["Cabin"].map(gender_map_a)

gender_map_b = {np.NaN: 0, 1: 1}
train_data["Cabin"] = train_data["Cabin"].map(gender_map_b)


#list_one_hot = ['Embarked']
#n_estimators默认100，max_depth默认3，min_child_weight默认1
cv_param = {
            'n_estimators': np.arange(50, 100, 5),
            'max_depth': np.arange(1, 4, 1),
            'min_child_weight': np.arange(1, 2, 1),
            'gamma': np.arange(0, 0.3, 0.1),
            'learning_rate': np.arange(0.1, 0.3, 0.1)
            }

psl = Amtl(train_data, 'PassengerId', 'Survived', model='R', fearture_list=list_org,
            cv_dic=cv_param, pca_param_s=0.95, pca_param_d=0.95)
psl.fit()
print(psl.rf.score(psl.X_train, psl.Y_train))


