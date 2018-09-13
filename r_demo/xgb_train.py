

import pandas as pd
import numpy as np
from myTools.pdTools.auto_ml_tool.amt_2_3 import Amtl

data_path = 'train.csv'
train_data = pd.read_csv('train.csv', encoding='ISO-8859-1', low_memory=False)

list_org = ['Id', 'MSSubClass', 'MSZoning', 'SalePrice']
#list_org = ['PassengerId', 'Survived', 'Pclass',  "Age", "SibSp", "Fare", "Cabin"]
one_hot_list =["City"]


#n_estimators默认100，max_depth默认3，min_child_weight默认1,gamma默认0，learning_rate默认0.1
cv_param = {
            'n_estimators': np.arange(20, 80, 5),
            'max_depth': np.arange(2, 5, 1),
            'min_child_weight': np.arange(1, 2, 1),
            'gamma': np.arange(0.1, 0.2, 0.05),
            'learning_rate': np.arange(0.1, 0.2, 0.05)
            }

psl = Amtl(train_data, 'Id', 'SalePrice', model='R', fearture_list=None, one_hot_list='off',
             pca_param_s=0.9, pca_param_d=0.9)
psl.fit()