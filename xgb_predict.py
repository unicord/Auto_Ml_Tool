from myTools.pdTools.auto_ml_tool.amt_2_2 import Amtl_predict
import pandas as pd
import numpy as np
model_path = 'model_save.pickle'
test_path = 'test.csv'
train_data = pd.read_csv(test_path)

gender_map_a = {np.NaN: 1}
train_data["Cabin"] = train_data["Cabin"].map(gender_map_a)

gender_map_b = {np.NaN: 0, 1: 1}
train_data["Cabin"] = train_data["Cabin"].map(gender_map_b)

pslp =Amtl_predict(train_data, model_path, Show_title=True, math_format=0)
pslp.print_csv()