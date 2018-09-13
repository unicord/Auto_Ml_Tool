from myTools.pdTools.auto_ml_tool.amt_2_3 import Amtl_predict
import pandas as pd
import numpy as np
model_path = 'model_save.pickle'
test_path = 'test.csv'
train_data = pd.read_csv(test_path, encoding='ISO-8859-1', low_memory=False)



pslp =Amtl_predict(train_data, model_path, Show_title=True, math_format=0)
pslp.print_csv()