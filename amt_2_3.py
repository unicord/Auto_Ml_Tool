import csv
import datetime
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.decomposition import PCA
import logging
import pickle

from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler, Normalizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

class Amtl(object):
    '''
    版本2.3
    更新内容修复没有数值型特征时不填空值的bug，并将自动搜索文本和数值型变为随机1000个值判断
    '''

    def __init__(self, train_pandas, Id, target, model='R', pca_param_s=None, pca_param_d=None,
                 fearture_list=None, one_hot_list=None, cv_dic=None):
        '''
        预创建

        :param train_pandas: pandas格式的训练集
        :param Id: 唯一id号，排序定位用,不参与计算
        :param target: 目标特征标题
        :param Model:   'R' = XGBRegressor  'C'= XGBClassifier 默认R
        :param pca_param_d: >0 and <1 PCA降维数据的保留比例，数值型，默认关闭
        :param pca_param_s: >0 and <1 PCA降维数据的保留比例，文本型，默认关闭
        :param fearture_list:  = [] list中是特征名称. 注意：输入顺序要按pandas载入的顺序输入
        :param cv_dic: ={} 是否使用cv函数寻找最佳参数，None,关闭将使用默认参数或自建模型
                    暂时只提供n_estimators ，min_child_weight和 max_depth 自动调参，更多内容可关闭手动加入模型
        :param one_hot_list: = [] or ‘off’需要进行one_hot编码的文本型特征名列表，需要注意按顺序输入,
        当为‘off’时，关闭所有one_hot编码,将直接使用转换后数值进行计算
        注释：
        '''

        logging.getLogger().setLevel(logging.INFO)  #日志打印级别
        self.Id = Id
        self.target = target
        self.train_dataset = shuffle(train_pandas)  #原始数据表载入并混淆数据
        self.model = model
        self.pca_param_s = pca_param_s
        self.pca_param_d = pca_param_d
        self.fearture_param = fearture_list
        self.cv_param = cv_dic
        self.one_hot_param = one_hot_list
        self.pca_D = None                           #数值型pca降维器
        self.pca_S = None                           #文本型pca降维器
        #开始流程处理
        self.dataset_load()                         #提取Y_train
        self.laber_deal()                           #建立特征列表
        self.CRclass_deal()                         #分离数值型和文本型
        self.one_hot_prepare()                      #文本转数值
        self.laber_str()                            #特征筛选条件生成
        self.imputer()                              #缺失值处理
        self.one_hot_deal()                         #对文本数据进行哑编码
        self.standardScaler()                       #标准化
        #self.normalizer()                          #归一化
        self.PCA()                                  #PCA降维
        self.reshape()                              #连接字符型和数值型特征
        self.creat_model()                          #建立模型
        if self.cv_param is not None:
            self.skl_cv()                           #建立cv函数调参

    def now_time(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def dataset_load(self):
        if self.model == 'R':
            logging.info("{0}:模型选择为xgb回归".format(self.now_time()))
        elif self.model == 'C':
            logging.info("{0}:模型选择为xgb分类".format(self.now_time()))

        if self.fearture_param is not None:
            self.train_dataset = self.train_dataset[self.fearture_param]
        self.Y_train = self.train_dataset.filter(regex=self.target).values
        self.Y_train = np.array(self.Y_train).T[0]

    def laber_deal(self):
        # 特征列表
        self.laber_list = []
        for i in self.train_dataset:
            self.laber_list.append(i)

        self.laber_list_D = []                  #数值型特征
        self.laber_list_S = []                  #文本型特征总表
        self.laber_list_SA = None               #文本型特征在one_hot编码中
        self.laber_list_SB = None               #文本型特征不在one_hot编码中
        self.train_dataset_loc_S = None         #文本型处理后数据
        self.train_dataset_loc_D = None         #数值型处理后数据
        self.laber_dir_S = {}                   #文本型词标签字典
        self.train_dataset_loc = self.train_dataset.loc[:, ]    #中间表

    def CRclass_deal(self):
        #将数值型和文本型分开
        logging.info("{0}:正在检查分离数值型和文本型特征".format(self.now_time()))
        for i in self.laber_list:
            try:
                if self.train_dataset[:1000].mean()[i]:
                    self.laber_list_D.append(i)
            except:
                self.laber_list_S.append(i)

    def one_hot_prepare(self):
        #self.train_dataset_loc 将文本替换为0，1，2
        #self.laber_dir_S 为替换的字典表

        for laber_S in self.laber_list_S:
            temp_S = self.train_dataset[laber_S].value_counts().index.values
            dir_temp_S = {}

            for index, value in enumerate(temp_S):
                dir_temp_S[value] = index
            self.laber_dir_S[laber_S] = dir_temp_S
            self.train_dataset_loc[laber_S] = self.train_dataset_loc[laber_S].map(dir_temp_S)

    def laber_str(self):
        #self.laber_list_SA 为需要one_hot编码的字段list
        #self.laber_list_SA 为不需要one_hot编码的字段list
        if self.Id in self.laber_list_S:
            self.laber_list_S.remove(self.Id)
        if self.Id in self.laber_list_D:
            self.laber_list_D.remove(self.Id)
        if self.target in self.laber_list_S:
            self.laber_list_S.remove(self.target)
        if self.target in self.laber_list_D:
            self.laber_list_D.remove(self.target)
        logging.info("{0}:文本型特征为{1}".format(self.now_time(), self.laber_list_S))
        logging.info("{0}:数值型特征为{1}".format(self.now_time(), self.laber_list_D))
        if self.one_hot_param is None or self.one_hot_param == 'off':
            self.laber_list_SA = self.laber_list_S

        else:
            self.laber_list_SA = self.one_hot_param
            self.laber_list_SB = copy.deepcopy(self.laber_list_S)
            for i in self.laber_list_SA:
                self.laber_list_SB.remove(i)



    def imputer(self):

        if self.laber_list_SA is not None and self.laber_list_SA != []:
            self.train_dataset_loc_SA = self.train_dataset_loc[self.laber_list_SA]
            imp_S = Imputer(missing_values='NaN', strategy='median', axis=0)
            imp_S.fit(self.train_dataset_loc_SA)
            self.train_dataset_loc_S = imp_S.transform(self.train_dataset_loc_SA).astype('int')

        if self.laber_list_SB is not None and self.laber_list_SB != []:
            self.train_dataset_loc_SB = self.train_dataset_loc[self.laber_list_SB]
            imp_S = Imputer(missing_values='NaN', strategy='median', axis=0)
            imp_S.fit(self.train_dataset_loc_SB)
            self.train_dataset_loc_SB = imp_S.transform(self.train_dataset_loc_SB).astype('int')

        if self.laber_list_D != []:
            self.train_dataset_loc_D = self.train_dataset_loc[self.laber_list_D]
            imp_D = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imp_D.fit(self.train_dataset_loc_D)
            self.train_dataset_loc_D = imp_D.transform(self.train_dataset_loc_D).astype('float32')
            if self.laber_list_SB is not None:
                self.train_dataset_loc_D = np.hstack((self.train_dataset_loc_D, self.train_dataset_loc_SB))

        elif self.laber_list_D == [] and self.laber_list_SB is not None and self.laber_list_SB != []:
            self.train_dataset_loc_D = self.train_dataset_loc_SB

    def one_hot_deal(self):
        self.ohe = OneHotEncoder()

        if self.train_dataset_loc_S is not None and self.one_hot_param != 'off':
            logging.info("{0}:对{1}进行one_hot编码".format(self.now_time(), self.laber_list_SA))
            self.ohe.fit(self.train_dataset_loc_S)
            self.train_dataset_loc_S = self.ohe.transform(self.train_dataset_loc_S).toarray()

    def standardScaler(self):
        #数据标准化
        self.std_D = None
        self.std_S = None
        if self.train_dataset_loc_D is not None:
            self.std_D = StandardScaler()
            if self.pca_param_d is not None:
                self.train_dataset_loc_D = self.std_D.fit_transform(self.train_dataset_loc_D)

        if self.train_dataset_loc_S is not None:

            self.std_S = StandardScaler()
            if self.pca_param_s is not None:
                self.train_dataset_loc_S = self.std_S.fit_transform(self.train_dataset_loc_S)

    def normalizer(self):
        #数据归一化
        if self.train_dataset_loc_D is not None:
            self.train_dataset_loc_D = Normalizer().fit_transform(self.train_dataset_loc_D)

    def PCA(self):
        #pca降维
        #白化一般用于图片处理，这里暂不使用
        if self.pca_param_d is None or self.train_dataset_loc_D is  None:
            logging.info("{0}:数值型不进行降维".format(self.now_time()))
        else:
            logging.info("{0}:数值型降维前维度:{1},信息保留比例{2}".format(self.now_time(),
                len(self.train_dataset_loc_D[0]), self.pca_param_d))
            self.pca_D = PCA(n_components=self.pca_param_d, svd_solver='full')
            self.pca_D.fit(self.train_dataset_loc_D)
            self.train_dataset_loc_D = self.pca_D.transform(self.train_dataset_loc_D)
            logging.info("{0}:数值型降维后维度:{1}".format(self.now_time(),
                len(self.train_dataset_loc_D[0])))

        if self.pca_param_s is None or self.train_dataset_loc_S is None:
            logging.info("{0}:文本型不进行降维".format(self.now_time()))
        else:
            logging.info("{0}:文本型降维前维度:{1},信息保留比例{2}".format(self.now_time(),
                len(self.train_dataset_loc_S[0]), self.pca_param_s))
            self.pca_S = PCA(n_components=self.pca_param_s, svd_solver='full')
            self.pca_S.fit(self.train_dataset_loc_S)
            self.train_dataset_loc_S = self.pca_S.transform(self.train_dataset_loc_S)
            logging.info("{0}:文本型降维后维度:{1}".format(self.now_time(),
                                                          len(self.train_dataset_loc_S[0])))


    def reshape(self):
        #合并数据

        if self.train_dataset_loc_D is not None and self.train_dataset_loc_S is not None:
            self.X_train = np.hstack((self.train_dataset_loc_D, self.train_dataset_loc_S))
        elif self.train_dataset_loc_D is not None:
            self.X_train = self.train_dataset_loc_D
        else:
            self.X_train = self.train_dataset_loc_S


    def creat_model(self):
        #建立默认模型
        if self.model == 'R':
            self.rf = XGBRegressor()
        elif self.model == 'C':
            self.rf = XGBClassifier()

    def skl_cv(self):
        logging.info("{0}:正在进行网格搜索".format(self.now_time()))
        if self.model == 'C':
            grid_search = GridSearchCV(estimator=self.rf, param_grid=self.cv_param, scoring='accuracy')
            grid_search.fit(self.X_train, self.Y_train)
            logging.info("{0}:最优参数:{1}".format(self.now_time(), grid_search.best_params_))
            logging.info("{0}:最优参数acc结果:{1}".format(self.now_time(), grid_search.best_score_))
            self.rf = XGBClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                    max_depth=grid_search.best_params_['max_depth'],
                                    min_child_weight=grid_search.best_params_['min_child_weight'],
                                    gamma=grid_search.best_params_['gamma'],
                                    learning_rate=grid_search.best_params_['learning_rate'])

        elif self.model == 'R':
            grid_search = GridSearchCV(estimator=self.rf, param_grid=self.cv_param, scoring='neg_mean_absolute_error')

            grid_search.fit(self.X_train, self.Y_train)
            logging.info("{0}:最优参数:{1}".format(self.now_time(), grid_search.best_params_))
            logging.info("{0}:最优参数R平方结果:{1}".format(self.now_time(), grid_search.best_score_))
            self.rf = XGBRegressor(n_estimators=grid_search.best_params_['n_estimators'],
                                    max_depth=grid_search.best_params_['max_depth'],
                                    min_child_weight=grid_search.best_params_['min_child_weight'],
                                    gamma=grid_search.best_params_['gamma'],
                                    learning_rate=grid_search.best_params_['learning_rate'])


    def fit(self):
        #训练数据并保存
        self.rf.fit(self.X_train, self.Y_train)

        model_save = self.rf, self.laber_list_S, self.laber_dir_S, self.ohe, self.Id, self.target,\
                     self.laber_list_SA, self.laber_list_SB, self.pca_D, self.pca_S, \
                     self.one_hot_param, self.laber_list_D, self.std_D, self.std_S,\
                     self.pca_param_d, self.pca_param_s
        pickle.dump(model_save, open('model_save.pickle', 'wb'))
        logging.info("{0}:已保存为文件:{1}".format(self.now_time(), 'model_save.pickle'))

class Amtl_predict(object):
    '''
    预测类，默认导入为csv文件，需要手动打开，也可直接提取self.result自定义保存
    '''
    def __init__(self, test_pandas, model_path, Show_title=True,math_format=0):
        '''

        :param test_pandas: 测试类的pandas文件
        :param model_path: Amtl产生的model_save.pickle
        :param Show_title: 打印的csv文件是否带有标题 默认True
        :param math_format: csc文件打印的小数位数 默认0，整数
        '''
        self.Show_title = Show_title
        self.math_format = math_format
        self.test_dataset = test_pandas
        self.model_path = model_path
        self.load_param()                                   #载入训练模型参数
        self.test_dataset_loc = self.test_dataset.loc[:, ]  #中间表
        self.Id_list = self.test_dataset.filter(regex=self.Id).values
        self.test_dataset_loc_D = None
        self.test_dataset_loc_S = None
        self.one_hot_prepare()
        self.imputer()
        self.one_hot_deal()
        self.standardScaler()
        #self.normalizer()
        self.PCA()
        self.reshape()
        self.rf_predict()
        #self.print_csv()

    def load_param(self):
        self.rf, self.laber_list_S, self.laber_dir_S, self.ohe, self.Id, self.target, \
        self.laber_list_SA, self.laber_list_SB, self.pca_D, self.pca_S,\
        self.one_hot_param, self.laber_list_D, self.std_D, self.std_S,\
        self.pca_param_d, self.pca_param_s\
            = pickle.load(open(self.model_path, mode='rb'))

    def now_time(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def one_hot_prepare(self):
        #self.train_dataset_loc 将文本替换为0，1，2
        #self.laber_dir_S 为替换的字典表

        for laber_S in self.laber_list_S:
            temp_S = self.test_dataset[laber_S].value_counts().index.values
            dir_temp_S = {}

            for index, value in enumerate(temp_S):
                dir_temp_S[value] = index
            self.laber_dir_S[laber_S] = dir_temp_S
            self.test_dataset_loc[laber_S] = self.test_dataset_loc[laber_S].map(dir_temp_S)

    def imputer(self):

        if self.laber_list_SA is not None and self.laber_list_SA != []:
            self.test_dataset_loc_SA = self.test_dataset_loc[self.laber_list_SA]
            imp_S = Imputer(missing_values='NaN', strategy='median', axis=0)
            imp_S.fit(self.test_dataset_loc_SA)
            self.test_dataset_loc_S = imp_S.transform(self.test_dataset_loc_SA).astype('int')

        if self.laber_list_SB is not None and self.laber_list_SB != []:
            self.test_dataset_loc_SB = self.test_dataset_loc[self.laber_list_SB]
            imp_S = Imputer(missing_values='NaN', strategy='median', axis=0)
            imp_S.fit(self.test_dataset_loc_SB)
            self.test_dataset_loc_SB = imp_S.transform(self.test_dataset_loc_SB).astype('int')

        if self.laber_list_D != []:
            self.test_dataset_loc_D = self.test_dataset_loc[self.laber_list_D]
            imp_D = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imp_D.fit(self.test_dataset_loc_D)
            self.test_dataset_loc_D = imp_D.transform(self.test_dataset_loc_D).astype('float32')
            if self.laber_list_SB is not None:
                self.test_dataset_loc_D = np.hstack((self.test_dataset_loc_D, self.test_dataset_loc_SB))

    def one_hot_deal(self):
        if self.test_dataset_loc_S is not None and self.one_hot_param != 'off':
            self.test_dataset_loc_S = self.ohe.transform(self.test_dataset_loc_S).toarray()

    def standardScaler(self):
        #数据标准化
        if self.test_dataset_loc_D is not None:
            if self.pca_param_d is not None:
                self.test_dataset_loc_D = self.std_D.fit_transform(self.test_dataset_loc_D)
        if self.test_dataset_loc_S is not None:
            if self.pca_param_s is not None:
                self.test_dataset_loc_S = self.std_S.fit_transform(self.test_dataset_loc_S)

    def normalizer(self):
        #数据归一化,扩展keras时使用
        self.test_dataset_loc_D = Normalizer().fit_transform(self.test_dataset_loc_D)

    def PCA(self):
        if self.test_dataset_loc_D is not None:
            if self.pca_param_d is not None:
                self.test_dataset_loc_D = self.pca_D.transform(self.test_dataset_loc_D)
        if self.test_dataset_loc_S is not None:
            if self.pca_param_s is not None:
                self.test_dataset_loc_S = self.pca_S.transform(self.test_dataset_loc_S)

    def reshape(self):
        if self.test_dataset_loc_D is not None and self.test_dataset_loc_S is not None:
            self.X_test = np.hstack((self.test_dataset_loc_D, self.test_dataset_loc_S))
        elif self.test_dataset_loc_D is not None:
            self.X_test = self.test_dataset_loc_D
        else:
            self.X_test = self.test_dataset_loc_S

    def rf_predict(self):
        self.result = self.rf.predict(self.X_test)

    def print_csv(self):
        with open("submit.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 先写入columns_name
            if self.Show_title:
                writer.writerow([self.Id, self.target])
            for id, predict in zip(self.Id_list, self.result):
                predict = '%.{0}f'.format(self.math_format) % predict
                writer.writerow([id[0], predict])