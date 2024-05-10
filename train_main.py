import time
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors    # k近邻算法
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC as SVC
from sklearn.ensemble import RandomForestClassifier as RF
from lightgbm import LGBMClassifier as LGB  # pip install lightGBM

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 设置忽略警告
import warnings
warnings.filterwarnings('ignore')

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

### 设置不使用科学计数法  #为了直观的显示数字，不采用科学计数法
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.float_format', lambda x: '%.2f' % x)



# 定义Smote类
class Smote:
    def __init__(self, samples, N, k):  #samples是少数样本
        self.n_samples, self.n_attrs = samples.shape
        self.N = N  #采样倍率
        self.k = k
        self.samles = samples
        self.newindex = 0

    def over_sampling(self):
        N = int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))   # 存放新合成样本
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samles)   #建KD-tree
        for i in range(len(self.samles)):   #对每个样本求k近邻
            nnarray = neighbors.kneighbors(self.samles[i].reshape(1, -1), return_distance=False)[0]
            self._populate(N, i, nnarray)
        return self.synthetic

    # 为少数类样本选择k个最近邻中的N个，并生成N个合成样本
    def _populate(self, N, i, nnarray):  # i:第i个样本       nnarray:这个样本的k近邻
        for i in range(N):
            nn = random.randint(0, self.k - 1)
            dif = self.samles[nnarray[nn]] - self.samles[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samles[i] + gap * dif
            self.newindex+=1


def up_samples():
    # 加载前面处理好的数据
    data = pd.read_csv('./data/processed_data.csv')


    # 每个正样本用SMOTE方法随机生成两个新的样本
    posDf = data[data['Churn'] == 1].drop(['Churn'], axis=1)    # 共1869条正样本, 取其所有特征列(删除标签列)
    posArray = posDf.values    # pd.DataFrame -> np.array, 以满足SMOTE方法的输入要求
    newPosArray = Smote(posArray, 2, 5).over_sampling()
    newPosDf = pd.DataFrame(newPosArray)    # np.array -> pd.DataFrame


    # 调整为正样本在数据集中应有的格式
    newPosDf.columns = posDf.columns    # 还原特征名
    cateCols = list(newPosDf.columns.drop(['tenure', 'MonthlyCharges']))   # 提取离散特征名组成的列表
    for i in cateCols:
        newPosDf[i] = newPosDf[i].apply(lambda x: 1 if x >= 0.5 else 0)    # 将特征值变回0、1二元数值  由于二类别的数据的值都是1或者0，多类别的数据已经做了one-hot处理，所以这种方法可行
    newPosDf['Churn'] = 1    # 添加目标变量列


    # 为保证正负样本平衡，从新生成的样本中取出（5174 - 1869 = 3305）条样本，并加入原数据集进行shuffle操作。也完全可以不做这个操作
    # 构建类别平衡的数据集
    from sklearn.utils import shuffle

    # newPosDf = newPosDf[:3305]    # 直接选取前3305条样本
    newPosDf = newPosDf.sample(n=3305)  # 随机抽取3305条样本
    data = pd.concat([data, newPosDf])  # 竖向拼接
    data = shuffle(data).reset_index(drop=True)  # 样本打乱
    # print(data["Churn"].value_counts())
    # print("此时数据集的规模为：", data.shape)
    # print(data.head())
    data.to_csv("./data/processed_smote.csv", index=False)


# - K折交叉验证，这里可以获得预测结果，后面用于评估模型的性能，从而选择最佳的模型
def kFold_cv(X, Y, Classifier, **kwargs):
    """
    :param X: 特征
    :param y: 目标变量
    :param classifier: 分类器
    :param **kwargs: 参数
    :return: 预测结果
    """
    kf = KFold(n_splits=5, shuffle=True)  # 5折交叉验证
    y_pred = np.zeros(len(Y))   #初始化y_pred
    start = time.time()
    for train_index, test_index in kf.split(X):   #kf.split得到5组训练集和测试集
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = Y[train_index]
        clf = Classifier(**kwargs)
        clf.fit(X_train, y_train)   #模型训练
        y_pred[test_index] = clf.predict(X_test)   #模型预测

    print(f"use time:{time.time() - start}")
    return y_pred



# 模型选型与训练
# - 我们使用逻辑回归、SVC、随机森林、LightBGM，然后从中选择最优的
def train():
    # 获取X,Y,加载数据
    data = pd.read_csv("./data/processed_smote.csv")
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # 这里仅作演示的baseline，参数没仔细调，这里大家可以调完参数选择最优化模型进行训练/或者进行模型融合，再进行预测
    # 进行K折交叉验证
    lr_pred = kFold_cv(x_train.values, y_train.values, LR, penalty='l2', C=1.0)
    svc_pred = kFold_cv(x_train.values, y_train.values, SVC, C=1.0)
    rf_pred = kFold_cv(x_train.values, y_train.values, RF, n_estimators=100, max_depth=10)
    lgb_pred = kFold_cv(x_train.values, y_train.values, LGB, learning_rate=0.1, n_estimators=1000, max_depth=10)

    # 模型评估
    scoreDf = pd.DataFrame(columns=['LR', 'SVC', 'RandomForest', 'LGB'])
    pred = [lr_pred, svc_pred, rf_pred, lgb_pred]
    for i in range(len(pred)):
        r = recall_score(y_train.values, pred[i])
        p = precision_score(y_train.values, pred[i])
        f1 = f1_score(y_train.values, pred[i])
        scoreDf.iloc[:, i] = pd.Series([r, p, f1])

    scoreDf.index = ['Recall', 'Precision', 'F1-score']
    # print(scoreDf)


    # LGB模型效果最好，我们选择LGB模型单模型进行训练，并且输出其特征重要性
    lgb = LGB(learning_rate=0.1, n_estimators=1000, max_depth=10)
    lgb.fit(x_train, y_train)
    y_train_pred = lgb.predict(x_train)
    y_test_pred = lgb.predict(x_test)
    # print(classification_report(y_train, y_train_pred))
    # print(classification_report(y_test, y_test_pred))

    # 特征重要度，树模型都有特征重要度
    feature_importances = lgb.feature_importances_

    import matplotlib.pyplot as plt
    import lightgbm
    fig, ax = plt.subplots(figsize=(20, 20))
    lightgbm.plot_importance(lgb, ax=ax, height=0.5, grid=False)
    plt.title("Feature importances")
    # plt.show()

    # 训练好后对模型进行存储
    import joblib
    joblib.dump(lgb, "./models/lgb.m")
    # joblib.load()




    ## 在模型预测阶段，可以结合预测出的概率值决定对哪些客户进行重点留存：
    # 加载前面处理好的数据
    data = pd.read_csv("./data/processed_data.csv")
    X, Y = data.iloc[:, :-1], data.iloc[:, -1]
    pred_prob = lgb.predict_proba(X)
    pred_prob = np.round(pred_prob, 1)  # 对预测出的概率值保留两位小数，便于分组观察

    # 合并预测值和真实值
    probDf = pd.DataFrame(pred_prob)
    churnDf = pd.DataFrame(Y)
    df1 = pd.concat([probDf, churnDf], axis=1)
    df1.columns = ['prob_0', 'prob_1', 'churn']

    # 分组计算每种预测概率值所对应的真实流失率
    df1.drop("prob_0", inplace=True, axis=1)
    group = df1.groupby(['prob_1'])
    cnt = group.count()  # 每种概率值对应的样本数
    true_prob = group.sum() / group.count()  # 真实流失率
    df2 = pd.concat([cnt, true_prob], axis=1).reset_index()
    df2.columns = ['prob_1', 'cnt', 'true_prob']
    print(df2)
    # 由表可知：预测流失率越大的客户中越有可能真正发生流失。对运营商而言，可以根据各预测概率值分组的真实流失率设定阈值进行决策。
    # 例如，假设阈值为true_prob = 0.5，即优先关注真正流失为50 % 以上的群体，也就表示运营商可以对预测结果中大于等于0.6的客户进行重点留存。

    
if __name__ == '__main__':
    # up_samples()         上采样
    train()

