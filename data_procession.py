import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# 加载数据
data = pd.read_csv('./data/Telco-Customer-Churn.csv')
# print(data.head())      # 查看头10行
# print(data.tail())
# print(data.info())     # 查看字段信息


# 1.数据预处理

# 1.1缺失值的处理,查看有没有那个特征有缺失值
# print(data.isnull().any())

# readme中的字段描述提到，TotalCharges中有空格字符
# 查看TotalCharges的缺失值
# print(data[data['TotalCharges'] == ' '])

# TotalCharges格式为字符串，转换成数值（浮点），不可转换的空格字符转换成NaN
data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce')
# print("此时totalcharges类型是浮点型", data['TotalCharges'].dtype == 'float')
# print("totalcharges中缺失样本数：", data['TotalCharges'].isnull().sum())

# 缺失值填充    用pandas fillna或者sklearn imputer
# 方式一：填充均值/0/众数/中值等
# fill_data = data['TotalCharges'].fillna(0).to_frame()

# 方式二：
# 在这里我们根据实际业务场景的字段描述可以发现，MonthlyCharges【每月费用】 和TotalCharges【总费用】之间应该存在一定的关系，同时我们发现缺省值对应的数据tenure【入网月数】全部是0，且在整个数据集中tenure为0与TotalCharges为缺失值是一一对应的。
# 结合实际业务分析，这些样本对应的客户可能入网当月就流失了，但仍然要收取当月的费用，因此总费用即为该用户的每月费用（MonthlyCharges）。因此本案例我们最终采用MonthlyCharges的数值对TotalCharges进行填充。
data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'])
# print(data[data['tenure'] == 0][['MonthlyCharges', 'TotalCharges']])    # 观察处理后缺失值变化情况




# 1.2异常值的处理,查看数值类特征的统计信息
# print(data.describe())
# SeniorCitizen【是否为老年人】取值只有0和1，是离散特征, 因此只有tenure、MonthlyCharges及经过处理的TotalCharges是数值特征，继续结合箱型图进行分析：

import seaborn as sns
import matplotlib.pyplot as plt    # 可视化
# 分析百分比特征
fig = plt.figure(figsize=(15,6)) # 建立图像

# tenure特征
ax1 = fig.add_subplot(311)    # 子图1
list1 = list(data['tenure'])
ax1.boxplot(list1, vert=False, showmeans=True, flierprops = {"marker":"o","markerfacecolor":"steelblue"})
ax1.set_title('tenure')

# MonthlyCharges特征
ax2 = fig.add_subplot(312)    # 子图2
list2 = list(data['MonthlyCharges'])
ax2.boxplot(list2, vert=False, showmeans=True, flierprops = {"marker":"o","markerfacecolor":"steelblue"})
ax2.set_title('MonthlyCharges')

# TotalCharges
ax3 = fig.add_subplot(313)    # 子图3
list3 = list(data['TotalCharges'])
ax3.boxplot(list3, vert=False, showmeans=True, flierprops = {"marker":"o","markerfacecolor":"steelblue"})
ax3.set_title('TotalCharges')
plt.tight_layout(pad=1.5)    # 设置子图之间的间距
# plt.show() # 展示箱型图
# 由上图的箱型图可以看出来tenure、MonthlyCharges及经过处理的TotalCharges特征均不含离群点【即异常值】。




# 2.可视化分析

# 2.1流失客户占比,观察是否存在类别不平衡现象
p = data['Churn'].value_counts()    # 目标变量正负样本的分布
# # 绘制饼图
plt.figure(figsize=(10,6))
patches, l_text, p__text = plt.pie(p, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0,0.1))
# l_text是饼图对着的文字的大小，p_text是饼图内文字的大小
for t in p__text:
    t.set_size(15)
for t in l_text:
    t.set_size(15)
# plt.show()  # 展示图像




# 2.2基本特征对客户流失的影响
### 性别、是否老年人、是否有配偶、是否有家属等特征对客户流失的影响
baseCols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
for i in baseCols:
    cnt = pd.crosstab(data[i], data['Churn'])    # 构建特征与目标变量的列联表
    cnt.plot.bar(stacked=True)    # 绘制堆叠条形图，便于观察不同特征值流失的占比情况
    # plt.show()    # 展示图像

# 由图可知：
# - 性别对客户流失基本没有影响；
# - 年龄对客户流失有影响，老年人流失占比高于年轻人；
# - 是否有配偶对客户流失有影响，无配偶客户流失占比高于有配偶客户；
# - 是否有家属对客户流失有影响，无家属客户流失占比高于有家属客户.

### 观察流失率与入网月数的关系
# 折线图
groupDf = data[['tenure', 'Churn']]    # 只需要用到两列数据
groupDf['Churn'] = groupDf['Churn'].map({'Yes': 1, 'No': 0})    # 将正负样本目标变量改为1和0方便计算
pctDf = groupDf.groupby(['tenure']).sum() / groupDf.groupby(['tenure']).count()    # 计算不同入网月数对应的流失率
pctDf = pctDf.reset_index()    # 将索引变成列

plt.figure(figsize=(10, 5))
plt.plot(pctDf['tenure'], pctDf['Churn'], label='Churn percentage')    # 绘制折线图
plt.legend()    # 显示图例
# plt.show()
# 由图可知：除了刚入网（tenure=0）的客户之外，流失率随着入网时间的延长呈下降趋势；当入网超过两个月时，流失率小于留存率，这段时间可以看做客户的适应期。




# 2.3业务特征对客户流失的影响
# 电话业务
posDf = data[data['PhoneService'] == 'Yes']
negDf = data[data['PhoneService'] == 'No']

fig = plt.figure(figsize=(10,4)) # 建立图像

ax1 = fig.add_subplot(121)
p1 = posDf['Churn'].value_counts()
ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax1.set_title('Churn of (PhoneService = Yes)')

ax2 = fig.add_subplot(122)
p2 = negDf['Churn'].value_counts()
ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax2.set_title('Churn of (PhoneService = No)')

plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
# 由图可知，是否开通电话业务对客户流失影响很小。


# 多线业务
df1 = data[data['MultipleLines'] == 'Yes']
df2 = data[data['MultipleLines'] == 'No']
df3 = data[data['MultipleLines'] == 'No phone service']

fig = plt.figure(figsize=(15,6)) # 建立图像

ax1 = fig.add_subplot(131)
p1 = df1['Churn'].value_counts()
ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax1.set_title('Churn of (MultipleLines = Yes)')

ax2 = fig.add_subplot(132)
p2 = df2['Churn'].value_counts()
ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax2.set_title('Churn of (MultipleLines = No)')

ax3 = fig.add_subplot(133)
p3 = df3['Churn'].value_counts()
ax3.pie(p3,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax3.set_title('Churn of (MultipleLines = No phone service)')

plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
# 由图可知，是否开通多线业务对客户流失影响很小。此外 MultipleLines 取值为 'No'和 'No phone service' 的两种情况基本一致，后续可以合并在一起。


# 互联网业务
cnt = pd.crosstab(data['InternetService'], data['Churn'])    # 构建特征与目标变量的列联表
cnt.plot.barh(stacked=True, figsize=(15,6))    # 绘制堆叠条形图，便于观察不同特征值流失的占比情况
# plt.show()    # 展示图像
# 由图可知，未开通互联网的客户总数最少，而流失比例最低（7.40%）；开通光纤网络的客户总数最多，流失比例也最高（41.89%）；开通数字网络的客户则均居中（18.96%）。可以推测应该有更深层次的因素导致光纤用户流失更多客户，下一步观察与互联网相关的各项业务。


# 与互联网相关的业务
internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for i in internetCols:
    df1 = data[data[i] == 'Yes']
    df2 = data[data[i] == 'No']
    df3 = data[data[i] == 'No internet service']

    fig = plt.figure(figsize=(10, 3))  # 建立图像
    plt.title(i)

    ax1 = fig.add_subplot(131)
    p1 = df1['Churn'].value_counts()
    ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 开通业务

    ax2 = fig.add_subplot(132)
    p2 = df2['Churn'].value_counts()
    ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 未开通业务

    ax3 = fig.add_subplot(133)
    p3 = df3['Churn'].value_counts()
    ax3.pie(p3, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 未开通互联网业务

    plt.tight_layout()  # 设置子图之间的间距
    # plt.show()  # 展示饼状图
# 由图可知：所有互联网相关业务中未开通互联网的客户流失率均为7.40%，可以判断原因是上述六列特征均只在客户开通互联网业务之后才有实际意义，因而不会影响未开通互联网的客户；
# 开通了这些新业务之后，用户的流失率会有不同程度的降低，可以认为多绑定业务有助于用户的留存；
# 'StreamingTV'和 'StreamingMovies'两列特征对客户流失基本没有影响。此外，由于 'No internet service' 也算是 'No' 的一种情况，因此后续步骤中可以考虑将两种特征值进行合并。




# 2.4合约特征对客户流失的影响
# 合约期限
df1 = data[data['Contract'] == 'Month-to-month']
df2 = data[data['Contract'] == 'One year']
df3 = data[data['Contract'] == 'Two year']

fig = plt.figure(figsize=(15,4)) # 建立图像

ax1 = fig.add_subplot(131)
p1 = df1['Churn'].value_counts()
ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax1.set_title('Churn of (Contract = Month-to-month)')

ax2 = fig.add_subplot(132)
p2 = df2['Churn'].value_counts()
ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax2.set_title('Churn of (Contract = One year)')

ax3 = fig.add_subplot(133)
p3 = df3['Churn'].value_counts()
ax3.pie(p3,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax3.set_title('Churn of (Contract = Two year)')

plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
# 由图可知：合约期限越长，用户的流失率越低。


# 是否采用电子结算
df1 = data[data['PaperlessBilling'] == 'Yes']
df2 = data[data['PaperlessBilling'] == 'No']

fig = plt.figure(figsize=(10,4)) # 建立图像

ax1 = fig.add_subplot(121)
p1 = df1['Churn'].value_counts()
ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax1.set_title('Churn of (PaperlessBilling = Yes)')

ax2 = fig.add_subplot(122)
p2 = df2['Churn'].value_counts()
ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax2.set_title('Churn of (PaperlessBilling = No)')

plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
# 由图可知：采用电子结算的客户流失率较高，原因可能是电子结算多为按月支付的形式。


# 付款方式
df1 = data[data['PaymentMethod'] == 'Bank transfer (automatic)']    # 银行转账（自动）
df2 = data[data['PaymentMethod'] == 'Credit card (automatic)']    # 信用卡（自动）
df3 = data[data['PaymentMethod'] == 'Electronic check']    # 电子支票
df4 = data[data['PaymentMethod'] == 'Mailed check']    # 邮寄支票

fig = plt.figure(figsize=(10,8)) # 建立图像

ax1 = fig.add_subplot(221)
p1 = df1['Churn'].value_counts()
ax1.pie(p1,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax1.set_title('Churn of (PaymentMethod = Bank transfer')

ax2 = fig.add_subplot(222)
p2 = df2['Churn'].value_counts()
ax2.pie(p2,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax2.set_title('Churn of (PaymentMethod = Credit card)')

ax3 = fig.add_subplot(223)
p3 = df3['Churn'].value_counts()
ax3.pie(p3,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax3.set_title('Churn of (PaymentMethod = Electronic check)')

ax4 = fig.add_subplot(224)
p4 = df4['Churn'].value_counts()
ax4.pie(p4,labels=['No','Yes'],autopct='%1.2f%%',explode=(0,0.1))
ax4.set_title('Churn of (PaymentMethod = Mailed check)')

plt.tight_layout(pad=0.5)    # 设置子图之间的间距
# plt.show() # 展示饼状图
# 由图可知：四种付款方式中采用电子支票的客户流失率远高于其他三种。


# 每月费用核密度估计图
plt.figure(figsize=(10, 5))    # 构建图像

negDf = data[data['Churn'] == 'No']
sns.distplot(negDf['MonthlyCharges'], hist=False, label= 'No')
posDf = data[data['Churn'] == 'Yes']
sns.distplot(posDf['MonthlyCharges'], hist=False, label= 'Yes')
plt.legend()
# plt.show()    # 展示图像

# 总费用核密度估计图
plt.figure(figsize=(10, 5))    # 构建图像

negDf = data[data['Churn'] == 'No']
sns.distplot(negDf['TotalCharges'], hist=False, label= 'No')
posDf = data[data['Churn'] == 'Yes']
sns.distplot(posDf['TotalCharges'], hist=False, label= 'Yes')
plt.legend()
# plt.show()    # 展示图像

# 由图可知：客户的流失率的基本趋势是随每月费用的增加而增长，这与实际业务较为符合；当客户的总费用积累越多，流失率越低，这说明这些客户已经称为稳定的客户，不会轻易流失；
# 此外，当每月费用处于70～110之间时流失率较高。




# 3.特征工程

# 3.1特征提取
### 连续数值特征标准化 --  将数值特征缩放到同一尺度下，避免对特征重要性产生误判。【树模型可以不做处理】
from sklearn.preprocessing import StandardScaler    # 导入标准化库

'''
注：
新版本的sklearn库要求输入数据是二维的，而例如data['tenure']这样的Series格式本质上是一维的
如果直接进行标准化，可能报错 "ValueError: Expected 2D array, got 1D array instead"
解决方法是变一维的Series为二维的DataFrame，即多加一组[]，例如data[['tenure']]
'''
scaler = StandardScaler()
data[['tenure']] = scaler.fit_transform(data[['tenure']])
data[['MonthlyCharges']] = scaler.fit_transform(data[['MonthlyCharges']])
data[['TotalCharges']] = scaler.fit_transform(data[['TotalCharges']])
# print(data[['tenure', 'MonthlyCharges', 'TotalCharges']].head())    # 观察此时的数值特征
# print(data[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())



### 类别特征编码   - - 离散特征的处理
# 首先将部分特征值进行合并,这样做的好处是只剩下Yes，No两个值，这样就不用做one-hot编码了
data.loc[data['MultipleLines']=='No phone service', 'MultipleLines'] = 'No'

internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in internetCols:
    data.loc[data[i]=='No internet service', i] = 'No'
# print("MultipleLines特征还有%d条样本的值为 'No phone service'" % data[data['MultipleLines']=='No phone service'].shape[0])
# print("OnlineSecurity特征还有%d条样本的值为 'No internet service'" % data[data['OnlineSecurity']=='No internet service'].shape[0])


# 部分类别特征只有两类取值，可以直接用0、1代替；另外，可视化过程中发现有四列特征对结果影响可以忽略，后续直接删除
# 选择特征值为‘Yes’和 'No' 的列名，phoneservice,streamingtv..这几个后面要删除，就没转换成0,1
# 这里只是从data中取出数据然后进行了删除，并没有删除data的数据。
encodeCols = list(data.columns[3: 17].drop(['tenure', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract']))
for i in encodeCols:
    data[i] = data[i].map({'Yes': 1, 'No': 0})    # 用1代替'Yes’，0代替 'No'
# 顺便把目标变量也进行编码
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})


# 其他无序的类别特征采用one-hot编码
onehotCols = ['InternetService', 'Contract', 'PaymentMethod']
churnDf = data['Churn'].to_frame()    # 取出目标变量列，以便后续进行合并
featureDf = data.drop(['Churn'], axis=1)    # 所有特征列

for i in onehotCols:
    onehotDf = pd.get_dummies(featureDf[i],prefix=i)   #one_hot编码，prefix是前缀
    featureDf = pd.concat([featureDf, onehotDf],axis=1)    # 编码后特征拼接到去除目标变量的数据集中

data = pd.concat([featureDf, churnDf],axis=1)    # 拼回目标变量，确保目标变量在最后一列
data = data.drop(onehotCols, axis=1)    # 删除原特征列




# 3.2特征选择
'''
customerID'特征对模型预测不起贡献，可以直接删除。
'gender'、'PhoneService'、'StreamingTV' 和 'StreamingMovies' 则在可视化环节中较为明显地观察到其对目标变量的影响较小，因此也删去这四列特征。
'''
# 删去无用特征 'customerID'、'gender'、 'PhoneService'、'StreamingTV'和'StreamingMovies'
data = data.drop(['customerID', 'gender', 'PhoneService', 'StreamingTV', 'StreamingMovies'], axis=1)


'''
此外，还可以采用相关系数矩阵衡量连续型特征之间的相关性、用卡方检验衡量离散型特征与目标变量的相关关系等等，
从而进行进一步的特征选择。例如，可以对数据集中的三列连续型数值特征 'tenure', 'MonthlyCharges', 'TotalCharges' 计算相关系数，
其中 'TotalCharges' 与其他两列特征的相关系数均大于0.6，即存在较强相关性，因此可以考虑删除该列，以避免特征冗余。
'''
nu_fea = data[['tenure', 'MonthlyCharges', 'TotalCharges']]    # 选择连续型数值特征计算相关系数
nu_fea = list(nu_fea)    # 特征名列表
pearson_mat = data[nu_fea].corr(method='spearman')    # 计算皮尔逊相关系数矩阵

plt.figure(figsize=(8,8)) # 建立图像
sns.heatmap(pearson_mat, square=True, annot=True, cmap="YlGnBu")    # 用热度图表示相关系数矩阵
# plt.show() # 展示热度图

data = data.drop(['TotalCharges'], axis=1)

# print(data.head())



# 数据保存
data.to_csv("./data/processed_data.csv",index=False)
