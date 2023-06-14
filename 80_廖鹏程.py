import xgboost as xgboost
#导入所需要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#读取数据集
df=pd.read_csv("https://raw.githubusercontent.com/beidimeiyou1/data-analysis-project/main/Sleep_health_and_lifestyle_dataset.csv",encoding="gbk")
#查看数据集基本信息
df.info()
#查看是否有缺失值
print(df.isnull().sum())
#查看数据集的前五行
print(df.head())
# 检查数据集重复情况
print(df.duplicated().sum())
#查看数据集的描述性统计
print(df.describe())
df_new=df.copy()
#对分类型变量考察其分布#列名做成列表
cat_cols = []
for each in df_new.columns.tolist():
    if df_new[each].dtype == 'object' and each != '血压':
        cat_cols.append(each)
        print(df_new[each].value_counts().to_frame())
# 分类型变量编码
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_new[col])
    df_new[col] = le.transform(df_new[col])
print(df_new.head())
#将高低压进行分离
xueya=df_new['血压'].str.split('/',expand=True)
xueya.columns=['高压','低压']
xueya=xueya.astype(int)
df_new=pd.concat([df_new,xueya],axis=1)
df_new.info()
# 按照性别分组，计算各数值型变量的均值、中位数、标准差和最大最小值
stats = df_new.groupby('性别')[['年龄', '睡眠时长', '睡眠质量', '身体活动水平', '压力水平', '心率', '高压', '低压']].agg(['mean', 'median', 'std', 'min', 'max'])
# 按照性别分组，计算各数值型变量的均值、中位数、标准差和最大最小值
stats = df_new.groupby('年龄')[['性别', '睡眠时长', '睡眠质量', '身体活动水平', '压力水平', '心率', '高压', '低压']].agg(['mean', 'median', 'std', 'min', 'max'])
#分类变量的分布展示
#以性别和职业为标签
plt.figure(figsize=(10,6))
sns.countplot(x='性别',hue='职业',data=df)
plt.title('分别以性别和职业为分类标签的分布情况')
plt.show()
#以性别和BMI为标签
plt.figure(figsize=(10,6))
sns.countplot(x='性别',hue='BMI',data=df)
plt.title('分别以性别和BMI为分类标签的分布情况')
plt.show()
#考察数值型变量的分布：箱线图
#年龄的箱线图
plt.boxplot(df_new['年龄'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'orangered','facecolor':'pink'})
plt.title("年龄")
plt.show()
#睡眠时长的箱线图
plt.boxplot(df_new['睡眠时长'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("睡眠时长")
plt.show()
#睡眠质量的箱线图
plt.boxplot(df_new['睡眠质量'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("睡眠质量")
plt.show()
#身体活动水平的箱线图
plt.boxplot(df_new['身体活动水平'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("身体活动水平")
plt.show()
#压力水平的箱线图
plt.boxplot(df_new['压力水平'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("压力水平")
plt.show()
#心率的箱线图
plt.boxplot(df_new['心率'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("心率")
plt.show()
#高压的箱线图
plt.boxplot(df_new['高压'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("高压")
plt.show()
#低压的箱线图
plt.boxplot(df_new['低压'],notch=True,patch_artist=True,meanline=True,showmeans=True,boxprops={'color':'blue','facecolor':'pink'})
plt.title("低压")
plt.show()
#考察数值型变量的分布：密度图
# 年龄的密度图
sns.kdeplot(df_new['年龄'], shade=True)
plt.title("年龄")
plt.show()
# 睡眠时长的密度图
sns.kdeplot(df_new['睡眠时长'], shade=True)
plt.title("睡眠时长")
plt.show()
# 睡眠质量的密度图
sns.kdeplot(df_new['睡眠质量'], shade=True)
plt.title("睡眠质量")
plt.show()
# 身体活动水平的密度图
sns.kdeplot(df_new['身体活动水平'], shade=True)
plt.title("身体活动水平")
plt.show()
# 压力水平的密度图
sns.kdeplot(df_new['压力水平'], shade=True)
plt.title("压力水平")
plt.show()
# 心率的密度图
sns.kdeplot(df_new['心率'], shade=True)
plt.title("心率")
plt.show()
# 高压的密度图
sns.kdeplot(df_new['高压'], shade=True)
plt.title("高压")
plt.show()
# 低压的密度图
sns.kdeplot(df_new['低压'], shade=True)
plt.title("低压")
plt.show()
#考察数值型变量的分布：小提琴图
# 年龄的小提琴图
sns.violinplot(x='年龄', data=df_new)
plt.title("年龄")
plt.show()
# 睡眠时长的小提琴图
sns.violinplot(x='睡眠时长', data=df_new)
plt.title("睡眠时长")
plt.show()
# 睡眠质量的小提琴图
sns.violinplot(x='睡眠质量', data=df_new)
plt.title("睡眠质量")
plt.show()
# 身体活动水平的小提琴图
sns.violinplot(x='身体活动水平', data=df_new)
plt.title("身体活动水平")
plt.show()
# 压力水平的小提琴图
sns.violinplot(x='压力水平', data=df_new)
plt.title("压力水平")
plt.show()
# 心率的小提琴图
sns.violinplot(x='心率', data=df_new)
plt.title("心率")
plt.show()
# 高压的小提琴图
sns.violinplot(x='高压', data=df_new)
plt.title("高压")
plt.show()
# 低压的小提琴图
sns.violinplot(x='低压', data=df_new)
plt.title("低压")
plt.show()
# 查看数据集中因素之间的相关性
sns.pairplot(df_new[df_new.columns.tolist()[1:]])
plt.show()
#删除多余变量
target=['睡眠质量','睡眠时长']
df_new.drop(columns=['血压'],inplace=True)
#分别将两个变量作为因变量，查看各因素的重要性
for i in range(len(target[0:2])):
    y=df_new[target[i]]
    print(y)
    #去除因变量
    x=df_new.iloc[:,~df_new.columns.isin(target)]
    print(x)
    #随机森林回归
    model=RandomForestRegressor()
    model.fit(x,y)
    print('在'+target[i]+'作为因变量时，各因素重要性为：')
    plt.figure()
    #绘制子图
    plt.subplot(2,1,i+1)
    #以每个特征的重要性画图
    plt.imshow(model.feature_importances_.reshape(-1,1))
    #设置y轴坐标和标签
    plt.yticks(range(len(x.columns.tolist())),x.columns.tolist())
    plt.xticks(range(1))
    plt.xlabel(target[i])
    plt.colorbar()
    plt.show()
# 查看数据集各个变量的贡献度以及相关性
# 设置绘图风格
sns.set_style('whitegrid')
# 设置画板尺寸
plt.subplots(figsize = (15,10))
#画热力图
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
corr = df_new.corr()
sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(20,220,n=200),center=0)
# Give title.
plt.title("Heatmap of all the Features")
plt.show()
#建立模型
#决策树
df_new['BMI'] = LabelEncoder().fit_transform(df_new['BMI'])
x=df_new.loc[:,~df_new.columns.isin(['血压','职业','BMI'])]
#多元线性回归方程
#  定义因变量
y = df_new['睡眠质量']
# 定义自变量
X = df_new.drop(['年龄', '身体活动水平','压力水平', '心率','每日步数', '高压', '低压'], axis=1)
# 将常数项添加到自变量中
X = sm.add_constant(X)
# 拟合多元线性回归模型
model = sm.OLS(y, X).fit()
# 打印模型摘要
print(model.summary())
# 定义因变量
y = df_new['睡眠市场']
# 定义自变量
X = df_new.drop(['年龄', '身体活动水平','压力水平', '心率','每日步数', '高压', '低压'], axis=1)
# 将常数项添加到自变量中
X = sm.add_constant(X)
# 拟合多元线性回归模型
model = sm.OLS(y, X).fit()
# 打印模型摘要
print(model.summary())
# 检测多重共线性
# 定义自变量
X = df_new.drop(['年龄', '身体活动水平','压力水平', '心率','每日步数', '高压', '低压'], axis=1)
# 计算VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
# 打印结果
print(vif)
