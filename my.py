
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib


import matplotlib.pyplot as plt


#一、模型训练与预测
# 加载鸢尾花数据集150个样本
iris = load_iris()
X = iris.data  # 特征矩阵(150,4)   4个特征预测3类
y = iris.target  # 目标向量(150,1)
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #将数据集20%划分为测试集
#random_state：是一个随机种子，用于控制数据集划分的随机过程。通过指定相同的随机种子，可以保证每次运行代码时得到相同的划分结果，以便结果的可重复性。
# 创建决策树分类器
clf = DecisionTreeClassifier()

# 训练模型
clf.fit(X_train, y_train)
# 预测测试集
y_pred = clf.predict(X_test)  #测试集有30个样本



#二、计算各种性能指标来显示模型性能，这里计算的很多来说明性能好，其实一个也可以
# 1.计算准确率
accuracy = accuracy_score(y_test, y_pred)  #accuracy_score 函数接受两个参数：真实标签（y_true）和预测标签（y_pred）。它会比较这两个标签数组的对应元素，并计算预测正确的样本数
#accuracy 计算公式为 准确率 = 预测正确的样本数 / 总样本数
print("准确率：", accuracy)

# 2.计算精确率 
precision = precision_score(y_test, y_pred,average='macro') #精确率（Precision）：表示预测为正类别的样本中，实际为正类别的比例
print("精确率：", precision)
# 公式：
# Precision = TP / (TP + FP)
# 其中，TP（True Positive）表示真正例（预测为正类别且实际为正类别的样本数），FP（False Positive）表示假正例（预测为正类别但实际为负类别的样本数）。


# 3.计算召回率
recall = recall_score(y_test, y_pred,average='macro') #召回率（Recall）：表示实际为正类别的样本中，被正确预测为正类别的比例
print("召回率：", precision)
# 召回率表示实际为正类别的样本中，被正确预测为正类别的比例。
# 公式：
# Recall = TP / (TP + FN)
# 其中，FN（False Negative）表示假负例（预测为负类别但实际为正类别的样本数）。



# 4.计算F1分数
f1 = f1_score(y_test, y_pred,average='macro') #F1值：综合考虑了精确率和召回率，是精确率和召回率的调和平均值。F1值越高，模型在预测正类别和负类别上的表现越好。
print("F1分数：", precision)
# F1 分数综合了精确率和召回率，是它们的调和平均值。F1 分数用于综合评估模型在预测正类别和负类别上的表现。
# 公式：
# F1 = 2 * (Precision * Recall) / (Precision + Recall)


#average选项含义说明
# 1.average=None:

# 精确率（Precision）：返回一个数组，包含每个类别的精确率。
# 召回率（Recall）：返回一个数组，包含每个类别的召回率。
# F1 分数：返回一个数组，包含每个类别的 F1 分数。

# 2.average='micro':
# 精确率（Precision）：计算总体的真正例与假正例，然后计算精确率。
# 公式：
# Precision = TP_micro / (TP_micro + FP_micro)
# 其中，TP_micro 表示总体的真正例，FP_micro 表示总体的假正例。
# 召回率（Recall）：与精确率相同，因为计算的是总体的真正例与假正例。
# F1 分数：与精确率和召回率相同，因为它们都是基于总体的真正例和假正例计算的。

# 3.average='macro':
# 精确率（Precision）：对每个类别计算精确率，并取它们的算术平均值。
# 公式：
# Precision_macro = (Precision_class1 + Precision_class2 + ... + Precision_classN) / N
# 其中，Precision_classi 表示第 i 个类别的精确率，N 表示类别的总数。
# 召回率（Recall）：对每个类别计算召回率，并取它们的算术平均值。
# F1 分数：对每个类别计算 F1 分数，并取它们的算术平均值。

# 4.average='weighted':
# 精确率（Precision）：对每个类别计算精确率，并按照类别的支持样本数进行加权平均。
# 公式：
# Precision_weighted = (Precision_class1 * Support_class1 + Precision_class2 * Support_class2 + ... + Precision_classN * Support_classN) / (Support_class1 + Support_class2 + ... + Support_classN)
# 其中，Precision_classi 表示第 i 个类别的精确率，Support_classi 表示第 i 个类别的支持样本数，N 表示类别的总数。
# 召回率（Recall）：对每个类别计算召回率，并按照类别的支持样本数进行加权平均。
# F1 分数：对每个类别计算 F1 分数，并按照类别的支持样本数进行加权平均。




