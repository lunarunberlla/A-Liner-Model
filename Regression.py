import numpy as np
import random
import math
import matplotlib.pyplot as plt
#初始化随机数种子
random.seed(100)
learningRate=0.00000001 #初始化学习率
weight=random.randint(1,10) #为权重赋值
bais=random.randint(1,10) #为bais赋值

source_data_file=open(r'RegressionData.csv') #打开数据集文件
Data=source_data_file.read() #将文件数据读取到Data中
#   对数据进行处理
Data=Data.split('\n')
Data=Data[1:len(Data)-1]

_x,_y=[],[]

for i in Data:
    _x_midele=[]
    _x_midele.append(int(i.split(',')[0]))
    _x_midele.append(1)         #进行矩阵运算：将_x_train,_x_test的维度拓展‘1’，添加一个值为‘1’的特征，方便进行bais的运算
    _x.append(_x_midele)
    _y.append(int(i.split(',')[1]))


#_x=[[400, 1], [450, 1], [484, 1], [500, 1], [510, 1], [525, 1], [540, 1], [549, 1], [558, 1], [590, 1], [610, 1], [640, 1], [680, 1], [750, 1], [900, 1]]
#_y=[80, 89, 92, 102, 121, 160, 180, 189, 199, 203, 247, 250, 259, 289, 356]

#划分测试集与训练集并将数据转换为矩阵

_x_train,_x_test=np.array(_x[0:int(len(_x)*0.8)+1]),np.array(_x[int(len(_x)*0.8)+1:len(_x)])
_y_train,_y_test=np.array(_y[0:int(len(_x)*0.8)+1]),np.array(_y[int(len(_x)*0.8)+1:len(_x)])

#_x_train=[[400, 1], [450, 1], [484, 1], [500, 1], [510, 1], [525, 1], [540, 1], [549, 1], [558, 1], [590, 1], [610, 1], [640, 1], [680, 1]]
#_x_test=[[750, 1], [900, 1]]
#_y_train=[80, 89, 92, 102, 121, 160, 180, 189, 199, 203, 247, 250, 259]
#_y_test=[289, 356]

_weight_bais=np.array([[weight],[bais]]) #创建参数列向量

#Loss=Σ(_x_train*[_weight_bais]T - _y_train)*(_x_train*[_weight_bais]T - _y_train)
#求 MIN(Loss(weight*,bais*))
#梯度下降算法
#使用np.dot 进行矩阵运算
epoch=20 #定义训练的epoch



for i in range(epoch):
    #参数更新
    weight,bais=weight-learningRate*np.sum(np.dot(((np.dot(_x_train,_weight_bais)-_y_train)),(_x_train[:,0]))),bais-learningRate*np.sum(((np.dot(_x_train,_weight_bais)-_y_train)))
    _weight_bais = np.array([[weight], [bais]])
    Loss=(math.sqrt(np.sum(((np.dot(_x_train,_weight_bais)-_y_train)*(np.dot(_x_train,_weight_bais)-_y_train)))))/len(_x_train)
    print(Loss)



#画图
#画原来的曲线图
X=_x_train[:,0]
Y=_y_train
plt.plot(X,Y,'g-',label='Source')
y1=_weight_bais[0]*370+_weight_bais[1]
y2=_weight_bais[0]*700+_weight_bais[1]
#画拟合的函数图
X=[400,700]
Y=[y1,y2]
plt.plot()
plt.plot(X,Y,'r-',label='Batch_Gredient_Descent')
#定义画板
plt.xlabel("x")
plt.ylabel("y")
plt.title("LinearRegression")
plt.legend(loc='upper left')
plt.show()