# Liner Regression

## Introduction:

​			线性模型（Linear [Model](https://so.csdn.net/so/search?q=Model&spm=1001.2101.3001.7020)），是机器学习中的一类算法的总称，其形式化定义为：通过给定的样本数据集![\large D](https://private.codecogs.com/gif.latex?%5Clarge%20D)，线性模型试图学习到这样的一个模型，使得对于任意的输入，模型的预测输出![\large f(x)](https://private.codecogs.com/gif.latex?%5Clarge%20f%28x%29)能够表示为输入特征向量![\large x](https://private.codecogs.com/gif.latex?%5Clarge%20x)的线性函数。

## Deduction:

#### 	Start：

​			如果给定一个特征向量 x=(x<sub>1</sub>,x<sub>2</sub>,...,x<sub>i</sub>)<sup>T</sup>，函数f(x)是关于特征向量x的一个映射，我们期望可以通过这个x-->f(x)的映射来预测我们对于未来某一个事件的尽可能准确的答案。

​			假设我们已经收集了以往该事件的一些数据，并且通过一些手段，得到了每一条数据对应的标记，记这个数据的集合为样本数据集D,D={x<sub>i</sub>,f(x<sub>i</sub>)}。其中f(x)=w<sub>1</sub>x<sub>1</sub>,+w<sub>2</sub>x<sub>2</sub>+...+w<sub>i</sub>x<sub>i</sub>,我们将他写成矩阵的形式：f(x)=w<sup>T</sup>x+b，其中w=(w<sub>1</sub>,w<sub>2</sub>,...w<sub>i</sub>)<sup>T</sup>和b为模型的参数，我们期望通过以往发生的事情来找到这样的一组w，b，来预测未来发生的一些事情。

​			为了求解(w,b),我们需要定义一个决策函数来驱动w，b的值是需要增大还是减小。这里我们用Gredient_Descent的方法，也就是均方误差。

#### 	Gredient_Descent：

​			均方误差的公式为：L(w,b)=1/2Σ(f(x<sub>i</sub>)-y<sub>i</sub>)<sup>2</sup>,为了方便计算，我们让b=w*1，这样的话，对于特征向量多了一个为‘1’的特征，而对w来说，多了一个w<sub>0</sub>,这样的话，我们的代价函数就变成了一个关于w的函数，即L(w)==1/2Σ(f(x<sub>i</sub>)-y<sub>i</sub>)<sup>2</sup>,我们对于其优化的目标就是让其越小越好，即找到一组w尽可能使函数L(w)的最小。为了达到这一目的，我们对其求梯度，让其每一个参数，沿着梯度下降。考虑到可能每一次下降的步伐如果太快的话，我们可能会与最低点失之交臂，因此我们定义一个较小的数值乘以其倒数。即：w<sub>j</sub>=w<sub>j</sub>-aΣ(f(x<sub>i</sub>)-y<sub>i</sub>)<sup>2</sup>x<sub>i</sub><sub>j</sub>

#### End

![end](.\end.png)



## Code：

#### 	Requirement:

​						numpy,matplotlab.random,math

​		

```python
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
```







​			



































​	





​			








