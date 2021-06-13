# coding=UTF-8
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from time import *

#全局变量
train_datafile_name = 'file_data\\two_sklearn_1'
Test_datafile_name ='file_data\\two_sklearn_1'
option = 1 #核函数选择 1-高斯核 0-线性核
sigma = 1 #高斯核参数
maxIter = 100000 #最大迭代次数
total_sample = 1000 #训练样本总数
Test_num = 1000 #测试数据总数
feature_size = 2 #单条数据特征数
w=np.zeros([feature_size]) #W矩阵
b = 0.0 #模型的b值 f(x) = wx + b
C = 100 #优化目标重点的惩罚参数
toler = 0.001 #软间隔优化目标的ε 允许样本错误的参数
label = np.zeros([total_sample])  #样本标签初始化矩阵 矩阵类型 1 x total_sample
feature = np.zeros([total_sample, feature_size])  #样本特征初始化矩阵 矩阵类型 total_sample x feature_size
All_alpha = np.zeros((total_sample)) #存放所有alpha 矩阵类型 1 x total_sample
ALL_E = np.zeros([total_sample,2]) #存放所有误差值矩阵 矩阵类型 total_sample x 2 第一列判断Ei是否有效--默认都为0 第二列记录Ei值
Kernal = np.zeros([total_sample,total_sample]) #核函数矩阵初始化 total_sample x total_sample

#从文件获取训练数据
#返回特征集，标签值
#数据格式 [label] [index]:[value] ...
def data_load(file):
    i = 0
    for line in open(file, "r"):
        data = line.split(" ")
        if int(float(data[0])) <= 0:
            label[i] = -1
        else:
            label[i] = 1
        for fea in data[1:]:
            index,value = fea.split(":")
            feature[i,int(index)-1] = float(value)
        i = i + 1
    print("训练数据加载完成\n")
def Test_data_load(file,line):
    Data_X = np.zeros([line,feature_size])
    Data_Y = np.zeros([line])
    i = 0
    for line in open(file, "r"):
        data = line.split(" ")
        if int(float(data[0])) <= 0:
            Data_Y[i] = -1
        else:
            Data_Y[i] = 1
        for fea in data[1:]:
            index,value = fea.split(":")
            Data_X[i,int(index)-1] = float(value)
        i = i + 1
    print("测试数据加载完成\n")
    return Data_X,Data_Y
#计算核函数Kernal矩阵
#option 1高斯核
#option 0线性核
def K(Xi,Xj,option):
    if option:
        Xi_Xj = Xi-Xj
        return np.exp((Xi_Xj@Xi_Xj)/(-2 * sigma**2))
    else:
        return  Xi@Xj
#f(xi)
def f(x):
    fxi = 0.0
    if option:
        fxi = np.dot(Kernal[:,x].T,np.multiply(All_alpha,label)) + b
    else:
        fxi = np.multiply(All_alpha,label).T @ (feature @ feature[x,:]) + b
    return fxi
#计算Ei=f（xi）-yi
def E(i):
    return f(i) - float(label[i])
#更新Ek 由于alpha值改变
def Updata_E(i,value):
    ALL_E[i] = [1,value]
#对求偏导求出的alpa2 根据alpa范围进行更新
def update_alpha2(alpha2, High, Low):
    if alpha2 > High :
        return High
    elif alpha2 < Low :
        return Low
    else :
        return alpha2
#根据第一个alpha选取第二个alpha
def randomtest(i):
    temp = i
    while (temp == i):
        temp = random.randint(0, total_sample-1)
    return temp
#判断i样本是否满足KKT条件 若不违反KKT,则更新alpha[i]和alpha[j]
def in_(i):
    global b
    Ei = E(i)
    #KKT条件判别
    if(((label[i] * Ei < -toler)and(All_alpha[i] < C)) or ((label[i] * Ei > toler)and(All_alpha[i] > 0))):
        #内循环，选取|Ei-Ej|差值最大作为索引
        # j,Ej = select_alpha(i,Ei)
        j = randomtest(i)
        print(j)
        Ej = E(j)
        alphaIold = All_alpha[i].copy()
        alphaJold = All_alpha[j].copy()
        #对alpha范围内进行线性规划
        if label[i] == label[j]:
            Low = max(0,alphaJold+alphaIold-C)
            High = min(C,alphaJold+alphaIold)
        else:
            Low = max(0, alphaJold - alphaIold)
            High = min(C, C + alphaJold - alphaIold)
        #此情况alpha不改变，返回0
        if Low == High:
            return 0
        #计算η值=K11+K22-2K12=||X1-X2||^2 > 0 η=0则无法更新alpha
        eta = Kernal[i,i] + Kernal[j,j] - 2.0*Kernal[i,j]
        if eta <= 0:
            return 0
        #alpha求偏导
        alpha_j_new = alphaJold + label[j] *(Ei-Ej)/eta
        #根据范围进行alpha值更新
        alpha_j_new = update_alpha2(alpha_j_new,High,Low)
        # 若是变化量过小,视为没有改变,返回0
        if abs(alpha_j_new - alphaJold) < 0.00001:
            return 0
        #更新alphai
        alpha_i_new = alphaIold + label[i] * label[j] * (alphaJold - alpha_j_new)
        bi = float(-Ei + label[i] * Kernal[i, i] * (alphaIold - alpha_i_new) + label[j] * Kernal[i, j] * (alphaJold - alpha_j_new) + b)
        bj = float(-Ej + label[i] * Kernal[i, j] * (alphaIold - alpha_i_new) + label[j] * Kernal[j, j] * (alphaJold - alpha_j_new) + b)
        if 0 < alpha_i_new < C:  # 若是alpha_i_new是支持向量,那么根据公式,此时bi=b
            b = bi
        elif 0 < alpha_j_new  < C:  # 同理
            b = bj
        else:  # 若都不是支持向量,取均值
            b = (bi + bj) / 2.0
        #更新All_alpha[i]和All_alpha[j]以及Ei,Ej值
        All_alpha[i] = alpha_i_new
        All_alpha[j] = alpha_j_new
        Updata_E(i,Ei)
        Updata_E(j,Ej)
        #print("更新alpha")
        return 1
    return 0
#根据计算alpha使用测试集进行检验
def Test(Test_feature):
    global w
    # 测试集变量
    Test_total_sample = Test_feature.shape[0]  # 测试样本总数
    Test_label = np.zeros([Test_total_sample])  # 测试集真实标签
    # Test_feature = np.zeros([Test_total_sample, feature_size])  # 测试数据 特征矩阵 Test_total_sample x feature_size
    Test_prediction = np.zeros([Test_total_sample])  # 模型对测试样本预测值
    Test_prediction_label = np.zeros([Test_total_sample])  # 模型对测试样本预测标签
    #只有alpha > 0对模型起作用 即支持向量
    Vector_alpha_index = np.nonzero(All_alpha)[0] #返回(index元组,dtype) 取第一值为index元组
    Vector_alpha = All_alpha[Vector_alpha_index] #将支持向量对应alpha取出来 1 x 支持向量数
    Vector_label = label[Vector_alpha_index] #将支持向量对应的标签取出来 1 x 支持向量数
    Vector_feature = feature[Vector_alpha_index,:] #支持向量的样本值 支持向量数 x feature_size
    Kernal_Test = np.zeros([Vector_alpha_index.shape[0], Test_total_sample])  # 支持向量与测试样本的核函数矩阵 支持向量数 x 测试样本数
    # 计算支持向量和测试样本的核函数
    for m in range(Vector_alpha_index.shape[0]):
        for n in range(Test_total_sample):
            Kernal_Test[m,n] = K(Vector_feature[m],Test_feature[n],option)
    #不同核函数
    if option:#高斯核
        for i in range(Test_total_sample):
            for j in range(Vector_alpha_index.shape[0]):
                Test_prediction[i] += Vector_alpha[j]*Vector_label[j]*Kernal_Test[j,i]#Kernal_Test[j,i]
            Test_prediction[i] += b
    else:#线性核
        #根据支持向量计算w矩阵 1xfeature_size
        for i in range(Vector_alpha_index.shape[0]):
            w += np.multiply(np.multiply(Vector_alpha[i],Vector_label[i]),Vector_feature[i])
        #预测数据计算处理 fx = Σaiyixix+b i∈S为所有支持向量
        for j in range(Test_total_sample):
            Test_prediction[j] = w@(Test_feature[j].T) + b#np.multiply(Vector_alpha,Vector_label) @ Kernal_Test[:,j] + b
    for index in range(Test_total_sample):
        if Test_prediction[index] > 0:
            Test_prediction_label[index] = 1
        else:
            Test_prediction_label[index] = -1
    return Test_prediction_label
#二维数据测试显示
def predict(matr):
    tu_label = np.zeros([matr.shape[0]])
    Vector_alpha_index = np.nonzero(All_alpha)[0]  # 返回(index元组,dtype) 取第一值为index元组
    Vector_alpha = All_alpha[Vector_alpha_index]  # 将支持向量对应alpha取出来 1 x 支持向量数
    Vector_label = label[Vector_alpha_index]  # 将支持向量对应的标签取出来 1 x 支持向量数
    Vector_feature = feature[Vector_alpha_index, :]  # 支持向量的样本值 支持向量数 x feature_size
    Kernal_Test = np.zeros([Vector_alpha_index.shape[0], matr.shape[0]])  # 支持向量与测试样本的核函数矩阵 支持向量数 x 测试样本数
    # 计算支持向量和测试样本的核函数
    for m in range(Vector_alpha_index.shape[0]):
        for n in range(matr.shape[0]):
            Kernal_Test[m, n] = K(Vector_feature[m], matr[n], option)  # K(m, n, option, sigma)  # Test_feature[m] @ Vector_feature[n]
    for i in range(matr.shape[0]):
        for j in range(Vector_alpha_index.shape[0]):
            tu_label[i] += Vector_alpha[j] * Vector_label[j] * Kernal_Test[j, i]
        tu_label[i] += b
    for index in range(matr.shape[0]):
        if tu_label[index] > 0:
            tu_label[index] = 1
        else:
            tu_label[index] = -1
    return tu_label
#画图，只实现二维数据画图
def show():
    min_x = feature[:,0].min()
    max_x = feature[:,0].max()
    min_y = feature[:, 1].min()
    max_y = feature[:, 1].max()
    #画数据点
    data_plus = feature[np.where(label==1)]
    data_minute = feature[np.where(label==-1)]
    plt.scatter(data_plus[:, 0].flatten(), data_plus[:, 1].flatten(), s=30, c='r', marker='o',zorder=2)
    plt.scatter(data_minute[:, 0].flatten(), data_minute[:, 1].flatten(), s=30, c='b', marker='s',zorder=2)
    #画分割线
    x = np.linspace(feature[:,0].min(), feature[:,0].max(), 100)
    # 不同核函数
    if option:#高斯核
        temp_X1, temp_X2 = np.mgrid[min_x:max_x:200j, min_y:max_y:200j]  # 生成网络采样点
        grid_test = np.stack((temp_X1.flat,temp_X2.flat) ,axis=1) #测试点 n X 2
        grid_hat = predict(grid_test)  # 预测分类值
        grid_hat = grid_hat.reshape(temp_X1.shape)  # 使之与输入的形状相同
        cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        plt.pcolormesh(temp_X1, temp_X2, grid_hat, shading='auto',cmap=cm_light)
    else:     #线性核
        line1 = (-w[0]*x-b)/w[1]
        line2 = (-w[0] * x - b + 1 + toler) / w[1]
        line3 = (-w[0] * x - b - 1 - toler) / w[1]
        plt.plot(x, line1)
        plt.plot(x, line2,c='r',linestyle='--')
        plt.plot(x, line3,c='b',linestyle='--')
    #图大小限制
    plt.ylim((min_y,max_y))
    plt.xlim((min_x, max_x))
    plt.show()
def main():
    #初始化训练数据及计算内积
    data_load(train_datafile_name)
    for i in range(total_sample):
        for j in range(total_sample):
            Kernal[i,j] = K(feature[i],feature[j],option)
    print(Kernal)
    print("开始优化")
    switch = True  # 用于控制全遍历或局部遍历的开关,局部遍历指,只遍历支持向量.True全遍历,False局部遍历
    alpha_changed = 0  # 用于记录全部alpha在本次迭代中,是否有所改变,若值为0,说明都无改变,若值大于0,说明存在改变,执行下一次迭代
    iters = 0  # 用于记录迭代次数
    begin_time = time()
    while iters < maxIter and ((alpha_changed > 0) or switch):  # 当迭代轮次超过最大值或者 遍历全集后alpha值无变化， 则跳出外循环，训练结束
        alpha_changed = 0  # 每次迭代，重置为0
        print("第"+str(iters)+"次迭代")
        if switch:  # 全遍历,验证每一个样本
            print("全遍历")
            for i in range(total_sample):
                alpha_changed += in_(i)  # in_返回0或1,分别表示无变化和有变化.只要有一个alpha发生过变化,则认为整个alpha集发生变化
                print("第%d次迭代 i = %d alpha改变个数 = %d全遍历"%(iters,i,alpha_changed))
            iters += 1  # 一次更新完毕，迭代次数+1
        else:  # 全遍历后,再遍历全部支持向量,直到全部支持向量的alpha无变化,再进行全遍历.若是这次全遍历,整个alpha集都无变化,则训练结束,不然再次遍历支持向量,如此循环.
            print("支持向量遍历")
            bound_alpha = [i for i, a in enumerate(All_alpha) if 0 < a < C]  # 获取全部支持向量的索引
            for i in bound_alpha:
                alpha_changed += in_(i)
                print("第%d次迭代 i = %d alpha改变个数 = %d支持向量遍历"%(iters,i,alpha_changed))
            iters += 1
        if switch:  # 全遍历后,进入支持向量的遍历
            switch = False
        elif alpha_changed == 0:  # 支持向量遍历后,若是全部支持向量的alpha无变化,则进行全遍历
            switch = True
        print("alpha = "+str(alpha_changed))
    end_time = time()#统计优化时间
    run_time = end_time - begin_time
    print("总迭代次数%d,时间%f"%(iters,run_time))
    print("核函数:" + ("高斯核\n" if option else "线性核\n"))
    Test_data = Test_data_load(Test_datafile_name,Test_num)
    prediction = Test(Test_data[0])
    print('测试样本预测值：')
    print(prediction)#模型预测值矩阵prediction 原始标签矩阵Test_data[1]
    #精度计算显示
    print("正在将原始标签与模型预测标签比较。。。")
    correct = 0 #统计正确个数
    for i in range(Test_num):
        if Test_data[1][i] == prediction[i]:
            correct += 1
    print("预测精度：%f"%(correct/Test_num))
    print("正在画图")
    show()#3维以上数据注释此语句，否则报错，只实现了二维数据画图
main()