import numpy as np

#确定两个类的样本数量num1, num2,并告知属性类条件分布是否满足朴素贝叶斯
def gene_gauss_Samples(num_A,num_B,satisfy_naive):

    #指定分布中两个类的均值，并控制其协方差矩阵
    mean_A = [0.6,0.4]
    mean_B = [-0.6,-0.4]
    cov_AB = 0.15
    if satisfy_naive: # 直接调用sklearn中的方法生成高斯分布
        X1 = np.random.multivariate_normal(mean=mean_A, cov=[[0.2, cov_AB], [cov_AB, 0.2]], size=num_A).T
        X2 = np.random.multivariate_normal(mean=mean_B, cov=[[0.2, cov_AB], [cov_AB, 0.2]], size=num_B).T
    else: #由于sklearn中上述方法默认协方差矩阵为对角阵，我们该用numpy中的方法
        X1 = np.random.multivariate_normal(mean=mean_A, cov=[[0.2, 0], [0, 0.2]], size=num_A).T
        X2 = np.random.multivariate_normal(mean=mean_B, cov=[[0.2, 0], [0, 0.2]], size=num_B).T
    Y1 = np.array([[1 for i in range(num_A)]])
    Y2 = np.array([[0 for i in range(num_B)]])
    #X1、X2均为(2,n)矩阵，Y1、Y2为(1,n)矩阵
    return X1,Y1,X2,Y2
