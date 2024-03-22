# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensor_power_method import power_method
from train_test import kfold_validation
from train_test import permute_test
from train_test import sparsity_selection
from sklearn.model_selection import train_test_split
from scipy import stats

def generate_z(A1, M,n,time_series_length):
    '''
    生成仿真数据集
    '''
    random_state = np.random.get_state()# 保存当前随机状态
    np.random.seed(123)
    B = np.identity(M+1)
    random_matrix = np.random.rand(M, M)
    Q, R = np.linalg.qr(random_matrix)  # 使用QR分解生成正交矩阵
    if Q[0, 0] < 0:    # 确保Q是一个正交矩阵，如果为负则取反.因为QR分解的结果有时可能使得Q的首个元素为负
        Q = -Q
    B[1:, 1:] = Q
    A2 = np.diag(np.ones(M-3))
    A = np.block([
        [A1, np.zeros((4,M-3))],
        [np.zeros((M-3,4)), A2]
    ])

    white_noise_sd = 0.1    #(可作为参数讨论敏感性，调小，①信噪比，|a|/noise_sd ②n)
    # 初始化 Z_0
    np.random.set_state(random_state)# 恢复之前保存的随机状态
    Z_0 = np.random.normal(0, 1,  size=(M + 1, n))
    B_transpose_A_B = B.T @ A @ B
    Z = [None] * time_series_length
    Z[0] = Z_0
    for t in range(1, time_series_length):
        white_noise = np.zeros((M+1, n))
        white_noise[0, :] = np.random.normal(loc=0, scale=white_noise_sd, size=n)
        # 剩下的行生成服从多维正态分布的白噪声,多维正态分布需要一个均值向量和协方差矩阵.我们假定每一个维度的均值都是0，并且不同维度之间也是独立的
        mean_vector = np.zeros(M)
        covariance_matrix = np.eye(M)  # 单位矩阵，表明各维度是独立的
        for i in range(n):
            white_noise[1:, i] = np.random.multivariate_normal(mean_vector, covariance_matrix*white_noise_sd)
        Z[t] = B_transpose_A_B @ Z[t-1] + white_noise

    return B, Z


def test_simu_data(X_t_1, X_t, Y_t_1, Y_t, alpha, beta, gamma, B, Z, t):

    # method3：用 permutation 检验
    p_total, p_alpha, p_beta, p_gamma1, p_gamma2 = permute_test(X_t_1, X_t, Y_t_1, Y_t, alpha, beta, gamma, num_perm = 10000)
    print('oriiiii:!!!\n total_sig_p:'+str(p_total)+'\n alpha_sig_p:'+str(p_alpha) + '\n beta_sig_p:'+str(p_beta) + '\n gamma_sig_p:'+str(p_gamma1)+'\n'+str(p_gamma2))


def main():
    n = 3000    #被试个数
    time_series_length = 103     #生成的时间点
    num_dataset = 30   #随机生成num_dataset个数据
    t = 101       #取第t个时间点的数据进行乘幂法
    M = 68      #特征个数
    truncation = np.round(M*np.arange(1/np.sqrt(M), 1, 0.1)).astype(int)
    dim = np.arange(0,68)
    shift = 0
    ops = ['+','+','+']
    A1 = np.array([[0.5,0.5,0,0.4],[0,0.5,0,0],[0.5,0,0.5,0],[0.4,0,0,0.5]]) # (1,1,1)T########8
    
    count_alpha_p = 0; count_beta_p = 0; count_gamma1_p = 0; count_gamma2_p = 0
    count_test_alpha_p = 0; count_test_beta_p = 0; count_test_gamma1_p = 0; count_test_gamma2_p = 0
    for num in np.arange(0,num_dataset):
        # # 1.生成仿真数据集
        B, Z = generate_z(A1,M,n,time_series_length)
        X_t_1 = ((Z[t-1])[1:]).T
        X_t = ((Z[t])[1:]).T
        X_bl = pd.DataFrame(data=X_t)
        Y_t_1 = ((Z[t-1])[0].reshape(Z[t-1][0].shape[0],1))
        Y_t = ((Z[t])[0].reshape(Z[t][0].shape[0],1))
        alpha = B[1,1:].reshape(B[1,1:].shape[0],1)
        beta = B[2,1:].reshape(B[2,1:].shape[0],1)
        gamma = B[3,1:].reshape(B[3,1:].shape[0],1)
        # eigenvalues = np.linalg.eigvals(A1)        # 测试 A 是否稳定
        # is_stable = np.all(np.abs(eigenvalues) < 1)
        # print("矩阵A是否稳定:", is_stable)
        # test_simu_data(X_t_1, X_t, Y_t_1, Y_t, alpha, beta, gamma, B, Z, t)        # 测试 simulate data 是否符合我们想生成的显著性
        X_t_1_train,X_t_1_test, X_t_train,X_t_test, Y_t_1_train,Y_t_1_test, Y_t_train, Y_t_test = train_test_split(X_t_1, X_t, Y_t_1, Y_t, test_size=0.2)
        
        # # 2.选择稀疏性参数
        best_spar = sparsity_selection(X_t_1_train, X_t_train, Y_t_1_train, Y_t_train, alpha,beta,gamma,M, dim, X_bl, ops,truncation,shift,num_power_trials=100,num_perm=1000)
        print("best_spar:",best_spar)

        # # 3.交叉验证以及测试典型变量的显著性（确定最佳k参数后！！）
        p_total,p_alpha,p_beta,p_gamma1,p_gamma2 = kfold_validation(X_t_1_train, X_t_train, Y_t_1_train, Y_t_train, M, dim, X_bl, best_spar, ops,shift, num_power_trials=100,num_perm = 1000)
        best_alpha, best_beta, best_gamma, best_obj, _, _,_,_,_,_ = power_method(
            best_spar, ops, M, X_t, X_t_1, Y_t, Y_t_1, X_bl, dim, shift, 
            max_iter=1000, convergence_threshold=1e-6, total_trials=100
        )
        print('cross_validation_totest_composite\n'+'total_sig_p:'+str(p_total)+'\n alpha_sig_p:'+str(p_alpha) + '\n beta_sig_p:'+str(p_beta) + '\n gamma1_sig_p:'+str(p_gamma1) + '\n gamma2_sig_p:'+str(p_gamma2))
        if(p_alpha<0.05):count_alpha_p+=1
        if(p_beta<0.05):count_beta_p+=1
        if(p_gamma1<0.05):count_gamma1_p+=1
        if(p_gamma2<0.05):count_gamma2_p+=1

        # # 4. 对test数据集使用αβγ参数，计算相关性
        alpha_test_corr, alpha_test_p = stats.pearsonr((X_t_1_test@(best_alpha.to_numpy())).reshape(-1), Y_t_test.reshape(-1))
        beta_test_corr, beta_test_p = stats.pearsonr((X_t_test@(best_beta.to_numpy())).reshape(-1), Y_t_1_test.reshape(-1))
        gamma_test_corr1, gamma_test_p1 = stats.pearsonr((X_t_1_test@(best_gamma.to_numpy())).reshape(-1), Y_t_test.reshape(-1))
        gamma_test_corr2, gamma_test_p2 = stats.pearsonr((X_t_test@(best_gamma.to_numpy())).reshape(-1), Y_t_1_test.reshape(-1))
        print('test result\n'+'alpha_sig_p:'+str(alpha_test_p) + '\n beta_sig_p:'+str(beta_test_p) + '\n gamma1_sig_p:'+str(gamma_test_p1) + '\n gamma2_sig_p:'+str(gamma_test_p2))
        if(alpha_test_p<0.05):count_test_alpha_p+=1
        if(beta_test_p<0.05):count_test_beta_p+=1
        if(gamma_test_p1<0.05):count_test_gamma1_p+=1
        if(gamma_test_p2<0.05):count_test_gamma2_p+=1
    
    print(num_dataset,'次实验里的count_sig_composite\n'+'count_alpha_p:'+str(count_alpha_p) + '\n count_beta_p:'+str(count_beta_p) + '\n count_gamma1_p:'+str(count_gamma1_p) + '\n count_gamma2_p:'+str(count_gamma2_p))
    print(num_dataset,'次实验里的count_test\n'+'count_test_alpha_p:'+str(count_test_alpha_p) + '\n count_test_beta_p:'+str(count_test_beta_p) + '\n count_test_gamma1_p:'+str(count_test_gamma1_p) + '\n count_test_gamma2_p:'+str(count_test_gamma2_p))

if __name__== "__main__" :
    main()

