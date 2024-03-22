import numpy as np
import pandas as pd
from semopy import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
res = list(range(0,1000000,1))
np.seterr(all='ignore')

def compute_G_matrices(X_t, X_t_1, Y_t, Y_t_1, alpha, beta, gamma, ops):
    '''
    定义计算矩阵A，B，C的函数(全矩阵)
    '''
    M = X_t_1.shape[1]
    Y_t = Y_t.reshape(Y_t.shape[0],1)
    Y_t_1 = Y_t_1.reshape(Y_t_1.shape[0],1)

    a12_2 = a13_2 = a14_2 = (X_t_1.T @ Y_t @ Y_t.T @ X_t_1)
    a21_2 = a31_2 = a41_2 = (X_t.T @ Y_t_1 @ Y_t_1.T @ X_t)
    a23 = (X_t.T @ X_t_1)
    a32 = (X_t_1.T @ X_t)
    
    A11 = (a12_2 - a21_2)
    A22 = (-a13_2 + a31_2)
    A33 = (a14_2 + a41_2)
    B1 = C1 = D1 = a23
    B2 = C2 = D2 = a32
    
    G_alpha = 1e-4*(0.5*(A11+A11.T)*(beta.T@beta)*(gamma.T@gamma) + (gamma.T@gamma)*(beta.T@A22@beta)*np.eye(M) + (beta.T@beta)*(gamma.T@A33@gamma) *np.eye(M))
    +1e-8*(-(gamma.T@gamma)*(B1@beta@beta.T@B1.T) - (gamma.T@gamma)*(B2@beta@beta.T@B2.T)
    - (beta.T@beta)*(C1@gamma@gamma.T@C1.T) - (beta.T@beta)*(C2@gamma@gamma.T@C2.T)
    - (beta.T@D1@gamma@gamma.T@D1.T@beta)*np.eye(M) - (beta.T@D2@gamma@gamma.T@D2.T@beta)*np.eye(M))

    G_beta = 1e-4*(A22*(alpha.T@alpha)*(gamma.T@gamma) + (gamma.T@gamma)*(alpha.T@A11@alpha)*np.eye(M) + (alpha.T@alpha)*(gamma.T@A33@gamma)*np.eye(M))
    +1e-8*(-(gamma.T@gamma)*(B1.T@alpha@alpha.T@B1) - (gamma.T@gamma)*(B2.T@alpha@alpha.T@B2)
    - (alpha.T@C1@gamma@gamma.T@C1.T@alpha)*np.eye(M) - (alpha.T@C2@gamma@gamma.T@C2.T@alpha)*np.eye(M)
    - (alpha.T@alpha)*(D1@gamma@gamma.T@D1.T) - (alpha.T@alpha)*(D2@gamma@gamma.T@D2.T))

    G_gamma = 1e-4*(A33*(alpha.T@alpha)*(beta.T@beta) + (beta.T@beta)*(alpha.T@A11@alpha)*np.eye(M) + (alpha.T@alpha)*(beta.T@A22@beta)*np.eye(M))
    +1e-8*(- (alpha.T@B1@beta@beta.T@B1.T@alpha)*np.eye(M) - (alpha.T@B2@beta@beta.T@B2.T@alpha)*np.eye(M) 
    - (beta.T@beta)*(C1@alpha@alpha.T@C1.T) - (beta.T@beta)*(C2@alpha@alpha.T@C2.T)
    - (alpha.T@alpha)*(D1@beta@beta.T@D1.T) - (alpha.T@alpha)*(D2@beta@beta.T@D2.T))

    obj = 5e-5*((beta.T@beta)*(gamma.T@gamma)*(alpha.T@A11@alpha) - (alpha.T@alpha)*(gamma.T@gamma)*(beta.T@A22@beta) + (alpha.T@alpha)*(beta.T@beta)*(gamma.T@A33@gamma))
    +5e-9*(- (gamma.T@gamma)*(alpha.T@B1@beta@beta.T@B1.T@alpha) - (gamma.T@gamma)*(alpha.T@B2@beta@beta.T@B2.T@alpha)
    - (beta.T@beta)*(alpha.T@C1@gamma@gamma.T@C1.T@alpha) - (beta.T@beta)*(alpha.T@C2@gamma@gamma.T@C2.T@alpha)
    - (alpha.T@alpha)*(beta.T@D1@gamma@gamma.T@D1.T@beta) - (alpha.T@alpha)*(beta.T@D2@gamma@gamma.T@D2.T@beta))

    return G_alpha, G_beta, G_gamma, obj


def truncate_vector(v, k):
    '''
    截断power_method
    '''
    mask = np.full(v.shape, False)   # 创建一个遮罩向量，初始时全部为False
    indices = np.argsort(np.abs(v.T))[0][-k:]    # 选择绝对值最大的k个分量的索引
    mask[indices] = True    # 将这k个分量对应在遮罩中的位置设置为True，其余为False
    truncated_v = v * mask    # 生成新的向量，仅保留最大k个分量，其他分量置零

    return truncated_v


def power_method(k, ops, M, X_t, X_t_1, Y_t, Y_t_1, X_bl, dim, shift, max_iter, convergence_threshold, total_trials):
    '''
    张量乘幂法
    '''

    converge_count = 0  # 计数器
    best_alpha = best_beta = best_gamma = np.zeros((1,68))
    best_i = best_trial = best_obj = -999
    max_objs = []
    alphas = []
    betas = []
    gammas = []

    for trial in range(total_trials):  # 开始多轮迭代

        # 初始化alpha, beta, gamma
        alpha = np.random.randn(M, 1)
        beta = np.random.randn(M, 1)
        gamma = np.random.randn(M, 1)
        # 归一化向量
        alpha /= np.linalg.norm(alpha)
        beta /= np.linalg.norm(beta)
        gamma /= np.linalg.norm(gamma)
        phi = np.concatenate((alpha, beta, gamma))

        # 迭代过程
        errors = []
        objs = []
        iter_times = []

        for i in range(max_iter):
            G_alpha, G_beta, G_gamma, obj = compute_G_matrices(X_t, X_t_1, Y_t, Y_t_1, alpha, beta, gamma, ops)
            G = np.block([[G_alpha, np.zeros((M, M)), np.zeros((M, M))],
                        [np.zeros((M, M)), G_beta, np.zeros((M, M))],
                        [np.zeros((M, M)), np.zeros((M, M)), G_gamma]])
            new_phi = G @ phi
            new_phi = new_phi/np.linalg.norm(new_phi)
            
            new_alpha, new_beta, new_gamma = new_phi[:M], new_phi[M:2*M], new_phi[2*M:]
            # 现在为alpha, beta, gamma应用截断
            new_alpha = truncate_vector(new_alpha, k)
            new_beta = truncate_vector(new_beta, k)
            new_gamma = truncate_vector(new_gamma, k)
            new_alpha, new_beta, new_gamma = new_alpha/np.linalg.norm(new_alpha), new_beta/np.linalg.norm(new_beta), new_gamma/np.linalg.norm(new_gamma)
            new_phi = np.concatenate((new_alpha, new_beta, new_gamma), axis=0)

            error = 3-(np.abs(new_alpha.T@alpha) + np.abs(new_beta.T@beta) + np.abs(new_gamma.T@gamma))
            if error < convergence_threshold:
                # print("SCF Converged")
                break
            errors.append(float(error))
            objs.append(float(obj))
            alpha, beta, gamma, phi = new_alpha, new_beta, new_gamma, new_phi

        iter_times.append(i)

        if i < (max_iter - 1):  # 如果迭代次数少于999次，则增加计数
            converge_count += 1
        
        # total_trial每次的最终结果
        max_objs.append(obj)
        alphas.append(alpha.T)
        betas.append(beta.T)
        gammas.append(gamma.T)

        # 输出结果
        print("Trial:", trial, "Iter times:", i, "max obj:", obj)
        
        # 记录收敛的最优化参数信息
        if obj > best_obj:
            best_trial = trial
            best_i = i
            best_obj = obj
            best_alpha = alpha.T
            best_beta = beta.T
            best_gamma = gamma.T
   
        # # 作图：max obj / errors随迭代次数的变化图
        # plt.rcParams['font.family'] = 'Arial'
        # plt.figure(figsize=(10, 8))  # 增大整个图表的尺寸

        # # 绘制max obj随迭代次数的变化图
        # plt.subplot(2, 1, 1)  # 创建第一个子图，占据上半部分
        # plt.plot(range(len(objs)), objs, marker='o', linestyle='-', color='blue', label='Max Objective')  # 绘制折线图
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.xlabel('Iteration',fontsize=20)  # x轴标签
        # plt.ylabel('Objective',fontsize=20)  # y轴标签
        # plt.title('Objective vs. Iteration',fontsize=20)  # 图像标题
        # plt.grid(True)  # 显示网格
        # plt.legend(fontsize=20)  # 显示图例

        # # 绘制errors随迭代次数的变化图
        # plt.subplot(2, 1, 2)  # 创建第二个子图，占据下半部分
        # plt.plot(range(len(errors)), errors, marker='x', linestyle='--', color='red', label='Error')  # 绘制折线图
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.xlabel('Iteration',fontsize=20)  # x轴标签
        # plt.ylabel('Error',fontsize=20)  # y轴标签
        # plt.title('Error vs. Iteration',fontsize=20)  # 图像标题
        # plt.grid(True)  # 显示网格
        # plt.legend(fontsize=20)  # 显示图例

        # plt.tight_layout(pad=3)  # 调整子图布局，增加pad参数确保子图之间有更多空间
        # plt.yscale('log')  # 如果错误值差异很大，考虑使用对数刻度

        # plt.savefig('plot/simu_iter/trial_' + str(trial) + '_dim_' + str(dim[0]) + "_" + str(dim.shape) +'_objective_error_iteration.png')  # 保存图片
        # plt.show()  # 显示图形
        # plt.close()  # 关闭图形


    max_objs_std = np.round(np.std(max_objs), 12)
    # print("方差:", max_objs_var)
    alphas_std = np.average(np.round(np.std(alphas, axis=0), 12))
    betas_std = np.average(np.round(np.std(betas, axis=0), 12))
    gammas_std = np.average(np.round(np.std(gammas, axis=0), 12))
    # print("alpha/beta/gamma方差:", alphas_std,betas_std,gammas_std)

    # 输出结果
    # alpha,beta,gamma系数大小排序（脑区+大小）
    best_alpha = pd.DataFrame(best_alpha.T, index=X_bl.columns[dim], columns=['A'])
    best_beta = pd.DataFrame(best_beta.T, index=X_bl.columns[dim], columns=['B'])
    best_gamma = pd.DataFrame(best_gamma.T, index=X_bl.columns[dim], columns=['C'])

    # print("Optimized parameters:\n","alpha:\n", best_alpha[0:10],"\n beta:\n", best_beta[0:10],"\n gamma:\n", best_gamma[0:10])
    # print("Average iteration time:", np.average(iter_times))
    # print("Converge_prob:", converge_count / total_trials)
    # print("Best Trial:", best_trial, "Iter times:", best_i, "max obj:", best_obj,"\n")

    return best_alpha, best_beta, best_gamma, best_obj, np.average(iter_times), converge_count / total_trials, max_objs_std, alphas_std,betas_std,gammas_std
    