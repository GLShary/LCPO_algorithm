import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from semopy import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scipy import stats
from tensor_power_method import power_method
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
res = list(range(0,1000000,1))
np.seterr(all='ignore')

def cv_permute_test(k, ops, M, X_bl, dim, shift,X_t_1, X_t, Y_t_1, Y_t, alpha_corr, beta_corr, gamma_corr1, gamma_corr2, num_perm):
    '''
    交叉验证检验组合变量的显著性
    '''

    count_total = count_alpha = count_beta = count_gamma1 = count_gamma2 = 0
    for _ in range(num_perm):
        shuffled_Y_t = np.random.permutation(Y_t)
        shuffled_Y_t_1 = np.random.permutation(Y_t_1)

        alpha, beta, gamma, best_obj1, _, _,_,_,_,_ = power_method(
            k, ops, M, X_t, X_t_1, shuffled_Y_t, shuffled_Y_t_1, X_bl, dim, shift, 
            max_iter=1000, convergence_threshold=1e-6, total_trials=10)

        shuffled_alpha_corr, alpha_p = stats.pearsonr((X_t_1@(alpha.to_numpy())).reshape(-1), Y_t.reshape(-1))
        shuffled_beta_corr, beta_p = stats.pearsonr((X_t@(beta.to_numpy())).reshape(-1), Y_t_1.reshape(-1))
        shuffled_gamma_corr1, gamma_p1 = stats.pearsonr((X_t_1@(gamma.to_numpy())).reshape(-1), Y_t.reshape(-1))
        shuffled_gamma_corr2, gamma_p2 = stats.pearsonr((X_t@(gamma.to_numpy())).reshape(-1), Y_t_1.reshape(-1))

        if shuffled_alpha_corr >= alpha_corr:
            count_alpha += 1
        if shuffled_beta_corr >= beta_corr:
            count_beta += 1
        if shuffled_gamma_corr1 >= gamma_corr1:
            count_gamma1 += 1
        if shuffled_gamma_corr2 >= gamma_corr2:
            count_gamma2 += 1
        if shuffled_alpha_corr+shuffled_beta_corr+shuffled_gamma_corr1+shuffled_gamma_corr2 >= alpha_corr+beta_corr+gamma_corr1+gamma_corr2:
            count_total += 1
    p_total, p_alpha, p_beta, p_gamma1, p_gamma2 = count_total/num_perm, count_alpha/num_perm, count_beta/num_perm, count_gamma1/num_perm, count_gamma2/num_perm

    return p_total, p_alpha, p_beta, p_gamma1, p_gamma2


def kfold_validation(X_t_1, X_t, Y_t_1, Y_t, M, dim, X_bl, k, ops, shift, num_power_trials, num_perm):
    '''
    测试组合变量的显著性（交叉验证方法）
    '''

    kf = KFold(n_splits=5, shuffle=True)
    best_objs = []  # 用于存放每一折的最佳目标值
    alpha_corrs = []
    beta_corrs = []
    gamma_corr1s = []
    gamma_corr2s = []

    count_alpha = np.zeros((68,1))
    count_beta = np.zeros((68,1))
    count_gamma = np.zeros((68,1))

    print("交叉验证：")
    for i in range(0,1):
        for train_index, valid_index in kf.split(X_t_1):
            # 分割数据
            X_t_1_train, X_t_1_valid = X_t_1[train_index], X_t_1[valid_index]
            X_t_train, X_t_valid = X_t[train_index], X_t[valid_index]
            Y_t_1_train, Y_t_1_valid = Y_t_1[train_index], Y_t_1[valid_index]
            Y_t_train, Y_t_valid = Y_t[train_index], Y_t[valid_index]

            # 我们假设power_method函数接收训练数据并返回最优参数和目标值
            best_alpha, best_beta, best_gamma, best_obj, _, _,_,_,_,_ = power_method(
                k, ops, M, X_t_train, X_t_1_train, Y_t_train, Y_t_1_train, X_bl, dim, shift, 
                max_iter=1000, convergence_threshold=1e-6, total_trials=num_power_trials
            )
            count_alpha[(best_alpha.to_numpy())!=0]+=1
            count_beta[(best_beta.to_numpy())!=0]+=1
            count_gamma[(best_gamma.to_numpy())!=0]+=1

            # _, _, _, best_obj = compute_G_matrices(X_t_test, X_t_1_test, Y_t_test, Y_t_1_test, best_alpha.to_numpy(), best_beta.to_numpy(), best_gamma.to_numpy(),ops)
            alpha_corr, alpha_p = stats.pearsonr((X_t_1_valid@(best_alpha.to_numpy())).reshape(-1), Y_t_valid.reshape(-1))
            beta_corr, beta_p = stats.pearsonr((X_t_valid@(best_beta.to_numpy())).reshape(-1), Y_t_1_valid.reshape(-1))
            gamma_corr1, gamma_p1 = stats.pearsonr((X_t_1_valid@(best_gamma.to_numpy())).reshape(-1), Y_t_valid.reshape(-1))
            gamma_corr2, gamma_p2 = stats.pearsonr((X_t_valid@(best_gamma.to_numpy())).reshape(-1), Y_t_1_valid.reshape(-1))

            # 保存每一折得到的最佳目标值
            best_objs.append(best_obj)
            alpha_corrs.append(alpha_corr)
            beta_corrs.append(beta_corr)
            gamma_corr1s.append(gamma_corr1)
            gamma_corr2s.append(gamma_corr2)

    # 计算100*5次的平均best_obj
    # print(best_objs)
    average_best_obj = np.mean(best_objs)
    average_alpha_corrs = np.mean(alpha_corrs)
    average_beta_corrs = np.mean(beta_corrs)
    average_gamma_corr1s = np.mean(gamma_corr1s)
    average_gamma_corr2s = np.mean(gamma_corr2s)

    print("Average best_obj:", average_best_obj,"\n average alpha corr square:",average_alpha_corrs,
            "\n average beta corr square:",average_beta_corrs,"\n average gamma1 corr square:",average_gamma_corr1s,
            "\n average gamma2 corr square:",average_gamma_corr2s)
    # permutation 检验三套系数的显著性
    p_total, p_alpha, p_beta, p_gamma1, p_gamma2 = cv_permute_test(k, ops, M, X_bl, dim, shift, X_t_1, X_t, Y_t_1.reshape(Y_t_1.shape[0],1), Y_t.reshape(Y_t.shape[0],1), 
                                                        average_alpha_corrs, average_beta_corrs, average_gamma_corr1s, average_gamma_corr2s,
                                                        num_perm = num_perm)
    print('validation\n'+'total_sig_p:'+str(p_total)+'\n alpha_sig_p:'+str(p_alpha) + '\n beta_sig_p:'+str(p_beta) + '\n gamma1_sig_p:'+str(p_gamma1) + '\n gamma2_sig_p:'+str(p_gamma2))
    
    return  p_total, p_alpha, p_beta, p_gamma1, p_gamma2


def permute_test(X_t_1, X_t, Y_t_1, Y_t, alpha, beta, gamma, num_perm):
    '''
    permutation检验，选择稀疏性参数
    '''
    count_total = count_alpha = count_beta = count_gamma1 = count_gamma2 = 0
   
    alpha_corr, alpha_p = stats.pearsonr((X_t_1@alpha).reshape(-1), Y_t.reshape(-1))
    beta_corr, beta_p = stats.pearsonr((X_t@beta).reshape(-1), Y_t_1.reshape(-1))
    gamma_corr1, gamma_p1 = stats.pearsonr((X_t_1@gamma).reshape(-1), Y_t.reshape(-1))
    gamma_corr2, gamma_p2 = stats.pearsonr((X_t@gamma).reshape(-1), Y_t_1.reshape(-1))

    for _ in range(num_perm):
    # 随机打乱Y的顺序
        shuffled_Y_t = np.random.permutation(Y_t)
        shuffled_Y_t_1 = np.random.permutation(Y_t_1)

        # 重新计算打乱后的Y和Xα的相关系数
        shuffled_alpha_corr, shuffled_alpha_p = stats.pearsonr((X_t_1@alpha).reshape(-1), shuffled_Y_t.reshape(-1))
        shuffled_beta_corr, shuffled_beta_p = stats.pearsonr((X_t@beta).reshape(-1), shuffled_Y_t_1.reshape(-1))
        shuffled_gamma_corr1, shuffled_gamma_p1 = stats.pearsonr((X_t_1@gamma).reshape(-1), shuffled_Y_t.reshape(-1))
        shuffled_gamma_corr2, shuffled_gamma_p2 = stats.pearsonr((X_t@gamma).reshape(-1), shuffled_Y_t_1.reshape(-1))

        if shuffled_alpha_corr >= alpha_corr:
            count_alpha += 1
        if shuffled_beta_corr >= beta_corr:
            count_beta += 1
        if shuffled_gamma_corr1 >= gamma_corr1:
            count_gamma1 += 1
        if shuffled_gamma_corr2 >= gamma_corr2:
            count_gamma2 += 1
        if shuffled_alpha_corr+shuffled_beta_corr+shuffled_gamma_corr1+shuffled_gamma_corr2 >= alpha_corr+beta_corr+gamma_corr1+gamma_corr2:
            count_total += 1
    p_total, p_alpha, p_beta, p_gamma1, p_gamma2 = count_total/num_perm, count_alpha/num_perm, count_beta/num_perm, count_gamma1/num_perm, count_gamma2/num_perm

    return p_total, p_alpha, p_beta, p_gamma1, p_gamma2


def fig_select_sparse(objs,alpha_eva,beta_eva,gamma_eva,sparsity,ops):
    '''
    画图：针对每种组合，
    最大目标函数值vs不同的稀疏性比例;
    α^Tα等 vs 不同的稀疏性比例。
    '''
    plt.rcParams['font.family'] = 'Arial'
    # plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(10, 8))  # 增大整个图表的尺寸
    # 绘制max obj随迭代次数的变化图
    plt.subplot(2, 1, 1)  # 创建第一个子图，占据上半部分
    plt.plot(sparsity, objs, marker='o', linestyle='-', color='#264654', label='Max Objective')  # 绘制折线图
    plt.xticks(sparsity,fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Sparsity',fontsize=20)  # x轴标签
    plt.ylabel('Max Objective',fontsize=20)  # y轴标签
    plt.title('Max Objective vs. Sparsity ',fontsize=20)  # 图像标题
    plt.grid(True)  # 显示网格
    plt.legend(fontsize=15)  # 显示图例

    plt.subplot(2, 1, 2)  # 创建第一个子图，占据上半部分
    plt.plot(sparsity, alpha_eva, marker='s', linestyle='-', color=('#e76f51'), label='alpha_eva')  # 绘制折线图
    plt.plot(sparsity, beta_eva, marker='x', linestyle='-', color=('#299e8b'), label='beta_eva')  # 绘制折线图
    plt.plot(sparsity, gamma_eva, marker='^', linestyle='-', color=('#f2a461'), label='gamma_eva')  # 绘制折线图
    plt.xticks(sparsity,fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Sparsity',fontsize=20)  # x轴标签
    plt.ylabel('Parameter Evaluation',fontsize=20)  # y轴标签
    plt.title('Parameter Evaluation vs. Sparsity ',fontsize=20)  # 图像标题
    plt.grid(True)  # 显示网格
    plt.legend(fontsize=15)  # 显示图例

    plt.tight_layout(pad=3)  # 调整子图布局，增加pad参数确保子图之间有更多空间
    # plt.yscale('log')  # 如果错误值差异很大，考虑使用对数刻度

    plt.savefig('plot/simu_sparsity_selection_0306/simu_'+ str(ops) + '.png')  # 保存图片
    plt.show()  # 显示图形
    plt.close()  # 关闭图形


def sparsity_selection(X_t_1, X_t, Y_t_1, Y_t, alpha,beta,gamma,M, dim, X_bl, ops,sparse,shift,num_power_trials,num_perm):   
    '''
    1.检验三套系数的显著性，以确定最佳稀疏参数
    2.abg_evaluation v.s. sparsity
    3.作图
    （若想运行simu数据并作图，可取消本函数注释）
    '''
    objs_of_dif_sparsity = []
    alpha_eva = []
    beta_eva = []
    gamma_eva = []

    p_ori = 100000
    for k in sparse:
        print('\n 以下是'+str(dim[0])+"_"+str(dim.shape[0])+"_sparsity"+str(k)+'的结果：')
        power_alpha1, power_beta1, power_gamma1, best_obj1, avg_iter_time, converge_prob,max_objs_std,alphas_std,betas_std,gammas_std = power_method(
                k, ops, M, X_t_1, X_t_1, Y_t, Y_t_1, X_bl, dim, shift, 
                max_iter=1000, convergence_threshold=1e-6, total_trials=num_power_trials
            )       
        # 2.permutation 检验三套系数的显著性
        # output_content += f"dim:{dim},ops:{ops},k:{k}\n"
        p_sum, p_alpha, p_beta, p_gamma1, p_gamma2 = permute_test(X_t_1, X_t, Y_t_1.reshape(Y_t_1.shape[0],1), Y_t.reshape(Y_t.shape[0],1), power_alpha1.to_numpy(), power_beta1.to_numpy(), power_gamma1.to_numpy(), num_perm = num_perm)
        print('系数显著性\n'+'total_sig_p:'+str(p_sum) + '\n alpha_sig_p:'+str(p_alpha) + '\n beta_sig_p:'+str(p_beta) + '\n gamma_sig_p:'+str(p_gamma1) + '\n gamma2_sig_p:'+str(p_gamma2) + '\n')
        # output_content += f"拼接\n alpha_sig_p:{str(p_alpha)},\n beta_sig_p:{str(p_beta)},\n gamma_sig_p:,{str(p_gamma)}\n"
        if(p_sum < p_ori):
            p_ori = p_sum
            best_k = k

        # 3.obj和参数结果的稳定性（标准差）
        print("目标函数标准差:",max_objs_std)
        print("alpha/beta/gamma标准差:", alphas_std,betas_std,gammas_std)
        print(num_power_trials,"次实验的收敛比率：", converge_prob,"\n")
        
        # # 记录每次实验的obj，α^Tα, β^Tβ, γ^Tγ
        # objs_of_dif_sparsity.append(float(best_obj1))
        # alpha_eva.append((np.abs(alpha.T@(power_alpha1))).iloc[0,0])
        # beta_eva.append((np.abs(beta.T@(power_beta1))).iloc[0,0])
        # gamma_eva.append((np.abs(gamma.T@(power_gamma1))).iloc[0,0])

    # print("alpha\n",alpha_eva,"\n beta\n",beta_eva,"\n gamma\n",gamma_eva)
    # fig_select_sparse(objs_of_dif_sparsity,alpha_eva,beta_eva,gamma_eva,sparse,ops)

    return best_k
