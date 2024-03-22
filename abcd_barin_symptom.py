import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from tensor_power_method import power_method
from train_test import sparsity_selection
from train_test import kfold_validation

warnings.filterwarnings("ignore", category=DeprecationWarning)
res = list(range(0,1000000,1))
np.seterr(all='ignore')

def read_data(dim):
    '''
    读取abcd_v5数据
    '''
    df = pd.read_csv("data/wide_abcd_v5_rm_NA_coding_sel_dummy_scale_resid.txt", sep="\t")
    df = df.loc[:, ['cbcl_scr_dsm5_depress_r.bl','cbcl_scr_dsm5_depress_r.2y','cbcl_scr_dsm5_depress_r.4y'] + 
                    list(df.loc[:, 'smri_vol_cdk_cdmdfrlh.bl':'smri_vol_scs_wholeb.bl'].columns) + 
                    list(df.loc[:, 'smri_vol_cdk_cdmdfrlh.2y':'smri_vol_scs_wholeb.2y'].columns) + 
                    list(df.loc[:, 'smri_vol_cdk_cdmdfrlh.4y':'smri_vol_scs_wholeb.4y'].columns)]
    df = df.dropna()

    Y_bl = df.loc[:,'cbcl_scr_dsm5_depress_r.bl']
    Y_fu2 = df.loc[:,'cbcl_scr_dsm5_depress_r.2y']
    Y_fu4 = df.loc[:,'cbcl_scr_dsm5_depress_r.4y']
    Y_fu2.rename(index={'cbcl_scr_dsm5_depress_r.2y': 'cbcl_scr_dsm5_depress_r.bl'}, inplace=True)
    Y_fu4.rename(index={'cbcl_scr_dsm5_depress_r.4y': 'cbcl_scr_dsm5_depress_r.bl'}, inplace=True)

    X_bl = df.loc[:,'smri_vol_cdk_cdmdfrlh.bl':"smri_vol_scs_wholeb.bl"]
    X_fu2 = df.loc[:,'smri_vol_cdk_cdmdfrlh.2y':"smri_vol_scs_wholeb.2y"]
    X_fu4 = df.loc[:,'smri_vol_cdk_cdmdfrlh.4y':"smri_vol_scs_wholeb.4y"]
    X_fu2.columns = X_bl.columns
    X_fu4.columns = X_bl.columns

    X_t_1 = pd.concat([X_bl,X_fu2]).to_numpy()[:,dim]
    X_t = pd.concat([X_fu2,X_fu4]).to_numpy()[:,dim]
    Y_t_1 = pd.concat([Y_bl,Y_fu2]).to_numpy()
    Y_t = pd.concat([Y_fu2,Y_fu4]).to_numpy()

    M = X_t_1.shape[1]  # 参数向量的维度为M

    return X_t_1, X_t, Y_t_1, Y_t, M, dim, X_bl, X_fu2, X_fu4, Y_bl, Y_fu2, Y_fu4


def main():
    shift = 0
    M = 68
    truncation = np.round(M*np.arange(1/np.sqrt(68), 1, 0.1)).astype(int)
    ops = ["+","+","+"]
    for dim in  [np.arange(0,68),np.arange(68,68*2),np.arange(68*2,68*3)]:
        print("以下是",str(dim[0])+"_"+str(dim.shape[0]),"的结果")
        X_t_1, X_t, Y_t_1, Y_t, M, dim, X_bl, X_fu2, X_fu4, Y_bl, Y_fu2, Y_fu4 = read_data(dim)
        X_t_1_train,X_t_1_test, X_t_train,X_t_test, Y_t_1_train,Y_t_1_test, Y_t_train, Y_t_test = train_test_split(X_t_1, X_t, Y_t_1, Y_t, test_size=0.2)
        
        # # 1.选择稀疏性参数
        best_spar = sparsity_selection(X_t_1_train, X_t_train, Y_t_1_train, Y_t_train, np.random.randn(M, 1),np.random.randn(M, 1),np.random.randn(M, 1),M, dim, X_bl, ops,truncation,shift,num_power_trials=100,num_perm=1000)
        print("best_spar:",best_spar)
        # # 2.交叉验证以及测试典型变量的显著性（确定最佳k参数后！！）
        p_total,p_alpha,p_beta,p_gamma1,p_gamma2 = kfold_validation(X_t_1_train, X_t_train, Y_t_1_train, Y_t_train, M, dim, X_bl, best_spar, ops,shift, num_power_trials=100,num_perm = 1000)
        print('cross_validation_totest_composite\n'+'total_sig_p:'+str(p_total)+'\n alpha_sig_p:'+str(p_alpha) + '\n beta_sig_p:'+str(p_beta) + '\n gamma1_sig_p:'+str(p_gamma1) + '\n gamma2_sig_p:'+str(p_gamma2))
        best_alpha, best_beta, best_gamma, best_obj, _, _,_,_,_,_ = power_method(
            best_spar, ops, M, X_t, X_t_1, Y_t, Y_t_1, X_bl, dim, shift, 
            max_iter=1000, convergence_threshold=1e-6, total_trials=100
        )
        # # 3.对test数据集使用αβγ参数，计算相关性
        alpha_test_corr, alpha_test_p = stats.pearsonr((X_t_1_test@(best_alpha.to_numpy())).reshape(-1), Y_t_test.reshape(-1))
        beta_test_corr, beta_test_p = stats.pearsonr((X_t_test@(best_beta.to_numpy())).reshape(-1), Y_t_1_test.reshape(-1))
        gamma_test_corr1, gamma_test_p1 = stats.pearsonr((X_t_1_test@(best_gamma.to_numpy())).reshape(-1), Y_t_test.reshape(-1))
        gamma_test_corr2, gamma_test_p2 = stats.pearsonr((X_t_test@(best_gamma.to_numpy())).reshape(-1), Y_t_1_test.reshape(-1))
        print('test result\n'+'alpha_sig_p:'+str(alpha_test_p) + '\n beta_sig_p:'+str(beta_test_p) + '\n gamma1_sig_p:'+str(gamma_test_p1) + '\n gamma2_sig_p:'+str(gamma_test_p2))
 

if __name__== "__main__" :
    main()



