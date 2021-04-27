import torch
import pyro
import argparse
import numpy as np
import pandas as pd
import warnings
from svi import SVIIRT,SVIIH
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import load_npz, csr_matrix,save_npz,hstack
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss,confusion_matrix
from sklearn.linear_model import LogisticRegression
from math import sqrt
warnings.filterwarnings("ignore", category=UserWarning)

def get_splits(users,n_splits):
    train_idx_lst = []
    test_idx_lst = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(np.zeros(users.shape[0]), users):
        train_idx_lst.append(train_index)
        test_idx_lst.append(test_index)
    return train_idx_lst,test_idx_lst

def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)    
    mse = brier_score_loss(y, y_pred)
    return acc, auc, mse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify dataset to encode')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--model',type=str,default='SVIRT')
    parser.add_argument('-w',action='store_true')
    args = parser.parse_args()
    print('enter')
    df = pd.read_csv('./data/{}/preprocessed_data.csv'.format(args.dataset),sep='\t',
        dtype={'user_id':np.int32,'item_id':np.int32},usecols=['user_id','item_id','correct'])
    X = csr_matrix(load_npz('./data/{}/X-file.npz'.format(args.dataset)))
    index_df = pd.read_csv('./data/{}/index.csv'.format(args.dataset),dtype={'Start_idx':np.int32,'End_idx':np.int32})
    num_items = df['item_id'].unique().shape[0]
    num_models = df['user_id'].unique().shape[0]
    
    acc_train_lst,auc_train_lst,mse_train_lst = [],[],[]
    acc_test_lst,auc_test_lst,mse_test_lst = [],[],[]
    print('Dataset: ',args.dataset)
    print('Model: ',args.model)
    for i in range(args.n_splits):
        df_train_idx = pd.read_csv('./data/{}/Train_idx{}.csv'.format(args.dataset,i),usecols=['Train_idx'],dtype={'Train_idx':np.int32})
        df_test_idx = pd.read_csv('./data/{}/Test_idx{}.csv'.format(args.dataset,i),usecols=['Test_idx'],dtype={'Test_idx':np.int32})
        train_idx_lst = df_train_idx['Train_idx'].to_list()
        test_idx_lst = df_test_idx['Test_idx'].to_list()
        df_train = df.iloc[train_idx_lst]
        models = df_train['user_id'].values
        items = df_train['item_id'].values
        responses = df_train['correct'].values
        end_idx = index_df[index_df['Feature']=='skills']['End_idx'].values[0]
        if(args.model=='SVIRT' or args.model=='SVIRTH-SC' or args.model=='SVIRTH'):
            svi = SVIIRT(torch.device('cpu'),num_items,num_models,False)
            #Convert variables to tensors required by SVI module 
            models = torch.tensor(models,dtype=torch.long)
            items = torch.tensor(items,dtype=torch.long)
            responses = torch.tensor(responses,dtype=torch.float32)
            svi.fit(models,items,responses,args.n_iter)
            theta_loc_arr = pyro.get_param_store().get_param('loc_ability').data.numpy()
            diff_loc_arr = pyro.get_param_store().get_param('loc_diff').data.numpy()
            models = models.detach().numpy()
            items = items.detach().numpy()
            responses = df_train['correct'].values
            exp = (theta_loc_arr[models] - diff_loc_arr[items])
            e_max = np.max(exp)
            if(args.model=='SVIRT'):
                probs = expit(exp)
                preds = np.round(probs)
            else:
                exp = csr_matrix(exp.reshape(-1,1))
                X_train = X[train_idx_lst]
                # X_train = X
                num_skills = end_idx
                skills = X_train[:,:num_skills]
                attempts = X_train[:,num_skills:2*num_skills+2]
                wins = X_train[:,2*num_skills+2:]
                skill_attempts,ic_attempts,tc_attempts = attempts[:,:-2],attempts[:,-2],attempts[:,-1] 
                skill_wins,ic_wins,tc_wins = wins[:,:-2],wins[:,-2],wins[:,-1]
                exp = exp.reshape(-1,1)
                if args.model == 'SVIRTH':
                    features = hstack((exp,X[train_idx_lst]))
                else:
                    features = hstack((exp,skills,skill_attempts,skill_wins))
                lr = LogisticRegression(solver="lbfgs", max_iter=args.n_iter+1000)
                lr.fit(features,responses)
                probs = lr.predict_proba(features)[:,1]
                preds = np.round(probs)
            acc, auc, mse = compute_metrics(probs,responses)
            acc_train_lst.append(acc)
            auc_train_lst.append(auc)
            mse_train_lst.append(sqrt(mse))
            if args.w:
                pd.DataFrame({'Y_True':responses,'Y_prob':probs}).to_csv(f'./data/{args.dataset}/Train_preds_{args.model}_{i}.csv',index=False)
            # Get predictions for test set
            df_test = df.iloc[test_idx_lst]
            models = df_test['user_id'].values
            items = df_test['item_id'].values
            responses = df_test['correct'].values
            exp = (theta_loc_arr[models] - diff_loc_arr[items])
            if(args.model=='SVIRT'):
                probs = expit(exp)
                preds = np.round(probs)
            else:
                exp = csr_matrix(exp.reshape(-1,1)) 
                X_test = X[test_idx_lst]
                # X_test = X
                if args.model == 'SVIRTH':
                    features = hstack((exp,X_test))
                else:
                    features = hstack((exp,X_test[:,:num_skills],X_test[:,num_skills:2*num_skills],X_test[:,2*num_skills+2:3*num_skills+2]))
                probs = lr.predict_proba(features)[:,1]
                preds = np.round(probs)
            acc, auc, mse = compute_metrics(probs,responses)
            acc_test_lst.append(acc)
            auc_test_lst.append(auc)
            mse_test_lst.append(sqrt(mse))
            if args.w:
                pd.DataFrame({'Y_True':responses,'Y_prob':probs}).to_csv(f'./data/{args.dataset}/Test_preds_{args.model}_{i}.csv',index=False)
            # print(theta_loc_arr.shape,diff_loc_arr.shape,e_max.shape)
            # print('T max:',max(theta_loc_arr),'I max:',max(diff_loc_arr),'exp max:',(e_max))
        else:
            svih = SVIIH(torch.device('cpu'),items.shape[0],False)
            #Convert variables to tensors required by SVI module 
            items = torch.tensor(items,dtype=torch.long)
            responses = torch.tensor(responses,dtype=torch.float32)
            svih.fit(items,responses,args.n_iter)
            diff_loc_arr = pyro.get_param_store().get_param('loc_diff').data.numpy()
            models = df_train['user_id'].values
            items = items.detach().numpy()
            responses = df_train['correct'].values
            exp = (diff_loc_arr[items]).reshape(-1,1)
            exp = csr_matrix(exp)
            features = hstack((exp,X[train_idx_lst]))
            lr = LogisticRegression(solver="lbfgs", max_iter=args.n_iter+1000)
            lr.fit(features,responses)
            probs = lr.predict_proba(features)[:,1]
            preds = np.round(probs)
            acc, auc, mse = compute_metrics(probs,responses)
            acc_train_lst.append(acc)
            auc_train_lst.append(auc)
            mse_train_lst.append(sqrt(mse))
            if args.w:
                pd.DataFrame({'Y_True':responses,'Y_prob':probs}).to_csv(f'./data/{args.dataset}/Train_preds_{args.model}_{i}.csv',index=False)
            # Get predictions for test set 
            df_test = df.iloc[test_idx_lst]
            models = df_test['user_id'].values
            items = df_test['item_id'].values
            responses = df_test['correct'].values
            exp = diff_loc_arr[items].reshape(-1,1)
            exp = csr_matrix(exp)
            X_test = X[test_idx_lst]
            features = hstack((exp,X_test))
            probs = lr.predict_proba(features)[:,1]
            preds = np.round(probs)
            acc, auc, mse = compute_metrics(probs,responses)
            acc_test_lst.append(acc)
            auc_test_lst.append(auc)
            mse_test_lst.append(sqrt(mse))
            if args.w:
                pd.DataFrame({'Y_True':responses,'Y_prob':probs}).to_csv(f'./data/{args.dataset}/Test_preds_{args.model}_{i}.csv',index=False)
            
        print('-----------------------------------------')
        print('fold',i)        
        print('Training set')
        print('Acc:',acc_train_lst[i],'Auc:',auc_train_lst[i],'RMSE:',(mse_train_lst[i]))
        print()
        # print('Non 0 sores true',np.count_nonzero(responses)/responses.shape[0],'Non 0 sores preds',np.count_nonzero(preds)/preds.shape[0])
        # print()
        print('Test set')
        print()
        print('Non 0 sores true',np.count_nonzero(responses)/responses.shape[0],'Non 0 sores preds',np.count_nonzero(preds)/preds.shape[0])
        print('Acc:',acc_test_lst[i],'Auc:',auc_test_lst[i],'RMSE:',(mse_test_lst[i]))
        print()
        # break
        
        

    if(args.w):    
        op_df = pd.DataFrame({'Fold':np.arange(args.n_splits)+1,'Acc Train':acc_train_lst,'Acc test':acc_test_lst,
                            'Auc Train':auc_train_lst,'Auc Test':auc_test_lst,'RMSE Train':mse_train_lst,'RMSE test':mse_test_lst})
        op_df.to_csv('./data/{}/Prediction_metrics_{}.csv'.format(args.dataset,args.model),index=False)





