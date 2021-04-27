import argparse
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from math import sqrt
warnings.filterwarnings("ignore", category=UserWarning)

def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)
    nll = log_loss(y, y_pred)
    mse = brier_score_loss(y, y_pred)
    return acc, auc, nll, mse

def get_splits(users,n_splits=5):
    train_idx_lst = []
    test_idx_lst = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(np.zeros(users.shape[0]), users):
        train_idx_lst.append(train_index)
        test_idx_lst.append(test_index)
    return train_idx_lst,test_idx_lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train logistic regression on sparse feature matrix.')
    parser.add_argument('--iter', type=int, default=1500)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--X_file', type=str,default='X-isicsctcwa.npz',
                        help='The name of the X file.')
    parser.add_argument('--write_op', action='store_true',
                        help='If True, write predictions and metrics to file.')
    args = parser.parse_args()
    print('-------------------------------------------------------------')
    print('Dataset',args.dataset,'\n')
    X = csr_matrix(load_npz('./data/{}/X-isicsctcwa.npz'.format(args.dataset)))
    acc_train_lst,auc_train_lst,rmse_train_lst = [],[],[]
    acc_test_lst,auc_test_lst,rmse_test_lst = [],[],[]
    users = X[:,0].toarray().flatten()    
    train_idx_lst,test_idx_lst = get_splits(users,n_splits=5)
    for i in range(5):
        train_idx = train_idx_lst[i]
        test_idx = test_idx_lst[i]
        train = X[train_idx]
        test = X[test_idx]
        X_train,y_train = train[:,5:],train[:,3].toarray().flatten()
        X_test,y_test = test[:,5:],test[:,3].toarray().flatten()
        model = LogisticRegression(solver="lbfgs", max_iter=args.iter)
        model.fit(X_train, y_train)
        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]
        if(args.write_op):
            pd.DataFrame({'Y_True':y_test,'Y_prob':y_pred_test}).to_csv(f'./data/{args.dataset}/Train_preds_Best-Lr_{i}.csv',index=False)
        acc_train, auc_train, nll_train, mse_train = compute_metrics(y_pred_train, y_train)
        acc_test, auc_test, nll_test, mse_test = compute_metrics(y_pred_test, y_test)
        acc_train_lst.append(acc_train)
        acc_test_lst.append(acc_test)
        auc_train_lst.append(auc_train)
        auc_test_lst.append(auc_test)
        rmse_train_lst.append(sqrt(mse_train))
        rmse_test_lst.append(sqrt(mse_test))
        print(f"{args.dataset} split no: {i} "
          f"auc_train = {auc_train}, auc_test = {(auc_test)}, "
          f"mse_train = {mse_train}, mse_test = {sqrt(mse_test)}")
        print()
    print()
    op_df = pd.DataFrame({'Acc Train':acc_train_lst,'Acc Test':acc_test_lst,'Auc Train':auc_train_lst,'Auc Test':auc_test_lst,
                        'RMSE Train':rmse_train_lst,'RMSE Test':rmse_test_lst})
    op_df.to_csv('./data/{}/Prediction_metrics_lr.csv'.format(args.dataset))