# Borrowed with minor changes from theophilee's repo https://github.com/theophilee/learner-performance-prediction

import numpy as np
import pandas as pd
import warnings
import argparse
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def phi(x):
    return np.log(1 + x)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify dataset to encode')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    Q_mat = sparse.load_npz('./data/{}/q_mat.npz'.format(args.dataset)).toarray()
    df = pd.read_csv('./data/{}/preprocessed_data.csv'.format(args.dataset),sep='\t',dtype={'user_id':np.int32,'item_id':np.int32})
    print('done reading data')
    num_items, num_skills = Q_mat.shape
    features = {}

    features["s"] = sparse.csr_matrix(np.empty((0, num_skills))) #Skill feature to be stacked
    features['a'] = sparse.csr_matrix(np.empty((0, num_skills + 2))) #Attempt feature to be stacked
    features['w'] = sparse.csr_matrix(np.empty((0, num_skills + 2))) #Win feature to be stacked
    # print(df.columns)
    # Build feature rows by iterating over users
    for user_id in df["user_id"].unique():
        if args.dataset != 'synthetic':
            df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct", "skill_id"]].copy()
        else:
            df_user = df[df["user_id"] == user_id][["user_id", "item_id",'score']].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]
        item_ids = df_user[:, 1].reshape(-1, 1)
        if args.dataset != 'synthetic':
            labels = df_user[:, 3].reshape(-1, 1)
        else:
            labels = df_user[:, -1].reshape(-1, 1)
        
        #Skill encoding
        skills = Q_mat[df_user[:, 1].astype(int)].copy()
        features['s'] = sparse.vstack([features["s"], sparse.csr_matrix(skills)])

        #Attempts encoding
        attempts = np.zeros((num_items_user, num_skills + 2))
        ## skill attempt encodig
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        attempts[:, :num_skills] = phi(np.cumsum(tmp, 0) * skills) #first num_skills columns store log of skill attempt counts 
        ## item attempt encoding
        onehot = OneHotEncoder(categories=np.arange(df_user[:, 1].max() + 1).reshape(1,-1))
        item_ids_onehot = onehot.fit_transform(item_ids).toarray()
        tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot, 0)))[:-1]
        attempts[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1].astype(int)].astype(np.float)) #last but 1 column stores log of item attempt counts
        ## total items encoding
        attempts[:, -1] = phi(np.arange(num_items_user)) #last column holds log of total question attempt count
        features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

        #Wins encoding
        wins = np.zeros((num_items_user, num_skills + 2))
        ## skill wins encoding
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        if args.dataset != 'synthetic':
            corrects = np.hstack((np.array(0), df_user[:, 3].astype(int))).reshape(-1, 1)[:-1]
        else:
            corrects = np.hstack((np.array(0), df_user[:, -1])).reshape(-1, 1)[:-1]
        wins[:, :num_skills] = phi(np.cumsum(tmp * corrects, 0) * skills) #first num_skills columns store log of skill win counts 
        ## item wins encoding
        onehot = OneHotEncoder(categories=np.arange(df_user[:, 1].max() + 1).reshape(1,-1))
        item_ids_onehot = onehot.fit_transform(item_ids).toarray()
        tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot * labels, 0)))[:-1]
        wins[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1].astype(int)].astype(np.float)) #last but 1 column stores log of item win counts
        ## total items encoding
        if args.dataset != 'synthetic':
            wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, 3].astype(int))[:-1]))) #last column holds log of total question win count
        else:
            wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, -1])[:-1])))
        features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])

    if args.dataset == 'synthetic':
        onehot = OneHotEncoder()
        features['i'] = onehot.fit_transform(df.values[:, 1].reshape(-1, 1))
    
    # Create and write the npz array to disk
    X = sparse.hstack([features[x] for x in ['s','a','w','i'] if x!='i' or args.dataset == 'synthetic' ]).tocsr()
    sparse.save_npz('./data/{}/X-file.npz'.format(args.dataset), X)

    index_df = pd.DataFrame({'Feature':['skills','attempts','wins'],'Start_idx':[0,num_skills,2*num_skills+2],'End_idx':
                            [num_skills,2*num_skills+2,3*num_skills+4]})
    index_df.to_csv('./data/{}/index.csv'.format(args.dataset),index=False)

    print('done writing encoded files')