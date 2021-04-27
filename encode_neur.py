import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def phi(x):
    return np.log(1 + x)



# Build feature rows by iterating over users
for k in range(20):
    Q_mat = sparse.load_npz('./data/chunks/q_mat_full_{}_.npz'.format(k)).toarray()
    df = pd.read_csv('./data/chunks/dat_{}.csv'.format(k),sep='\t',dtype={'user_id':np.int16,'item_id':np.int16})
    num_items, num_skills = Q_mat.shape
    features = {}
    #Transform Q mat for fast lookup
    Q_mat_dict = {i: set() for i in range(num_items)}
    for i, j in np.argwhere(Q_mat == 1):
        Q_mat_dict[i].add(j)

    features["s"] = sparse.csr_matrix(np.empty((0, num_skills))) #Skill feature to be stacked
    features['a'] = sparse.csr_matrix(np.empty((0, num_skills + 2))) #Attempt feature to be stacked
    features['w'] = sparse.csr_matrix(np.empty((0, num_skills + 2))) #Win feature to be stacked
    
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct", "skill_id"]].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]
        item_ids = df_user[:, 1].reshape(-1, 1)
        labels = df_user[:, 3].reshape(-1, 1)
        
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
#         print(tmp)
        attempts[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1].astype(int)].astype(np.float)) #last but 1 column stores log of item attempt counts
        attempts[:, -1] = phi(np.arange(num_items_user)) #last column holds log of total question attempt count
        features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

        #Wins encoding
        wins = np.zeros((num_items_user, num_skills + 2))
        ## skill wins encoding
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        corrects = np.hstack((np.array(0), df_user[:, 3].astype(int))).reshape(-1, 1)[:-1]
        wins[:, :num_skills] = phi(np.cumsum(tmp * corrects, 0) * skills) #first num_skills columns store log of skill win counts 
        ## item wins encoding
        onehot = OneHotEncoder(categories=np.arange(df_user[:, 1].max() + 1).reshape(1,-1))
        item_ids_onehot = onehot.fit_transform(item_ids).toarray()
        tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot * labels, 0)))[:-1]
        wins[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1].astype(int)].astype(np.float)) #last but 1 column stores log of item win counts
        wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, 3].astype(int))[:-1]))) #last column holds log of total question win count
        features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])
    X = sparse.hstack([features[x] for x in ['s','a','w'] ]).tocsr()
    print('X file_',k,'done')
    sparse.save_npz('./data/chunks/X-saw_{}.npz'.format(k), X)
