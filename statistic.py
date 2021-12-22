import pandas as pd
import numpy as np
from pandas.core.indexes.base import Index

def statistic_csv(csv_list,save_path):
    data = []
    for csv in csv_list:
        df = pd.read_csv(csv)
        data_item =  np.asarray(df,dtype=np.float32)[:,2:]
        data_mean = np.mean(data_item,axis=0)
        data.append(np.round(data_mean,decimals=4))
    data = np.stack(data,axis=0)
    data = np.hstack((data,np.mean(data,axis=1)[:,None]))
    mean = np.round(np.mean(data,axis=0)[None,:],decimals=4)
    std = np.round(np.std(data,axis=0)[None,:],decimals=4)
    data = np.vstack((data,mean,std))
    col = ["Skull_x", "Bone_x", "Rib_x", "Shoulder_extremities_x", "Stemum_x", "Pelvis_x", "LN_x", "Total"]
    df = pd.DataFrame(data=data,columns=col)
    df.to_csv(save_path,index=False)

if __name__ == '__main__':

    version_list = ['v1.0','v1.1','v1.3','v2.1','v2.3','v4.1','v4.10-pretrain']

    for version in version_list[1:]:
        dice_csv_list = [f'./result/raw_data/{version}_fold{str(i)}_dice.csv' for i in range(1,6)]
        save_path = f'./result/analysis/{version}_dice.csv'
        dice_list = statistic_csv(dice_csv_list,save_path)

        hd_csv_list = [f'./result/raw_data/{version}_fold{str(i)}_hd.csv' for i in range(1,6)]
        save_path = f'./result/analysis/{version}_hd.csv'
        hd_list = statistic_csv(hd_csv_list,save_path)
        break