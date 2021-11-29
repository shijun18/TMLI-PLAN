import os 


def dfs_rename_dir(path):
    for item in os.scandir(path):
        if item.is_dir():
            dfs_rename_dir(item.path)       
        else:
            base_name = os.path.basename(path)
            print(base_name)
            dir_name = os.path.dirname(path)
            new_name = base_name.split('_')[1].replace("-",'')
            if 'RTst' in path:
                new_name = new_name + '_RT'
            elif 'CT' in path:
                new_name = new_name + '_CT'
            else:
                raise ValueError('Error!!')
            print(new_name)
            os.rename(path,os.path.join(dir_name,new_name))

            break


if __name__ == '__main__':
    # path = '/staff/shijun/dataset/TMLI/raw_data'
    path = '/staff/shijun/dataset/TMLI/zip_data/tmp_zip'
    dfs_rename_dir(path)