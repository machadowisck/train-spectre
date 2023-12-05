import os
import numpy as np
import os.path as osp

if __name__=="__main__":
    assert_root='/media/datou/disk/Data/DATA_PRODUCT/AVATAR/'
    save_dir='train_spectre.txt'
    meta_id_list = os.listdir(assert_root)
    meta_id_list = sorted(meta_id_list)
    with open(save_dir,'w') as f:
        for id in meta_id_list:
            if id =='aomei_xiaodongxue_sit':
                meta_dir=osp.join(assert_root,id,'DATAPROCESS')
                img_dir=osp.join(meta_dir,'crop_head_imgs')
                head_info_dir=osp.join(meta_dir,'crop_head_info.npy')
                f.write(meta_dir+','+str(len(os.listdir(img_dir)))+'\n')


