import os
import numpy as np
import os.path as osp
if __name__=="__main__":
    # assert_root='/media/datou/disk/Data/DATA_PRODUCT/AVATAR/yuanru/'
    # #獲取根目錄下所有的.png的路徑，其中存在多個子級目錄,將其中一個子目錄img下的所有png路徑寫入txt文件中
    # img_list=[]
    # for root,dirs,files in os.walk(assert_root):
    #     print(root)
    #     if root.split('/')[-1] != 'crop_head_imgs':
    #         continue
    #     files=sorted(files)
    #     for file in files:
    #         if file.endswith('.png'):
    #             # print(os.path.join(root,file))
    #             img_list.append(os.path.join(root,file))
    #
    # np.save('experiments/img_list.npy',img_list)
    # with open('experiments/img_list.txt','w') as f:
    #     for img in img_list:
    #         f.write(img)
    #         f.write('\n')
    num_train_img=3600
    name=os.getenv('meta_id')
    assert_root = f'/data1/lihaojie/data/DATA_PRODUCT/AVATAR/{name}/DATAPROCESS/crop_head_imgs/'
    imgs_list=os.listdir(assert_root)
    imgs_list=sorted(imgs_list)
    img_num=min(len(imgs_list),num_train_img)
    with open(f'data/{name}.txt','w+') as f:
        for i in range(img_num):
            f.write(osp.join(assert_root,imgs_list[i])+'\n')


