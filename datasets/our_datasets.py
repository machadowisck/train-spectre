import os
import torch
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
import random
import pickle
from .data_utils import landmarks_interpolate
import logging
import random
class OurSpectreDataset(Dataset):
    def __init__(self):
        with open('train_spectre.txt','r') as f:
            self.meta_list=f.readlines()
        self.meta_list=[meta.strip().split(',') for meta in self.meta_list]

        self.K=20
        self.size=224
        self.crop_landmark3DReconstruction_thresh=1.4
        self.min_ratio_crop_size_w=1.2
        self.min_ratio_crop_size_h=1.2
        self.ratio_crop_size_interval=0.1
    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, index):
        real_index=random.randint(0,int(self.meta_list[index][-1])-1-self.K)
        self.imgs_dir=osp.join(self.meta_list[index][0],'crop_head_imgs')
        self.landmarks=np.load(osp.join(self.meta_list[index][0],'crop_head_info.npy'),allow_pickle=True).item()['landmarks']

        kpt_list=[]
        images_list=[]
        for  i in range(self.K):  # TODO: 这里会不会出现溢出的情况
            img_index=i+real_index
            img_dir=osp.join(self.imgs_dir,str(img_index+1).zfill(6)+'.png')
            img=cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kpt=self.algin_98lmkTo68(self.landmarks[img_index])

            cropped_image,cropped_kpt=self.crop_face(img,kpt)
            images_list.append(cropped_image.transpose(2, 0, 1))

            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.size * 2 - 1
            kpt_list.append(cropped_kpt)

        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.cat((kpt_array, torch.ones((kpt_array.shape[0],kpt_array.shape[1], 1))), dim=-1)
        return  {
            'image': images_array,
            'landmark': kpt_array,
            'vid_name': 'vid_name',
        }

    #寫一個函數，判斷圖片是否存在越界的情況
    def judge_img(self,img,kpt):
        x_min, y_min, w, h = cv2.boundingRect(kpt.astype(np.int32))
        if x_min<0 or y_min<0 or x_min+w>img.shape[1] or y_min+h>img.shape[0]:
            return False
        return True

    def crop_face(self,img, kpt, expand_ratio=0.2):
        # 计算mask的外接矩形
        crop_face_head_scale_w = self.crop_landmark3DReconstruction_thresh
        crop_face_head_scale_h = self.crop_landmark3DReconstruction_thresh

        # 计算mask的外接矩形
        x_min, y_min, w, h = cv2.boundingRect(kpt.astype(np.int32))
        while crop_face_head_scale_w > self.min_ratio_crop_size_w and crop_face_head_scale_h > self.min_ratio_crop_size_h:
            expand_ratio_w = (crop_face_head_scale_w - 1) / 2
            expand_ratio_h = (crop_face_head_scale_h - 1) / 2

            x_cop = int(x_min - expand_ratio_w * w)
            y_crop = int(y_min - expand_ratio_h * h)
            w_crop = int(w * (1 + 2 * expand_ratio_w))
            h_cop = int(h * (1 + 2 * expand_ratio_h))

            if x_cop < 0 or x_cop + w_crop > img.shape[1]:
                crop_face_head_scale_w -= self.ratio_crop_size_interval
                continue
            elif y_crop < 0 or y_crop + h_cop > img.shape[0]:
                crop_face_head_scale_h -= self.ratio_crop_size_interval
                continue
            else:
                break

        crop_bbox = [x_cop, y_crop, x_cop + w_crop, y_crop + h_cop]
        img_crop = img[y_crop:y_crop + h_cop, x_cop:x_cop + w_crop, :]
        scale_w = float(self.size) / img_crop.shape[0]
        scale_h = float(self.size) / img_crop.shape[1]
        img_crop = cv2.resize(img_crop, (self.size, self.size))
        kpt_zoom = self.resize_kpt(kpt, crop_bbox, scale_w, scale_h)
        # self.visual_kpt(kpt_zoom,img_crop,'test.png')
        img_crop = img_crop / 255.
        return img_crop, kpt_zoom

    def resize_kpt(self,kpt,crop_bbox,scale_w,scale_h):
        kpt_crop=kpt.copy()
        kpt_crop[:,0]=(kpt[:,0]-crop_bbox[0])*scale_h
        kpt_crop[:,1]=(kpt[:,1]-crop_bbox[1])*scale_w
        return kpt_crop

    def visual_kpt(self,kpt,img,name):
        image=img.copy()
        for j in range(kpt.shape[0]):
            cv2.circle(image, (int(kpt[j][0]), int(kpt[j][1])), 2, (0, 0, 255), -1)
        cv2.imwrite(name,image)
        return

    def algin_98lmkTo68(self,kpt):
        lmk98to68 = {
        'face_edge_l':[i*2 for i in range(9)],
        'face_edge_t':[i*2 for i in range(9,17)],
        'eyebrow_l':[33+i for i in range(5)],
        'eyebrow_r': [42 + i for i in range(5)],
        'nose':[51+i for i in range(9)],
        'eye_l':[60,61,63,64,65,67],
        'eye_r':[68,69,71,72,73,75],
        'mouth_e':[76+i for i in range(12)],
        'mouth_i':[88+i for i in range(8)]
        }
        kpt_68=[]
        list_i=[]
        for key, value in lmk98to68.items():
            for i in value:
                list_i.append(i)
                kpt_68.append(kpt[i])
        # print(f'list:{list_i}')
        kpt_68 = np.array(kpt_68)
        return kpt_68
def get_datasets_OURS(cfg):
    return OurSpectreDataset(), \
        OurSpectreDataset(), \
        OurSpectreDataset()