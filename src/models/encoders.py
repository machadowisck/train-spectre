# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet,mobilenet


class PerceptualEncoder(nn.Module):
    def __init__(self, outsize, backbone):
        super(PerceptualEncoder, self).__init__()
        if backbone == "mobilenetv2":
            self.encoder = mobilenet.mobilenet_v2()
            # self.encoder.load_state_dict(torch.load('sftp://connect.westa.seetacloud.com:31455/root/autodl-tmp/code/lhj/spectre/pretrained/mobilenet_v2.pth'))
            # self.encoder.cuda()
            # self.encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
            feature_size = 1280
        elif backbone == "resnet50":
            self.encoder = resnet.load_ResNet50Model() #out: 2048
            feature_size = 2048

        ### regressor
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(256, 53),
        )

        self.backbone = backbone
        self.adapter=True
        if self.adapter:
            # self.adapter_experssion2=Adapter_Conv(96)
            self.adapter_fc=Adapter_FC(53)
            self.adapter_experssion1=Adapter_Conv(320)
    def forward(self, inputs):
        is_video_batch = inputs.ndim == 5

        if self.backbone == 'resnet50':
            features = self.encoder(inputs).squeeze(-1).squeeze(-1)
        else:
            inputs_ = inputs
            if is_video_batch:
                B, T, C, H, W = inputs.shape
                inputs_ = inputs.view(B * T, C, H, W)
            j=0
            for mobile_layer in self.encoder.features:
                inputs_=mobile_layer(inputs_)
                if self.adapter:
                    # if j==13:
                    #     inputs_ = self.adapter_experssion2(inputs_)
                    if inputs_.shape[1]==320:
                        inputs_ = self.adapter_experssion1(inputs_)

                # print(f'第{j}網絡層:{inputs_.shape}')
                #[32,16,24,24,32,32,32,64,64,64,64,96,96,96,160,160,160,320,1280]
                j += 1
            features =inputs_
            # features = self.encoder.features(inputs_)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            if is_video_batch:
                features = features.view(B, T, -1)

        features = features
        if is_video_batch:
            features = features.permute(0, 2, 1)
        else:
            features = features.permute(1,0).unsqueeze(0)

        features = self.temporal(features)


        if is_video_batch:
            features = features.permute(0, 2, 1)
        else:
            features = features.squeeze(0).permute(1,0)

        parameters = self.layers(features)
        if self.adapter:
            parameters=self.adapter_fc(parameters)
        parameters[...,50] = F.relu(parameters[...,50].clone()) # jaw x is highly improbably negative and can introduce artifacts

        return parameters[...,:50], parameters[...,50:]



class ResnetEncoder(nn.Module):
    def __init__(self, outsize):
        super(ResnetEncoder, self).__init__()
        self.adapter=False
        # if self.adapter:
        #     self.adapter_fc = Adapter_FC(1,m_channels=3)

        feature_size = 2048

        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )


    def forward(self, inputs):
        inputs_ = inputs
        if inputs.ndim == 5: # batch of videos
            B, T, C, H, W = inputs.shape
            inputs_ = inputs.view(B * T, C, H, W)
        features = self.encoder(inputs_)
        parameters = self.layers(features)
        # if self.adapter:
        #     pose_param=parameters[:,203:204]
        #     pose_param = self.adapter_fc(pose_param)
        #     parameters[:, 203] = pose_param[:,0]
            # pose_param=torch.cat([parameters[:,150:200],parameters[:,203:206]],dim=-1)
            # pose_param = self.adapter_fc(pose_param)
            # parameters[:, 150:200]=pose_param[:,:50]
            # parameters[:, 203:206]=pose_param[:,50:]
            # mouth_parame=torch.cat([parameters[:,150:151],parameters[:,203:204]],dim=-1)
            # mouth_parame = self.adapter_fc(mouth_parame)
            # parameters[:, 150]=mouth_parame[:,0]
            # parameters[:, 203]=mouth_parame[:,1]

            # parameters[:,3:56] = self.adapter_fc(parameters[:,3:56])

        if inputs.ndim == 5: # batch of videos
            parameters = parameters.view(B, T, -1)
        return parameters

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1_3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class convlayer(nn.Module):
    def __init__(self,in_channels,out_channels,conv_type='conv2d'):
        super(convlayer, self).__init__()
        if conv_type =='conv2d':
            self.conv1 = conv3x3(in_channels, out_channels, 1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = conv1_3x3(in_channels, out_channels, 1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
class Adapter_Conv(nn.Module):
    def __init__(self, in_channels, m_channels=16,conv_type='conv2d'):
        super(Adapter_Conv, self).__init__()
        self.conv1 = convlayer(in_channels, m_channels,conv_type)
        self.conv2 = convlayer(m_channels, m_channels,conv_type)
        self.conv3 = convlayer(m_channels, in_channels,conv_type)

    def forward(self,x):
        output=self.conv1(x)
        output=self.conv2(output)
        output=self.conv3(output)
        output=output+x
        return  output

class Adapter_FC(nn.Module):
    def __init__(self, in_channels, m_channels=16,):
        super(Adapter_FC, self).__init__()
        self.fc1 =nn.Sequential(nn.Linear(in_channels,m_channels),nn.ReLU())
        self.fc2= nn.Sequential(nn.Linear(m_channels, m_channels),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(m_channels, in_channels))

    def forward(self,x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        output=output+x
        return  output