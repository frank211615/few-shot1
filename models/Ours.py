import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4,ResNet
from .backbones.FSRM import FSRM
from .backbones.align_attention import AlignAttention
from torchvision.transforms import Resize


class Ours(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,keep_rate=0.5):
        
        super().__init__()
        
        self.resolution = 5*5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
            self.dim = self.num_channel*5*5
            self.patch_sz = (5, 5)
            
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)            
            self.dim = self.num_channel*5*5
            self.patch_sz = (5, 5)


        self.div = self.num_channel ** -0.5
        self.d = self.num_channel

        self.shots = shots
        self.way = way

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([1./3]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([1./3]), requires_grad=True)

        self.fsrm = FSRM(
                sequence_length=self.resolution,
                embedding_dim=self.num_channel,
                num_layers=1,
                num_heads=1,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine')

        self.FAFM = AlignAttention(hidden_size=self.num_channel, inner_size=self.num_channel, num_patch=self.resolution, feats_size=5, drop_prob=0., shot=self.shots[0], keep_rate=keep_rate)
            

    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_map = self.fsrm(feature_map).transpose(1, 2).view(batch_size, self.num_channel, 5, 5)

        return feature_map

    def get_neg_l2_dist(self,inp,way,shot,query_shot):
        d = self.d

        feature_map = self.get_feature_map(inp)
        feature_vector = feature_map.view(-1, d, self.resolution)

        support = feature_vector[:way*shot].view(way, shot, d, *self.patch_sz).permute(0, 2, 1, 3, 4).contiguous()
        query = feature_vector[way*shot:].view(way*query_shot, d, *self.patch_sz)

        scores_F, scores_A = self.FAFM(support, query)

        scores = self.w1*scores_F + self.w2*scores_A

        return scores



    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index


    def forward(self,inp):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=self.way,
                                        shot=self.shots[0],
                                        query_shot=self.shots[1])
        
        logits = neg_l2_dist*self.div*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction
