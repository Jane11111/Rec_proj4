# -*- coding: utf-8 -*-
# @Time    : 2020-06-06 10:00
# @Author  : zxl
# @FileName: NARRE.py


import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from gensim.models import word2vec,KeyedVectors

class CNNNet(nn.Module):
    """
    基于评论生成向量表示
    """
    def __init__(self,config):
        super(CNNNet,self).__init__()
        self.config = config
        model = KeyedVectors.load_word2vec_format(
            config.word2vec_path, binary=True)
        weights = torch.FloatTensor(model.vectors).to(self.config.device)
        self.embedding = nn.Embedding.from_pretrained(weights).cuda()
        self.embedding.weight.requires_grad = False
        self.conv = nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.k_conv_emb,
                                    kernel_size=config.window_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_voc_len-config.window_size+1)).cuda()

    def forward(self,x):
        """
        :param x:  每一个review对应的id
        :return: 经过CNN后对应的向量
        """

        emb = self.embedding(x)
        emb = emb.permute(0,2,1) #TODO 这里可能要改
        # emb = emb.permute(0,1,3,2) #TODO 这里可能要改

        emb = self.conv(emb)
        emb = emb.view(-1,emb.size(1))

        return emb


class AttenNet(nn.Module):
    """
    attention部分
    """
    def __init__(self,config,n):
        """

        :param config:
        :param n: 用户数目 或者 item数目，因为要对id进行embedding
        """
        super(AttenNet,self).__init__()
        self.config = config
        self.embedding  = nn.Embedding(n,config.k_id_emb)
        torch.manual_seed(1)
        self.cnn_net = CNNNet(config)

        Wo = torch.randn((config.k_conv_emb,config.k_att),requires_grad=True).cuda()
        Wid = torch.randn((config.k_id_emb,config.k_att),requires_grad=True).cuda()
        ha = torch.randn((config.k_att,1),requires_grad=True).cuda()
        b1 = torch.randn((config.k_att,),requires_grad=True).cuda()
        b2 = torch.randn((1),requires_grad=True).cuda()

        self.Wo = torch.nn.Parameter(Wo).cuda()
        self.Wid = torch.nn.Parameter(Wid).cuda()
        self.ha = torch.nn.Parameter(ha).cuda()
        self.b1 = torch.nn.Parameter(b1).cuda()
        self.b2 = torch.nn.Parameter(b2).cuda()
        self.register_parameter('Wo',self.Wo)
        self.register_parameter('Wid',self.Wid)
        self.register_parameter('ha',self.ha)
        self.register_parameter('b1',self.b1)
        self.register_parameter('b2',self.b2)

        self.f = nn.Linear(in_features= config.k_conv_emb,
                           out_features=config.k_lfm).cuda()


    def forward(self,review_mat,id_vec):
        """

        :param review_mat: 一个item对应的所有review，每个review对应一个向量，表示对应的vocab ID，所有review向量长度相等
        :param id_mat: 每个review 对应的 user or item ID
        :return: Xu OR Yi
        """
        id_embeddings = self.embedding(id_vec) # batch_size * l * k_conv_emb

        d1,d2,d3= np.array(review_mat.shape)
        review_mat = review_mat.view(d1 * d2, d3)


        review_embedding = self.cnn_net(review_mat) # batch_size * l  * k_conv_emb
        # review_embedding = review_embedding.view(d1,d2,self.config.k_conv_emb)

        id_embeddings = id_embeddings.view(d1 * d2 , self.config.k_id_emb)
        A = (F.relu(review_embedding.mm(self.Wo) + id_embeddings.mm(self.Wid) ) + self.b1).mm(self.ha) + self.b2

        # TODO 归一化
        A = A.view(d1,d2,1) # batch_size * l * 1
        review_embedding = review_embedding.view(d1,d2,self.config.k_conv_emb) # batch_size * l * k_conv_emb

        # weighted_A = torch.zeros([d1, d2, 1]).type(torch.float)

        exp_A = torch.exp(A)

        weighted_A = exp_A / torch.sum(exp_A,dim = 1).view([d1,1,1])


        o = torch.sum(weighted_A * review_embedding,dim = 1)

        o = self.f(o) # batch_size * k_conv_emb

        return o



class NARRENet(nn.Module):
    """
    完整架构
    """

    def __init__(self, config,n,m):
        super(NARRENet,self).__init__()
        self.config = config
        self.n = n #用户数目
        self.m = m #item 数目
        self.u_embedding = nn.Embedding(n,config.k_lfm).cuda()
        self.i_embedding = nn.Embedding(m,config.k_lfm).cuda()
        self.bu = nn.Embedding(n,1).cuda()
        self.bi = nn.Embedding(m,1).cuda()

        W1 = torch.randn((config.k_lfm,1), requires_grad=True).cuda()
        # bu = torch.randn((n,1), requires_grad=True)
        # bi = torch.randn((m,1), requires_grad=True)
        mu = torch.randn((1), requires_grad=True).cuda()

        self.W1 = nn.Parameter(W1)
        # self.bu = nn.Parameter(bu)
        # self.bi = nn.Parameter(bi)
        self.mu = nn.Parameter(mu)

        self.register_parameter('W1',self.W1)
        # self.register_parameter('bu',self.bu)
        # self.register_parameter('bi',self.bi)
        self.register_parameter('mu',self.mu)

        self.u_net = AttenNet(config,m)
        self.i_net = AttenNet(config,n)






    def forward(self,U_review, U_rev_id , I_review, I_rev_id, uid, iid):
        """

        :param U_review: l * vocab_size l是这个用户对应review数量
        :param U_rev_id: l * 1 每个review 对应的id
        :param I_review: l * vocab_size l是这个item对应的review数量
        :param I_rev_id: l * 1 每个review 对应的id
        :param uid: 1 当前用户id
        :param iid: 1 当前item id
        :return:
        """

        qu = self.u_embedding(uid)
        pi = self.i_embedding(iid)
        bu = self.bu(uid)
        bi = self.bi(iid)

        Xu = self.u_net(U_review,U_rev_id)
        Yi = self.i_net(I_review,I_rev_id)

        h0 = (qu + Xu ).mul(pi + Yi)
        o = h0.mm(self.W1) + bu + bi + self.mu

        return o


class NARRE():

    def __init__(self,config,n,m):
        self.config = config
        self.n = n
        self.m = m


    def fit(self,U_review, U_rev_id , I_review, I_rev_id, uid, iid,y):

        device = torch.device(self.config.device)

        net = NARRENet(self.config,self.n,self.m)
        net.to(device)

        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(net.parameters(),lr = self.config.lr,weight_decay= self.config.weight_decay)
        trainset = Data.TensorDataset(torch.from_numpy(U_review),torch.from_numpy(U_rev_id),
                                      torch.from_numpy(I_review),torch.from_numpy(I_rev_id),
                                      torch.from_numpy(uid),torch.from_numpy(iid),
                                      torch.from_numpy(y))
        trainloader = Data.DataLoader(trainset,batch_size = self.config.batch_size,
                                      shuffle = True)

        best_loss = 1000000.0
        running_loss = 0.0
        for epoch in range(self.config.epoch):
            running_loss =0.0
            cur_loss =0.0
            for i, data in enumerate(trainloader,0):
                cur_U_review,cur_U_rev_id,cur_I_review, cur_I_rev_id,cur_uid,cur_iid,cur_y = data
                cur_U_review = cur_U_review.to(device)
                cur_U_rev_id = cur_U_rev_id.to(device)
                cur_I_review = cur_I_review.to(device)
                cur_I_rev_id = cur_I_rev_id.to(device)
                cur_uid = cur_uid.to(device)
                cur_iid = cur_iid.to(device)
                cur_y = cur_y.to(device)

                optimizer.zero_grad()
                outputs = net(cur_U_review,cur_U_rev_id,cur_I_review,cur_I_rev_id,cur_uid,cur_iid)
                loss = criterion(outputs.float().squeeze(),cur_y.float().squeeze()).to(dtype = torch.float64)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                cur_loss += loss.item()

                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d,%d] loss: %.5f' %
                          (epoch + 1, i, float(cur_loss) / (self.config.batch_size * 100)))
                    cur_loss = 0.0
            print('[%d] loss: %.5f' %
                  (epoch + 1, float(running_loss) / len(uid)))
            # if epoch % 5 == 4:
            #     if best_loss > running_loss / len(uid):
            #         best_loss = running_loss / len(uid)
            #         torch.save(self.net.state_dict(), self.config.save_path)
            #         print("---------save best model!---------")
        # if best_loss > running_loss / len(uid):
        torch.save(net.state_dict(), self.config.save_path)
        self.net = net
        # pred_y = net(self.val_U_review,self.val_U_rev_id,
        #              self.val_I_review,self.val_I_rev_id,self.val_uid,self.val_iid)
        # mse = mean_squared_error(self.val_y,pred_y)
        # print('mse: %.5f'%mse)


    def val(self,U_review, U_rev_id , I_review, I_rev_id, uid, iid,y):
        self.val_U_review = U_review
        self.val_U_rev_id = U_rev_id
        self.val_I_review = I_review
        self.val_I_rev_id = I_rev_id
        self.val_uid = uid
        self.val_iid = iid
        self.val_y = y


    def predict(self,U_review,U_rev_id,I_review,I_rev_id,uid,iid):
        device = torch.device(self.config.device)
        # net = NARRENet(self.config,self.n,self.m)
        # net = torch.nn.DataParallel(net,device_ids = [0,1])
        # net.load_state_dict(torch.load(self.config.save_path))
        net = self.net
        testset = Data.TensorDataset(torch.from_numpy(U_review),torch.from_numpy(U_rev_id),
                                      torch.from_numpy(I_review),torch.from_numpy(I_rev_id),
                                      torch.from_numpy(uid),torch.from_numpy(iid))
        testloader = Data.DataLoader(testset, batch_size = self.config.batch_size,
                                     shuffle = False)

        pred_lst = []
        with torch.no_grad():
            for i, data in enumerate(testloader,0):
                cur_U_review, cur_U_rev_id, cur_I_review, cur_I_rev_id, cur_uid, cur_iid= data
                cur_U_review = cur_U_review.to(device)
                cur_U_rev_id = cur_U_rev_id.to(device)
                cur_I_review = cur_I_review.to(device)
                cur_I_rev_id = cur_I_rev_id.to(device)
                cur_uid = cur_uid.to(device)
                cur_iid = cur_iid.to(device)
                cur_res = net(cur_U_review,cur_U_rev_id,cur_I_review,cur_I_rev_id,cur_uid,cur_iid)
                pred_lst.extend(list(cur_res))
        return np.array(pred_lst)








