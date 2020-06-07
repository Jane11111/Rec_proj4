# -*- coding: utf-8 -*-
# @Time    : 2020-05-17 11:55
# @Author  : zxl
# @FileName: preclean.py

import json
import numpy as np
import pandas as pd
import nltk.tokenize as tk
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


def save_data(file_path,train_path,test_path,val_path):
   f = open(file_path, 'r')
   l = f.readline()
   reviewID_lst = []
   asin_lst = []
   reviewText_lst = []
   overall_lst = []

   while l:
      json_obj = json.loads(l)
      reviewID_lst.append(json_obj['reviewerID'])
      asin_lst.append(json_obj['asin'])
      reviewText_lst.append(json_obj['reviewText'])
      overall_lst.append(json_obj['overall'])

      l = f.readline()

   df = pd.DataFrame(
      {'reviewerID': reviewID_lst, 'asin': asin_lst, 'reviewText': reviewText_lst, 'overall': overall_lst})

   trainset, testset = train_test_split(df[['reviewerID', 'asin', 'reviewText', 'overall']], test_size=0.2,
                                        shuffle=True)
   testset, valset = train_test_split(testset[['reviewerID', 'asin', 'reviewText', 'overall']], test_size=0.5,
                                      shuffle=True)

   trainset.to_csv(train_path, sep='\t', index=False, header=True)
   testset.to_csv(test_path, sep='\t', index=False, header=True)
   valset.to_csv(val_path, sep='\t', index=False, header=True)

   train_user = set(trainset.reviewerID.values)
   test_user = set(testset.reviewerID.values)
   val_user = set(valset.reviewerID.values)
   train_item = set(trainset.asin.values)
   test_item = set(testset.asin.values)
   val_item = set(valset.asin.values)

   print('train user: %d, test_user: %d, val_user: %d' % (len(train_user), len(test_user), len(val_user)))
   print('train item: %d, test_item: %d, val_item: %d' % (len(train_item), len(test_item), len(val_item)))
   print('train & test: %d, train & val: %d' % (
   len(train_user.intersection(test_user)), len(train_user.intersection(val_user))))
   print('train & test: %d, train & val: %d' % (
   len(train_item.intersection(test_item)), len(train_item.intersection(val_item))))


def fit_data(df,vocab_dic):
   tokenizer = tk.WordPunctTokenizer()
   max_review_len =0
   max_voc_len =0
   user_review_dic={}
   item_review_dic={}
   for user,group in df.groupby(['reviewerID']):
       user_review_dic[user]={}
       for asin, review in zip(group.asin,group.reviewText):
           user_review_dic[user][asin] = set([])
           if type(review) is not str:
               continue
           tokens = tokenizer.tokenize(review)

           for token in tokens:
               if token in vocab_dic:
                   user_review_dic[user][asin].add(vocab_dic[token])
           max_voc_len = max(max_voc_len,len(user_review_dic[user][asin]))
       max_review_len = max(max_review_len,len(user_review_dic[user]))

   for asin, group in df.groupby(['asin']):
       item_review_dic[asin] = {}
       for reviewerID, review in zip(group.reviewerID, group.reviewText):
           item_review_dic[asin][reviewerID] = set([])
           if type(review) is not str:
               continue
           tokens = tokenizer.tokenize(review)
           for token in tokens:
               if token in vocab_dic:
                   item_review_dic[asin][reviewerID].add(vocab_dic[token])
           max_voc_len = max(max_voc_len, len(item_review_dic[asin][reviewerID]))
       max_review_len = max(max_review_len, len(item_review_dic[asin]))

   return  user_review_dic,item_review_dic,max_review_len,max_voc_len

def transform_data(df,user_review_dic,item_review_dic,max_review_len,max_voc_len):

    U_review_tensor = []
    U_rev_id_mat = []
    I_review_tensor = []
    I_rev_id_mat = []
    uid_lst = []
    iid_lst = []
    uid_idx = {'none':0}
    iid_idx = {'none':0}
    y_lst = list(df.overall.values)

    for reviewerID,asin in zip(df.reviewerID,df.asin):
        if reviewerID not in uid_idx:
            uid_idx[reviewerID] = len(uid_idx)
        if asin not in iid_idx:
            iid_idx[asin] = len(iid_idx)
        uidx = uid_idx[reviewerID]
        iidx = iid_idx[asin]
        uid_lst.append(uidx)
        iid_lst.append(iidx)
        U_rev_id_vec = []
        I_rev_id_vec = []
        U_review_mat = []
        I_review_mat = []
        cur_u_review_dic = user_review_dic[reviewerID]
        for cur_asin in cur_u_review_dic:
            cur_review = list(cur_u_review_dic[cur_asin]) #是一个list，里面是对应单词的id

            if cur_asin not in iid_idx:
                iid_idx[cur_asin] = len(iid_idx)
            U_rev_id_vec.append(iid_idx[cur_asin]) #TODO 每个用户对应review条数对齐
            cur_review = cur_review[:max_voc_len]
            while len(cur_review) < max_voc_len:# padding
                cur_review.append(0)
            U_review_mat.append(cur_review)#TODO 每个reviewer对齐
        U_rev_id_vec = U_rev_id_vec[:max_review_len]
        while len(U_rev_id_vec) < max_review_len: #padding
            U_rev_id_vec.append(0)
        U_review_mat = U_review_mat[:max_review_len]
        while len(U_review_mat) < max_review_len:
            U_review_mat.append([0 for i in range(max_voc_len)])

        cur_i_review_dic = item_review_dic[asin]
        for cur_reviewerID in cur_i_review_dic:
            cur_review = list(cur_i_review_dic[cur_reviewerID])
            if cur_reviewerID not in uid_idx:
                uid_idx[cur_reviewerID] = len(uid_idx)
            I_rev_id_vec.append(uid_idx[cur_reviewerID]) #TODO 每个item对应review条数对齐
            cur_review = cur_review[:max_voc_len]
            while len(cur_review) < max_voc_len: # padding
                cur_review.append(0)
            I_review_mat.append(cur_review) #TODO 每个reviewer对齐
        I_rev_id_vec = I_rev_id_vec[:max_review_len]
        while len(I_rev_id_vec) < max_review_len: # padding
            I_rev_id_vec.append(0)
        I_review_mat = I_review_mat[:max_review_len]
        while len(I_review_mat) < max_review_len:
            I_review_mat.append([0 for i in range(max_voc_len)])

        U_review_tensor.append(U_review_mat)
        U_rev_id_mat.append(U_rev_id_vec)
        I_review_tensor.append(I_review_mat)
        I_rev_id_mat.append(I_rev_id_vec)

    return np.array(U_review_tensor),np.array(U_rev_id_mat),np.array(uid_lst),np.array(I_review_tensor),np.array(I_rev_id_mat),np.array(iid_lst),np.array(y_lst)



if __name__ == "__main__":

   file_path="./data/Digital_Music_5.json"
   train_path='./data/train.csv'
   test_path='./data/test.csv'
   val_path='./data/val.csv'

   train_U_review_tensor_path = './data/train_U_review_tensor.npy'
   train_U_rev_id_mat_path = './data/train_U_rev_id_mat.npy'
   train_uid_lst_path = './data/train_uid_lst.npy'
   train_I_review_tensor_path = './data/train_I_review_tensor.npy'
   train_I_rev_id_mat_path = './data/train_I_rev_id_mat.npy'
   train_iid_lst_path = './data/train_iid_lst.npy'
   train_y_lst_path = './data/train_y_lst.npy'

   test_U_review_tensor_path = './data/test_U_review_tensor.npy'
   test_U_rev_id_mat_path = './data/test_U_rev_id_mat.npy'
   test_uid_lst_path = './data/test_uid_lst.npy'
   test_I_review_tensor_path = './data/test_I_review_tensor.npy'
   test_I_rev_id_mat_path = './data/test_I_rev_id_mat.npy'
   test_iid_lst_path = './data/test_iid_lst.npy'
   test_y_lst_path = './data/test_y_lst.npy'

   config_path='./config/model.yml'


   # save_data(file_path,train_path,test_path,val_path)

   train_df=pd.read_csv(train_path,sep='\t')
   test_df=pd.read_csv(test_path,sep='\t')
   val_df=pd.read_csv(val_path,sep='\t')

   model = KeyedVectors.load_word2vec_format(
       './data/GoogleNews-vectors-negative300.bin', binary=True)

   print('GoogleNews-vectors-negative300 loaded!')
   vocab_dic={model.index2word[i]:i for i in range(len(model.index2word))}
   user_review_dic, item_review_dic, max_review_len, max_voc_len = fit_data(train_df,vocab_dic)
   print('data fitting finished!')
   print('max review length: %d'%max_review_len)
   print('max voc length: %d'%max_voc_len)


   max_review_len = 20
   max_voc_len = 50

   train_U_review_tensor, train_U_rev_id_mat, train_uid_lst, train_I_review_tensor, train_I_rev_id_mat, train_iid_lst, train_y_lst = transform_data(
       train_df, user_review_dic, item_review_dic, max_review_len, max_voc_len)

   test_U_review_tensor, test_U_rev_id_mat, test_uid_lst, test_I_review_tensor, test_I_rev_id_mat, test_iid_lst, test_y_lst = transform_data(test_df, user_review_dic, item_review_dic, max_review_len,max_voc_len)
   print('test transforming finished!')

   print(test_U_review_tensor.shape)
   print(test_U_rev_id_mat.shape)
   print(test_uid_lst.shape)
   print(test_I_review_tensor.shape)
   print(test_I_rev_id_mat.shape)
   print(test_iid_lst.shape)
   print(test_y_lst.shape)



   print('start saving.....')
   np.save(test_U_review_tensor_path,test_U_review_tensor)
   np.save(test_U_rev_id_mat_path,test_U_rev_id_mat)
   np.save(test_uid_lst_path,test_uid_lst)
   np.save(test_I_review_tensor_path,test_I_review_tensor)
   np.save(test_I_rev_id_mat_path,test_I_rev_id_mat)
   np.save(test_iid_lst_path,test_iid_lst)
   np.save(test_y_lst_path,test_y_lst)

   np.save(train_U_review_tensor_path, train_U_review_tensor)
   np.save(train_U_rev_id_mat_path, train_U_rev_id_mat)
   np.save(train_uid_lst_path, train_uid_lst)
   np.save(train_I_review_tensor_path, train_I_review_tensor)
   np.save(train_I_rev_id_mat_path, train_I_rev_id_mat)
   np.save(train_iid_lst_path, train_iid_lst)
   np.save(train_y_lst_path, train_y_lst)

