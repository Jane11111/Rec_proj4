# -*- coding: utf-8 -*-
# @Time    : 2020-06-06 17:15
# @Author  : zxl
# @FileName: main.py

import numpy as np
from model.NARRE import NARRE
from Util.ConfigLoader import get_config
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
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
    config_path = './config/model.yml'

    train_U_review_tensor = np.load(train_U_review_tensor_path)
    train_U_rev_id_mat = np.load(train_U_rev_id_mat_path)
    train_uid_lst = np.load(train_uid_lst_path)
    train_I_review_tensor = np.load(train_I_review_tensor_path)
    train_I_rev_id_mat = np.load(train_I_rev_id_mat_path)
    train_iid_lst = np.load(train_iid_lst_path)
    train_y_lst = np.load(train_y_lst_path)

    test_U_review_tensor = np.load(test_U_review_tensor_path)
    test_U_rev_id_mat = np.load(test_U_rev_id_mat_path)
    test_uid_lst = np.load(test_uid_lst_path)
    test_I_review_tensor = np.load(test_I_review_tensor_path)
    test_I_rev_id_mat = np.load(test_I_rev_id_mat_path)
    test_iid_lst = np.load(test_iid_lst_path)
    test_y_lst = np.load(test_y_lst_path)

    n = 5542
    m = 3569

    print('data loaded!')

    config = get_config(config_path)

    model = NARRE(config,n,m)
    print('start training!')
    # model.fit(test_U_review_tensor,test_U_rev_id_mat,test_I_review_tensor,test_I_rev_id_mat,
    #           test_uid_lst,test_iid_lst,test_y_lst)
    model.val(test_U_review_tensor, test_U_rev_id_mat, test_I_review_tensor, test_I_rev_id_mat,
                  test_uid_lst, test_iid_lst,test_y_lst)
    model.fit(train_U_review_tensor, train_U_rev_id_mat, train_I_review_tensor, train_I_rev_id_mat,
              train_uid_lst, train_iid_lst, train_y_lst)
    print('training finished!')
    y_pred = model.predict(test_U_review_tensor,test_U_rev_id_mat,test_I_review_tensor,test_I_rev_id_mat,
                           test_uid_lst,test_iid_lst)
    mse = mean_squared_error(test_y_lst,y_pred)
    print("mse: %.5f"%mse)



