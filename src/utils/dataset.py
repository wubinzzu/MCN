# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path'] + self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

        self.ori_df = None

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)  # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num / tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num / tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)


class SRecDataset(RecDataset):
    def __init__(self, config, df=None):
        super().__init__(config, df)

        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.time_field = self.config['TIME_FIELD']
        self.load_inter_graph_with_time(config['inter_file_name'])
        if self.config['drop_one']:
            self.data_augmentation()



    def load_inter_graph_with_time(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.time_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def data_augmentation(self):
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        df_sorted = self.df.sort_values(by=[self.uid_field, self.time_field], ascending=True)
        df_sorted = df_sorted.reset_index(drop=True)
        last_uid = None
        drop_index, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(df_sorted[self.uid_field]):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
                drop_index.append(i)
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)
        shape = (item_list_index.shape[0], max_item_list_len)
        value = df_sorted[self.iid_field]
        item_list = np.full(shape, self.item_num, dtype=np.int64)
        for i, (index, length) in enumerate(
                zip(item_list_index, item_list_length)
        ):
            item_list[i][-length:] = value[index].values
        df_sorted = df_sorted.iloc[target_index]
        df_sorted[self.config['ITEM_LIST_FIELD']] = item_list.tolist()
        self.df = df_sorted.drop(self.time_field, axis=1)
        if self.config['model'] in self.config['seq2seq_models']:
            self.ori_df = self.df.copy(deep=True)
            df = self.df
            # 对每个用户组，保留最后一个标签为0的记录
            last_0_df = df[df['x_label'] == 0].groupby('userID').tail(1)
            # 保留标签为1或2的记录
            label_1_2_df = df[df['x_label'].isin([1, 2])]
            # 合并保留的记录
            df = pd.concat([last_0_df, label_1_2_df]).sort_values(['userID', 'x_label']).reset_index(drop=True)
            self.df = df

    # def data_augmentation(self):
    #     max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
    #     df_sorted = self.df.sort_values(by=[self.uid_field, self.time_field], ascending=True)
    #     df_sorted = df_sorted.reset_index(drop=True)
    #     last_uid = None
    #     item_list_index, target_index, item_list_length = [], [], []
    #     seq_start = 0
    #     for i, uid in enumerate(df_sorted[self.uid_field]):
    #         if last_uid != uid:
    #             last_uid = uid
    #             seq_start = i
    #         else:
    #             if i - seq_start > max_item_list_len:
    #                 seq_start += 1
    #             item_list_index.append(slice(seq_start, i))
    #             target_index.append(i)
    #             item_list_length.append(i - seq_start)
    #     item_list_index = np.array(item_list_index)
    #     target_index = np.array(target_index)
    #     item_list_length = np.array(item_list_length, dtype=np.int64)
    #     shape = (df_sorted.shape[0], max_item_list_len)
    #     value = df_sorted[self.iid_field]
    #     item_list = np.full(shape, self.item_num, dtype=np.int64)
    #     for ii, (i, index, length) in enumerate(
    #             zip(target_index, item_list_index, item_list_length)
    #     ):
    #         item_list[i][-length:] = value[index].values
    #     df_sorted[self.config['ITEM_LIST_FIELD']] = item_list.tolist()
    #     self.df = df_sorted.drop(self.time_field, axis=1)
