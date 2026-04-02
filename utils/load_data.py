import os
import sys
import numpy as np
from glob import glob
import pickle
import pandas as pd
from scipy import sparse as sp
import re
import csv
from tqdm import tqdm
from scipy import io
import hashlib
from sklearn.model_selection import (
    KFold, 
    StratifiedShuffleSplit, 
    StratifiedKFold
)


import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

# import dgl
from .construct import *
from .batch import Batch
from .utils import *
from .helper import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def decide_dataset(root):  # 返回标签
    """ Return labels of the dataset. (old)
    Args:
        root: _description_
    Returns:
        label: `dict` {
            'HC':   0,
            'ASD':  1}
    """
    if 'ABIDE' in root:
        class_dict = {
            "HC": 0,
            "ASD": 1,
        }
    elif 'Multi' in root or\
         "xinxiang" in root or\
         "zhongda" in root or\
         "merge" in root:
        class_dict = {
            "HC": 0,
            "MDD": 1,
        }
    elif "HCP" in root:
        class_dict = {
            "female": 0,
            "male": 1,
        }
    else:
        class_dict = {
            "HC": 0,
            "MDD": 1,
        }
    return class_dict


def decide_dataset_info_via_id(id):
    """

    Args:
        id (int): _description_

    Returns:
        dict: {
            "dataset":          
            "category":         
            "task":             
            "label":            
        }
    """
    if id == 1:
        return {
            "dataset":          "Rest_Meta_MDD",
            "category":         1,
            "task":             "MDD diagnois",
            "label":            ["HC", "MDD"]
        }
    
    elif id == 2:
        return {
            "dataset":          "HCP",
            "category":         2,
            "task":             "gender classification",
            "label":            ["female", "male"]
        }
    
    elif id == 3:
        return {
            "dataset":          "ABIDE",
            "category":         3,
            "task":             "ASD diagnois",
            "label":            ["HC", "ASD"]
        }
    
    elif id == 4:
        return {
            "dataset":          "ADHD",
            "category":         4,
            "task":             "ADHD diagnois",
            "label":            ["HC", "ADHD"]
        }
    
    # private dataset
    elif id == 5:
        return {
            "dataset":          "zhongdaxinxiang",
            "category":         5,
            "task":             "MDD diagonois",
            "label":            ["HC", "MDD"]
        }

def deicde_dataset_info(dataset):
    """
        {
        MDD:    1,
        HCP:    2,
        ABIDE:   3,
        ADHD:    4,
        
        zhongdaxinxiang: 5,
    
    }
    Args:
        dataset (str): name of dataset

    Returns:
        dict: information of given dataset
        dict = {
            "dataset":      name of dataset
            "category":     id for dataset,
            "task":         task
        }
    """
    if dataset == "MDD":
        return {
            "dataset":      "Rest_Meta_MDD",
            "category":     1,
            "task":         "MDD diagnois",
        }
    elif dataset == "HCP":
        return {
            "dataset":      "HCP",
            "category":     2,
            "task":         "gender classification",
        }
    elif dataset == "ABIDE":
        return {
            "dataset":      "ABIDE",
            "category":     3,
            "task":         "ASD diagnois",
        }
    elif dataset == "ADHD": # default option
        return {
            "dataset":      "ADHD",
            "category":     4,
            "task":         "ADHD diagnois",            
    }
    
    # for private dataset
    elif dataset == "zhongdaxinxiang":
        return {
            "dataset":      "zhongdaxinxiang",
            "category":     5,
            "task":         "MDD diagonois",
        }

    else:
        return {
            
        }

def read_dataset_fc_regionSeries(root, label_files, files):  
    FC_dir = "RegionSeries.mat"
    if 'ABIDE' in root:
        subj_fc_dir = os.path.join(root, label_files, files)
        subj_mat_fc = np.loadtxt(subj_fc_dir)[:176, :90]

    else: 
        subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
        subj_mat_fc = io.loadmat(subj_fc_dir)['RegionSeries']
    return subj_mat_fc

def read_dataset_fc_edgeWeight(root, label_files, files):  
    FC_dir = "EdgeWeight.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['EdgeWeight']
    return subj_mat_fc

def read_dataset_fc_region_features(root, label_files, files):  
    FC_dir = "region_features_norm.mat"
    subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
    subj_mat_fc = io.loadmat(subj_fc_dir)['region_features']
    return subj_mat_fc

def read_dataset_sc_connectivity(root, label_files, files): 
    SC_connectivity_dir = "DTI_connectivity_count.mat"
    subj_sc_connectivity_dir = os.path.join(root, label_files, files, SC_connectivity_dir)
    subj_sc_connectivity = io.loadmat(subj_sc_connectivity_dir)['connectivity']
    return subj_sc_connectivity


def read_dataset_sc_features(root, label_files, files):  
    SC_features_dir = "DTI_connectivity_count.mat"
    subj_sc_feature_dir = os.path.join(root, label_files, files, SC_features_dir)
    subj_sc_feature = io.loadmat(subj_sc_feature_dir)['connectivity']
    return subj_sc_feature


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    wl_pos_enc = torch.LongTensor(list(node_color_dict.values()))
    return wl_pos_enc

"""Dataset for multi-modal classification"""
class MyOwnDataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None, 
                 args=None):
        self.args = args
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        
        data_sc_list = []
        data_fc_list = []
        
        self.sc = []
        self.fc = []
        self.scfc = []
        self.class_dict = decide_dataset(self.args.path)
        self.y = []

        root = self.args.path
        label_list = os.listdir(root)
        label_list.sort()
        maxx = 0

        for label_files in label_list:
            list = os.listdir(os.path.join(root, label_files))
            list.sort()
            # print(label_files)
            label = torch.LongTensor([self.class_dict[label_files]])
            for files in list:
                ############
                sc_feature = read_dataset_sc_features(root, label_files, files)
                sc_connectivity = read_dataset_sc_connectivity(root, label_files, files)
                
                subj_sc_feature = sc_feature  # 构造特征
                subj_sc_adj = get_sc_adj(sc_connectivity, self.args)  # 构造邻接矩阵

                ############ 230*90
                fc_feature = read_dataset_fc_regionSeries(root, label_files, files)                 # 230*90
                fc_connectivity = read_dataset_fc_regionSeries(root, label_files, files)            # 230*90
                # fc_region_features = read_dataset_fc_region_features(root, label_files, files)

                subj_fc_feature = np.corrcoef(np.transpose(fc_feature))
                subj_fc_adj = get_Pearson_fc(fc_connectivity, self.args)  
                
                #* modify this
                # data_sc_list.append(get_sc_dataloader(subj_sc_adj, subj_sc_feature))            
                # data_fc_list.append(get_fc_dataloader(subj_fc_adj, subj_fc_feature))
                self.y.append(label)

        if self.pre_filter is not None:
            data_sc_list = [data for data in data_sc_list if self.pre_filter(data)]
            data_fc_list = [data for data in data_fc_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            new_data_sc_list = []
            for data in tqdm(data_sc_list):
                new_data_sc_list.append(self.pre_transform(data))
            data_sc_list = new_data_sc_list
                
            new_data_fc_list = []
            for data in tqdm(data_fc_list):
                new_data_fc_list.append(self.pre_transform(data))
            data_fc_list = new_data_fc_list
            
        for _, (sc, fc) in enumerate(zip(data_sc_list, data_fc_list)):
            # print(sc)
            data = Batch()
            for key in sc.keys():
                data.__setattr__('sc_{}'.format(key), sc.__getattr__(key))

            for key in fc.keys():
                data.__setattr__('fc_{}'.format(key), fc.__getattr__(key))
                # data.__setattr__('fc_{}'.format(key), fc.__getattr__(key))
            data.y = self.y[_]

            data_list.append(data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



"""Dataset for instruction tuning."""
class BrainDataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None, 
                 args=None,
                 # below are for FT dataset
                 type="pretrain",
                 finetune_args:dict=None):
        """Brain dataset for instruction tuning.

        Args:
            root (str): saving root path for processed dataset
            transform (optional): _description_. Defaults to None.
            pre_transform (optional): _description_. Defaults to None.
            pre_filter (optional): _description_. Defaults to None.
            args (optional): _description_. Defaults to None.

            type (str): [pretrain, fine-tune]   
            finetune_args (dict): _description_. Defaults to None.

        """
        self.args = args
        self.type = type
        self.finetune_args = finetune_args

        # add this path so that dataset can be loaded correctly 
        sys.path.append(args.root)
        current_script_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_script_path)
        sys.path.append(current_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    
    def process(self):
        if self.type == "pretrain":
            #* pretrain
            #* load Data only
            root = self.args.path
            
            if self.args.dataset == "zhongdaxinxiang":
                print("Private dataset!")
                data_list = load_dataset_private(root, self.args)
            else:
                data_list = load_all_datasets_from_root(root, self.args)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            #* fine-tune
            #* load Data and llm logits
            root = self.args.path
            if self.args.dataset in self.args.dataset_pool:
                data_list = preprocess_dataset_text_logits(
                    self.args,
                    graph_dataset=self.finetune_args["graph"],
                    text_dataset=self.finetune_args["logits"],
                    dataset_name=self.finetune_args["dataset"]
                )
            else:
                print("Private dataset!")
                data_list = load_dataset_private(root, 
                                                self.args, 
                                                self.finetune_args["logits"])

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])


dataset_interval = {
    "MDD":      [2595, 4759],
    "ABIDE":    [1977, 2594],
    "HCP":      [938, 1976],
    "ADHD":     [0, 937]
}

def preprocess_dataset_text_logits(args,
                                   graph_dataset,
                                   text_dataset,
                                   dataset_name):
    """All data in, certain data out.
    
    Args:
        args (Args): args object for arguments
        graph_dataset (Dataset): graph dataset
        text_dataset (Dataset): text dataset
        dataset_name (str): dataset name, 
            options: ["MDD", "ABIDE", "HCP", "ADHD"]
    
    Return:
        list(Data): processed data list (dataset_name)
    """
    assert len(graph_dataset) == len(text_dataset)

    data_list = []
    for idx, data in enumerate(graph_dataset):
        data_ = data
        data_.logits = torch.tensor(text_dataset[idx])
        data_list.append(data_)
    
    start, end = dataset_interval[dataset_name][0], \
                 dataset_interval[dataset_name][1] + 1
    return data_list[start: end]


def load_dataset_private(root_folder,
                         args,
                         logits=None):
    """
    load private dataset for finetune. (zhongdaxinxiang)
    Args:
        root_folder (str): root path for dataset
        args (Args): args object for arguments
        logits (Tensor, optional): logits for text. Only not none when exception is not None.
    
    Returns:
    
    """
    if 'zhongdaxinxiang' in root_folder:
        print("zhongdaxinxiang")
        data_files = glob(os.path.join(root_folder, "**", "**", "*.pkl")) # for BLEG
        # data_files = glob(os.path.join(root_folder, "**", "*.pkl")) # for compare

        data_files.sort()
        data_list = []
        idx = 0
        
        for file in data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            adj = data.adjacency_mat
            feature = np.corrcoef(data.source_mat.T)
            label = 1
            if 'HC' in file:
                label = 0
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))

            data_ = Data(
                x=torch.from_numpy(feature).float(), 
                edge_index=fcedge_index, 
                y=torch.as_tensor([label]).long(),
                category = torch.as_tensor([5]).long(),
            )
            if logits is not None:
                data_.logits = torch.tensor(logits[idx])
            idx += 1
            data_list.append(data_)

        print("number of zhongdaxinxiang dataset is ", len(data_list))
    return data_list



"""A copy from zhangpeng"""
def load_all_datasets_from_root(root_folder, 
                                args,
                                exception=None,
                                logits=None):
    """load dataset from root path for instruction tuning && SFT.
        Datasets: [ADHD, ABIDE, HCP, MDD]

    Args:
        root_folder (str): root path for dataset
        args (Args): args object for arguments
        exception (list, optional): exception list for dataset. Defaults to None.
        logits (Tensor, optional): logits for text. Only not none when exception is not None.

    Returns:
        list(Data): for Data object, format is:
            - x
            - edge_index
            - y
            - category: {
                1 for MDD
                2 for HCP
                3 for ABIDE
                4 for ADHD
            }
            
    """
    fc = []
    exception = [] if not exception else exception

    for dirpath, dirnames ,filenames in os.walk(root_folder):
        for dirname in dirnames:
            root_dir = os.path.join(dirpath, dirname)

            #* rest meta mdd
            if 'MDD' in root_dir and 'MDD' not in exception:
                MDD_num = 0
                print("Rest_meta_MDD", root_dir)
                
                # data_files = glob(os.path.join(root_dir, "**" , "**", "**","*.pkl")) # for BLEG
                data_files = glob(os.path.join(root_dir, "**" , "**", "*.pkl")) # for compare
                data_files.sort()
                # self.fc = []
                for file in data_files:
                    with open(file, "rb") as f:
                        data = pickle.load(f)
                    adj = data.adjacency_mat
                    feature = np.corrcoef(data.source_mat.T)
                    subj_fc_feature = max_min_norm(data.source_mat)
                    x = get_Pearson_fc(subj_fc_feature, args)
                    label = 1
                    if 'HC' in file:
                        label = 0
                    fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                    fc.append(Data(
                        x=torch.from_numpy(x).float(), 
                        edge_index=fcedge_index, 
                        y=torch.as_tensor([label]).long(),
                        category = torch.as_tensor([1]).long()
                    ))
                    MDD_num += 1
                print("MDD num: ", MDD_num)
            
            #* HCP
            elif 'HCP' in root_dir and 'HCP' not in exception: 
                HCP_num = 0
                print("HCP in ", root_dir)
                # data_files = glob(os.path.join(root_dir,"**", "**", "RegionSeries.mat")) # for BLEG
                data_files = glob(os.path.join(root_dir,"**", "RegionSeries.mat")) # for compare
                # print("HCP")
                data_files.sort()

                for file in data_files:
                    subj_fc_mat = io.loadmat(file)['RegionSeries'][: , :]
                    subj_fc_feature = max_min_norm(subj_fc_mat)

                    x = get_Pearson_fc(subj_fc_feature, args)
                    adj = get_Pearson_adj(subj_fc_feature, args.threshold)
                    label = 1
                    if 'female' in file:
                        label = 0
                    fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))

                    fc.append(Data(
                        x=torch.from_numpy(x).float(), 
                        edge_index=fcedge_index, 
                        y=torch.as_tensor([label]).long(),
                        category = torch.as_tensor([2]).long()
                    ))
                    HCP_num += 1
                print("HCP num: ", HCP_num)

            #* ABIDE
            elif 'ABIDE' in root_dir and 'ABIDE' not in exception:
                ABIDE_num = 0
                print("ABIDE in ", root_dir)
                data_files = glob(os.path.join(root_dir, "**" , "**", "**","*.pkl"))
                data_files.sort()
                for file in data_files:
                    with open(file, "rb") as f:
                        data = pickle.load(f)
                    adj = data.adjacency_mat
                    feature = np.corrcoef(data.source_mat.T)
                    subj_fc_feature = max_min_norm(data.source_mat)

                    x = get_Pearson_fc(subj_fc_feature, args)
                    label = 1
                    if 'HC' in file:
                        label = 0
                    fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))

                    fc.append(Data(
                        x=torch.from_numpy(x).float(), 
                        edge_index=fcedge_index, 
                        y=torch.as_tensor([label]).long(),
                        category = torch.as_tensor([3]).long()
                    ))
                    ABIDE_num += 1
                print("ABIDE num: ", ABIDE_num)
            
            #* ADHD
            elif 'ADHD' in root_dir and 'ADHD' not in exception:
                ADHD_num = 0
                print("ADHD in ", root_dir)
                # data_files = glob(os.path.join(root_dir, "**", "**", "**", "sf" + "*_rest_1_aal_TCs.1D")) # for BLEG
                data_files = glob(os.path.join(root_dir, "**", "**", "sf" + "*_rest_1_aal_TCs.1D")) # for compare
                data_files.sort()
                # self.fc = []

                for file in data_files:
                    subj_mat_fc = np.loadtxt(fname=file, skiprows=1, usecols=np.arange(2, 92))  # T*N
                    subj_mat_fc = max_min_norm(subj_mat_fc)
                    adj = get_Pearson_fc(subj_mat_fc, args)
                    feature = np.corrcoef(subj_mat_fc.transpose())

                    label = 1
                    if 'HC' in file:
                        label = 0

                    fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                    fc.append(Data(
                        x=torch.from_numpy(feature).float(), 
                        edge_index=fcedge_index, 
                        y=torch.as_tensor([label]).long(),
                        category = torch.as_tensor([4]).long()
                    ))
                    ADHD_num += 1
                print("ADHD num: ", ADHD_num)

    print('total number of fc is ',len(fc))

    return fc



# mark: Load in all the public datasets.
#* for data:
#*      - x
#*      - edge_index
#*      - y
#*      - task:   MDD/HCP/ASD
#*          - 1: MDD
#*          - 2: ABIDE
#*          - 3: HCP
#*          - 4: ADHD
#*      - file_name: only id for each data 
class FSDataset(object):
    def __init__(self, 
                 args,
                 root_dir, 
                 folds, 
                 seed, 
                 threshold=None, 
                 num_nodes=None, 
                 n_neighbors=None):
        """_summary_

        Args:
            args (_type_): _description_
            root_dir (_type_): _description_
            folds (_type_): _description_
            seed (_type_): _description_
            threshold (_type_, optional): _description_. Defaults to None.
            num_nodes (_type_, optional): _description_. Defaults to None.
            n_neighbors (_type_, optional): _description_. Defaults to None.
        """

        self.args = args
        self.fc = []

        if 'small_MDD' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "*.pkl"))
            data_files.sort()
            # self.fc = []
            for file in data_files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                adj = data.adjacency_mat
                feature = np.corrcoef(data.source_mat.T)
                site_num, label_num = re.match(r".*ROISignals_.(\d*)-(\d)-.+?\.pkl", file).group(1, 2)
                label = 2 - int(label_num)
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))

        elif 'ABIDE_useful' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "**", "*.pkl"))
            data_files.sort()
            # self.fc = []
            for file in data_files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                adj = data.adjacency_mat
                feature = np.corrcoef(data.source_mat.T)
                siteinfo = file.split('/').pop(-2).split('_').pop(0)
                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))

        elif 'lyy' in root_dir:
            data_files = glob(os.path.join(root_dir, '**', '**', '**', "*.1D"))
            data_files.sort()
            # self.fc = []
            
            for file in data_files:
                subj_mat_fc = np.loadtxt(fname=file, skiprows=1, usecols=np.arange(2, 92))  # T*N
                subj_mat_fc = max_min_norm(subj_mat_fc)
                adj = get_Pearson_fc(subj_mat_fc, 0.2, 90)
                feature = np.corrcoef(subj_mat_fc.transpose())
                subject_ID = file.split('\\').pop(-2)
                with open('C:/Users/zxt/Desktop/ABIDE_lyy/Phenotypic_V1_0b_preprocessed1.csv') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        if row['SUB_ID'] == subject_ID:
                            label = int(row['DX_GROUP']) - 1
                            break
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor([label]).long()
                ))

        elif 'ADHD' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "**", "sf" + "*_rest_1_aal_TCs.1D"))
            data_files.sort()
            # self.fc = []

            for file in data_files:
                subj_mat_fc = np.loadtxt(fname=file, skiprows=1, usecols=np.arange(2, 92))  # T*N
                subj_mat_fc = max_min_norm(subj_mat_fc)
                adj = get_Pearson_fc(subj_mat_fc, 0.2, 90)
                feature = np.corrcoef(subj_mat_fc.transpose())

                label = 1
                if 'HC' in file:
                    label = 0

                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))


        elif 'zhongdaxinxiang_new' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "*.pkl"))
            data_files.sort()
            # self.fc = []
            
            for file in data_files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                adj = data.adjacency_mat
                feature = np.corrcoef(data.source_mat.T)
                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))
        
        elif 'wwh' in root_dir:
            data_files = glob(os.path.join(root_dir,"**","*.mat"))
            data_files.sort()
            self.fc = []
            for file in data_files:
                subj_fc_mat=io.loadmat(file)['ROISignals_AAL'][:230,:]
                subj_fc_feature=max_min_norm(subj_fc_mat)
                adj=get_Pearson_fc(subj_fc_feature,0.2)
                feature = np.corrcoef(subj_fc_feature.T)
                label = 1
                if 'HC' in file:
                    label = 0
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))

        elif 'NJNK' in root_dir:
            ids = pd.read_excel('D:/Data/NJNK_FC_roisignals/database_样本信息.xlsx', sheet_name='198').values[:, 6]
            data_files = []
            for id in ids:
                data_files.append(root_dir+'/ROICorrelation_FisherZ_' + id + '.txt')
            self.fc = []

            for file in data_files:
                connectivity = np.loadtxt(file)[:90,:90]
                where_are_inf = np.isinf(connectivity)
                connectivity[where_are_inf] = 1

                # 留下前阈值的边设为1,对角线为1,其余为0,返回[N*N]
                subj_mat_fc_adj = connectivity - np.diag(np.diag(connectivity))  # 减去对角线元素
                subj_fc_adj_up = subj_mat_fc_adj[np.triu_indices(num_nodes, k=1)]  # 对角线及其以下都是0
                subj_fc_adj_list = subj_fc_adj_up.reshape((-1))  # 改成一串,没有行和列
                thindex = int(threshold * subj_fc_adj_list.shape[0])
                thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex - 1]]
                # avoiding Nan
                adj = np.zeros((num_nodes, num_nodes))
                adj[subj_mat_fc_adj > thremax] = 1

                ids = pd.read_excel('D:/Data/NJNK_FC_roisignals/database_样本信息.xlsx', sheet_name='198').values[:, 0]
                labels = pd.read_excel('D:/Data/NJNK_FC_roisignals/database_样本信息.xlsx', sheet_name='198').values[:, 5]
                
                id = int(file.split('/')[-1].split('_')[-2])
                for i in range(len(ids)):
                    if ids[i] == id:
                        label = labels[i]
                        break

                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(connectivity).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))

        elif 'zhongda_data_fmri' in root_dir:
            data_files = glob(os.path.join(root_dir, "**", "**", "RegionSeries.mat"))
            data_files.sort()
            self.fc = []
            for file in data_files:
                subj_fc_mat = io.loadmat(file)['RegionSeries']
                subj_fc_feature = max_min_norm(subj_fc_mat)

                adj = get_Pearson_fc(subj_fc_feature, 0.2)
                feature = np.corrcoef(subj_fc_feature.T)

                label = 0
                if 'RMD' in file:
                    label = 1
                elif 'UMD' in file:
                    label = 2
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
                self.fc.append(Data(
                    x=torch.from_numpy(feature).float(), 
                    edge_index=fcedge_index, 
                    y=torch.as_tensor([label]).long()
                ))            

        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)


    def kfold_split(self, batch_size, test_index):
        assert test_index < self.k_fold
        # valid_index = (test_index + 1) % self.k_fold
        valid_index = test_index
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]

        train_mask = np.ones(len(self.fc))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.fc, train_split.tolist())
        valid_subset = Subset(self.fc, valid_split.tolist())
        test_subset = Subset(self.fc, test_split.tolist())
        return train_subset, valid_subset, test_subset

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.fc)



if __name__ == "__main__":
    dataset = BrainDataset(root="/home/dongrui/code/LLM4Brain/dataset/pretrain", args=None)
    print(len(dataset))