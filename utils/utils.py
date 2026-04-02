import time
import os
import argparse
import numpy as np
import collections
import csv
import pandas as pd
from sklearn.model_selection import (
    KFold, 
    StratifiedKFold, 
    train_test_split
)

import torch
import random
from itertools import product
import json
import logging

import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data

import collections.abc as container_abcs

# from .load_data import *
from .batch import Batch

int_classes = int


#* ============================================================= *#
#* =============== dataset load function ======================= *#
#* ============================================================= *#

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

def load_text_dataset_private(args,
                              dataset):
    """_summary_

    Args:
        args (_type_): _description_
        dataset (list(data)): _description_
    
    Return:
        list(str):
    """
    assert len(dataset) > 0
    instruct_prompt = '''You are an experienced neuroscience researcher.
                    Now please give a description of the fMRI graph data from {} dataset. The task is {}. The result is '{}' or '{}'. 
                    Data is preprocessed through AAL template. The prediction result of the graph is "
                    '''

    text_list = []

    for idx, data in enumerate(dataset):
        id = data.category.detach().cpu().numpy()
        data_dict = decide_dataset_info_via_id(id)
        instruction = instruct_prompt.format(data_dict["dataset"], 
                                            data_dict["task"], 
                                            data_dict["label"][0], 
                                            data_dict["label"][1])
        text_list.append(instruction)
        
    return text_list



def load_text_dataset_pretrain(args,
                               root_path, 
                               max_length):
    """load pretrain dataset as text format. (HCP + MDD + ADHD + ABIDE)

    Args:
        args (args): 
        root_path (str): root_path for raw text_data
        max_length (int): make sure len(text_dataset) == max length
    
    Return:
        results (list(str)): SFT data format is:
            short features + question + answer (<sep>)
    """
    results = []
    json_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.json') and "settings" not in file:
                json_files.append(os.path.join(root, file))
    
    json_files.sort(key=lambda x : int(os.path.basename(x).replace('.json', '')))

    for file in json_files:
        text = json_2_text(file)

        # add question && answer for SFT stage
        if args.training_type == "sft":
            question = f"</question> What is the classification result of the input fmri graph?</question>"
            answer = f"</answer>"
            text = text + question + answer

        # print(f"length of {file} is {len(text)}")
        results.append(text)

    print(f"length of text dataset is {len(results)}")
    assert len(results) <= max_length
    return results


def load_json_dataset_pretrain(args,
                               root_path, 
                               max_length):
    """load pretrain dataset as json format. (HCP + MDD + ADHD + ABIDE)

    Args:
        args (args): _description_
        root_path (str): root path of the json dataset
        max_length (int): max_length (length of graph dataset)
    
    Return:
        list(json): 
    """
    json_results = []
    json_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.json') and "settings" not in file:
                json_files.append(os.path.join(root, file))
    
    json_files.sort(key=lambda x : int(os.path.basename(x).replace('.json', '')))
    
    for file in json_files:
        with open(file, 'r', encoding="utf-8") as f:
            try:
                # print(file)
                content = f.read()
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 文件时出错：{file}")
                print(f"错误信息：{e}")
                print(f"问题位置附近的文本：")
                print(content[e.pos:e.pos + 50])  # 打印出错误位置附近的文本
                return None
            json_results.append(data)

        f.close()

    print(f"length of text dataset is {len(json_results)}")
    assert len(json_results) <= max_length
    return json_results


def setup_logging(file_name):
    # 创建 logger
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 创建文件处理器，将日志写入指定文件
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加处理器到 logger
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger


def logging_result(args: None,
                   result_dict: dict,
                   result_path: str) -> None:

    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 创建文件处理器，将日志写入指定文件
    file_handler = logging.FileHandler(result_path)
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加处理器到 logger
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(file_handler)

    logger.info("=" * 20)
    # logger.info(f"Args: {args}")
    for key, value in result_dict.items():
        logger.info(f"{key}: {value}")

    logger.removeHandler(file_handler)


def save_as_csv(args, csv_file:str, save_results:dict=None):

    df = pd.DataFrame([save_results])

    if os.path.isfile(csv_file):
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(csv_file, index=False)


def run_experiment():
    result = {
        "accuracy": random.uniform(0.8, 1.0),  # 随机生成一个准确率
        "loss": random.uniform(0.1, 0.5),      # 随机生成一个损失值
        "learning_rate": random.uniform(0.001, 0.01),  # 随机生成一个学习率
        "batch_size": random.randint(16, 64)  # 随机生成一个批量大小
    }
    if random.choice([True, False]):
        result["experiments"] = random.randint(1, 10)
    return result


def default_collate(batch):
    return batch

def multimodal_collate(batch):
    graph_data_list = [item["graph_input"] for item in batch]
    text_data_list = [item["text_input"] for item in batch]
    labels = [item["label"] for item in batch]

    graph_batch = Batch.from_data_list(graph_data_list)

    return graph_batch, text_data_list


def json_2_text(json_path):
    """convert one single json file into text str. 

    Args:
        json_path (str): json_path for given json object

    Returns:
        str: json -> text
    """
    with open(json_path, 'r', encoding="utf-8") as file:
        try:
            content = file.read()
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"解析 JSON 文件时出错：{json_path}")
            print(f"错误信息：{e}")
            print(f"问题位置附近的文本：")
            print(content[e.pos:e.pos + 50])  # 打印出错误位置附近的文本
            return None

    file.close()

    text = ""
    for key, value in data.items():
        if isinstance(value, list):
            value = "".join(value)
        else:
            current_item = f'{key}: {value}. '
            text = text + current_item

    return text

def create_text_dataset(args, 
                        length=1039):
    """ Return text dataset list. (single dataset only)

    Args:
        args (args): _description_
        length (int, optional): _description_. Defaults to 1039.

    Returns:
        list(str): list of all the text data
    """
    text_root = "{}/text_data/{}".format(args.root, args.dataset)
    text_results = []
    for i in range(length):
        text_path = os.path.join(text_root, f"{i}.json")
        text = json_2_text(text_path)

        # add question for SFT stage
        if args.training_type == "sft":
            question = f"</question> What is the classification result of the input fmri graph?</question>"
            text = text + question
        text_results.append(text)

    # print(text_results[0])
    return text_results

def create_json_dataset(args,
                        length=1039):
    """Return json dataset (single dataset only)

    Args:
        args (args): _description_
        length (int, optional): _description_. Defaults to 1039.

    Returns:
        list(json): json dataset
    """
    text_root = "{}/text_data/{}".format(args.root, args.dataset)
    json_results = []
    for i in range(length):
        json_path = os.path.join(text_root, f"{i}.json")

        with open(json_path, 'r', encoding="utf-8") as file:
            try:
                content = file.read()
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 文件时出错：{json_path}")
                print(f"错误信息：{e}")
                print(f"问题位置附近的文本：")
                print(content[e.pos:e.pos + 50])  # 打印出错误位置附近的文本
                return None
            json_results.append(data)

        file.close()

    return json_results


#* ============================================================= *#
#* =============== function definition ========================= *#
#* ============================================================= *#


def contrastive_loss(graph_embeddings, 
                     text_embeddings, 
                     temperature=0.1):
    """_summary_

    Args:
        graph_embeddings (_type_): _description_
        text_embeddings (_type_): _description_
        temperature (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    graph_embeddings = graph_embeddings / graph_embeddings.norm(dim=1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    logits = torch.matmul(graph_embeddings, text_embeddings.T) / temperature

    labels = torch.arange(len(graph_embeddings)).to(graph_embeddings.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


def copy_module_weights(source_modules, target_modules):
    assert len(source_modules) == len(target_modules), "ModuleLists must have the same length"
    
    for source_module, target_module in zip(source_modules, target_modules):
        assert isinstance(target_module, type(source_module)), "Modules must be of the same type"
        
        if hasattr(source_module, 'state_dict'):
            source_state_dict = source_module.state_dict()
            target_state_dict = target_module.state_dict()
            
            assert set(source_state_dict.keys()) == set(target_state_dict.keys()), "Module state_dicts must have the same keys"
            
            for key in source_state_dict:
                target_state_dict[key].copy_(source_state_dict[key])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def KL_loss(teacher_output, student_output, T=3):   
    p_s = F.log_softmax(student_output / T, dim=1)
    p_t = F.softmax(teacher_output / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='sum') * (T ** 2) / student_output.shape[0]

    return loss

def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())



#* ============================================================= *#
#* =============== training definition ========================= *#
#* ============================================================= *#


def few_shot_k_way_split(dataset,
                         k:int=5,
                         q:int=-1,
                         seed:int=42):
    """few-shot learning for k-way scene

    Args:
        dataset (_type_): _description_
        k (int, optional): number of samples within each class for training.
        q (int, optional): number of samples within each clash for testing. 
                           If q == -1, then all the samples will be used for testing.
        seed (int, optional): _description_. Defaults to 42.
        
    Returns:
        tuple (train/val/test): train_indices, test_indices, test_indices
    """
    setup_seed(seed)
    class_2_dict = collections.defaultdict(list)
    for idx, data in enumerate(dataset):
        label = data.y.item()
        class_2_dict[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    for cls, indices in class_2_dict.items():
        np.random.shuffle(indices)
        if len(indices) < k:
            raise ValueError(f"Class {cls} has fewer samples than expected. Skipping this class.")

        train_indices.extend(indices[:k])

        if q == -1:
            test_indices.extend(indices[k:])
        else:
            remaining_indices = indices[k:]
            q_ = min(q, len(remaining_indices))
            test_indices.extend(indices[k: k+q_])

    # print(len(train_indices), len(test_indices), len(dataset))

    return np.array(train_indices), np.array(test_indices), np.array(test_indices)

def few_shot_split(dataset, 
                    train_ratio=0.9, 
                    random_state=42):
    """
    splitting ratio for few-shot scene.

    Args:
        dataset (dataset): _description_
        train_ratio (float, optional): Defaults to 0.9.
            (1) ratio < 1: for ratio few shot scene
            (2) ratio >= 1: for absolute few shot
        random_state (int, optional): Defaults to 42.
    
    Returns:
        list: train, val, test (indices)
    """
    setup_seed(random_state)
    indices = np.arange(len(dataset))

    np.random.shuffle(indices)

    train_size = int(len(dataset) * train_ratio) 
    val_size = int(len(dataset) * 0.1)
    test_size = int(len(dataset)) - train_size - val_size

    train_indices = indices[:train_size]
    test_indices = indices[train_size: train_size + test_size]
    val_indices = indices[train_size + test_size:]

    return train_indices, val_indices, test_indices
    


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices

def cross_validate_multimodal(folds, dataset):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))  # 获取数据集的索引列表
    for train_indices, val_indices in kf.split(indices):
        yield train_indices, val_indices, val_indices

def cross_validate(folds, dataset):
    # 交叉验证
    skf = StratifiedKFold(folds, shuffle=True)

    test_indices, train_indices = [], []
    # modify: data.y => data["label"]
    # print(dataset.indices())
    # print(dataset["label"])
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    # val_indices = [test_indices[i - 1] for i in range(folds)]
    val_indices = test_indices

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, val_indices, test_indices

def kfold_split(self, test_index, args):
    self.k_fold = args.repetitions
    self.batch_size = args.batch_size
    assert test_index < self.k_fold
    valid_index = test_index
    test_split = self.k_fold_split[test_index]
    valid_split = self.k_fold_split[valid_index]

    train_mask = np.ones(len(self.choose_data))
    train_mask[test_split] = 0
    train_mask[valid_split] = 0
    train_split = train_mask.nonzero()[0]

    train_subset = Subset(self.choose_data, train_split.tolist())
    valid_subset = Subset(self.choose_data, valid_split.tolist())
    test_subset = Subset(self.choose_data, test_split.tolist())

    # train_subset = GraphDataset(train_subset, None, degree=True)
    # valid_subset = GraphDataset(valid_subset, None, degree=True)
    # test_subset = GraphDataset(test_subset, None, degree=True)

    # train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, collate_fn=train_subset.collate_fn())
    # val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False, collate_fn=valid_subset.collate_fn())
    # test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, collate_fn=test_subset.collate_fn())
    
    train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)


    return train_split, train_subset, train_loader, valid_split, valid_subset, val_loader, test_split, test_subset, test_loader


def grid_search(args, param_names, params, function):
    """Grid search interface. 

    Args:
        param_names [list(str)]: name of all the hyper-parameters in `list` format.
            [name1, name2, ...]
        params [list(list)]: values of all hyper-parameters in `list` format
            [
                [val_11, val_12, val_13, ...],
                [val_21, val_22, val_23, ...],
            ]
    Return:
    """
    for upd_param in product(*params):
        print("Current parameter sets is :{}".format(upd_param))
        #* update args
        for idx, val in enumerate(upd_param):
            print(idx, val)
            setattr(args, param_names[idx], upd_param[idx]) 
        function(args)


def load_json_param(json_path):
    """load json file for hyper-parameters.

    Args:
        json_path (str): _description_
    
    Returns:
        dict: _description_
    """
    
    with open(json_path, 'r', encoding="utf-8") as file:
        try:
            content = file.read()
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"解析 JSON 文件时出错：{json_path}")
            print(f"错误信息：{e}")
            print(f"问题位置附近的文本：")
            print(content[e.pos:e.pos + 50])  # 打印出错误位置附近的文本
            return None

    file.close()

    return data


def save_json_param(json_path, 
                    json_dict):
    """save json file for hyper-parameters.

    Args:
        json_path (str): _description_
        json_dict (dict): _description_
    """

    with open(json_path, 'w', encoding="utf-8") as file:
        json.dump(json_dict, file, indent=4)
    file.close()
    print(f"save json file successfully for {json_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Convolutional Network')
    parser.add_argument('--dataset', type=str, default='test',
                        help='Choose a dataset from: cora, citeseer, pubmed')
    
    csv_file = "../logs/csv/test.csv"
    for i in range(10):
        result = run_experiment()
        result["test"] = "test"
        print(result)
        save_as_csv(args=parser,
                    csv_file=csv_file,
                    save_results=result)