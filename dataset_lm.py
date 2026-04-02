import numpy

from transformers.data.data_collator import DataCollatorWithPadding

import torch
from torch.utils.data import Subset, Dataset
from torch_geometric.data import DenseDataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score

from datasets import Dataset as LMDataset

from utils.dataloader import DataLoader as myDataLoader
from utils.utils import *
from utils.load_data import *
from utils.args import *
from utils.batch import Batch


def dict2graph(dict_list):
    """ Change dict graph back to PyG data."""
    data_list = []
    for data_ in dict_list:
        data = Batch()
        for key in data_.keys():
            setattr(data, key, torch.tensor(data_[key]))
        data_list.append(data)
    return data_list


def graph2txt(data):
    """ Return pyg format data into text format:
        e.g. TBD

    Args:

    Return: graph text information, text format
    """
    edge_index, fc_x = data.fc_edge_index, data.fc_x
    graph_info=[]

    id = 0
    for edge in edge_index.t(): 
            if fc_x[edge[0]][edge[1]] != 0:
            # if True:
                graph_info.append("[{}]-({})-[{}], ".format(
                    edge[0], fc_x[edge[0]][edge[1]].detach().cpu(), edge[1]
                ))
                id += 1
    graph_info = "".join(graph_info)
    return graph_info


def load_dataset_distill(args, 
                         graph_dataset,
                         json_dataset):
    """load text dataset from LLM API.

    Args:
        args (_type_): _description_
        graph_dataset (PYG batch?): _description_
        json_dataset (_type_): _description_

    Returns:
        list[dict]: [{
            - Input (txt):      graph,
            - Question(txt):    question,
            - Answer(txt):      answer, collected from LLM response
        }]
    """
    distill_dataset_list = []
    assert len(graph_dataset) == len(json_dataset)
    
    for i in range(len(graph_dataset)):
        graph_data = graph_dataset[i]
        json_data = json_dataset[i]
        
        graph_input = graph2txt(graph_data)
        description = ""
        for e in json_data["key_features"]:
            description += e + ". "
        
        input = graph_input + ". " + description
        
        question = "What is classification result of input graph?"
        
        label = graph_data.y.detach().cpu().numpy()
        label = "female" if label == 0 else "male"
        explanation = json_data["conclusion"]
        answer = "{}. Thus, the result is {}.".format(explanation, label)

        distill_dataset_list.append({
            "Input":        "Input: " + input,
            "Question":     "Question: " + question,
            "Answer":       "Answer: " + answer
        })
    
    return distill_dataset_list


def load_dataset_distill_mm(args, 
                            graph_dataset,
                            json_dataset):
    """load dataset from LLM API, use raw graph data. (multi_modal)

    Args:
        args (_type_): _description_
        graph_dataset (_type_): _description_
        json_dataset (_type_): _description_

    Returns:
        list[dict]: [{
            - Input (PYG):      graph
            - Question(txt):    can modify here
            - Answer(txt):      answer to the question, from LLM
        }]
    """
    distill_dataset_list = []
    # print(len(graph_dataset), len(json_dataset))
    assert len(graph_dataset) == len(json_dataset)
    
    for i in range(len(graph_dataset)):
        graph = graph_dataset[i]
        json_data = json_dataset[i]
        
        # mark. turn pyg Batch into simple format 
        #* for new data format
        graph_data = {
            "x":            graph.x.numpy(),
            "edge_index":   graph.edge_index.numpy(),
            "y":            graph.y.numpy(),  
            "category":     graph.category.numpy(),
        }
        
        graph_dict = decide_dataset_info_via_id(graph_data["category"])
        description = ""
        for e in json_data["key_features"]:
            description += e + ". "

        #@  question definition
        #@  Define 2 kinds of questions:
        #@      - classification task (SFT stage)
        #@      - description task (instruction tuning task)
        task = graph_dict["task"]
        label = graph_data["y"]
        label = graph_dict["label"][0] if label == 0 else graph_dict["label"][1]

        question = "What is classification result of input graph?"
        question = f"This is a graph for {task}. Please give a detailed summary of the input graph."


        #@ answer definition
        #@ According to quesitions.
        #@     (explanation + ) conclusion + label
        explanation = json_data["conclusion"]
        answer = "{}. {}. Thus, the result is {}.".format(
            description,
            explanation, 
            label)

        distill_dataset_list.append({
            "Input":        graph_data,
            "Question":     question,
            "Answer":       answer
        })

    return distill_dataset_list


def load_dataset_private_mm(graph_dataset):
    """_summary_

    Args:
        graph_dataset (_type_): _description_

    Returns:
        list: _description_
    """
    distill_dataset_list = []
    
    for i in range(len(graph_dataset)):
        graph = graph_dataset[i]
        
        # mark. turn pyg Batch into simple format 
        #* for new data format ()
        graph_data = {
            "x":            graph.x.numpy(),
            "edge_index":   graph.edge_index.numpy(),
            "y":            graph.y.numpy(),  
            "category":     graph.category.numpy(),
        }
        
        graph_dict = decide_dataset_info_via_id(graph_data["category"])

        #@  question definition
        #@      - classification task (SFT stage)
        task = graph_dict["task"]

        # question = f"This is a graph for {task}. What is classification result of input graph?"
        question = f"This is a graph for {task}. Please give a detailed summary of the input graph."


        distill_dataset_list.append({
            "Input":        graph_data,
            "Question":     question,
        })

    return distill_dataset_list



def get_prompt():
    """ (discarded) Return simple prompt. No more information."""
    description     = "You are an experienced Neuro Scientist and can handle well in fMRI related task. \
        Now give you a processed fMRI data which takes the form of a graph, please give the classification result."
    task            = "Gender classification, your answer should be 'female'(0) or 'male'(1)",
    question        = "What is the classification result of the fmri graph."
    answer          = "The classification result is: "

    prompt_dict = {
        "description":  description,
        "task":         task,
        "question":     question,
        "answer":       answer
    }

    result = []
    for key, value in prompt_dict.items():
        result.append("{}: {}; \n".format(key, value))
    result_t = "".join(result)
    return result_t



class GraphTextDataset(Dataset):
    def __init__(self, 
                 data, 
                 text_input,
                 tokenizer,
                 max_length=512):
        """Multi-modal dataset.

        Args:
            data (batch): Data or Batch format
            text_input (list): [str * n]
            tokenizer (tokenizer): skip
            max_length (int, optional): Defaults to 512.
        """        
        self.graph_data = data
        self.text_input = text_input
        assert len(text_input) == len(data)
        self.labels = data.y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        # text = get_prompt()
        graph = self.graph_data[idx]
        text = self.text_input[idx]
        label = self.labels[idx]

        return {
            "graph_input":      graph,
            "text_input":       text,
            "label":            label
        }


class GraphTextDataCollator(DataCollatorWithPadding):
    def __call__(self, batch):
        graph_input = [item["graph_input"] for item in batch]
        text_input  = [item["text_input"]  for item in batch]

        return {
            "graph_input":  graph_input,
            "text_input":   text_input,
        }



"""Graph Text Collator for instruction tuning"""
class GTDataCollator_instruct(DataCollatorWithPadding):
    def __init__(self, 
                 tokenizer, 
                 max_length=384,
                 device="cpu"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, features):
        
        graph_data =        [ex["graph"]      for ex in features]
        input_ids =         [ex["input_ids"]  for ex in features]
        labels =            [ex["labels"]     for ex in features]
        attention_masks =   [ex["attn_masks"] for ex in features]

        #* modify text batch for multi-gpu training
        # mark: need to transform graph: dict => Batch
        graph_data = dict2graph(graph_data)
        graph_data = Batch.from_data_list(graph_data)
        
        if isinstance(input_ids[0], list):
            input_ids = [torch.tensor(seq) for seq in input_ids]
        if isinstance(labels[0], list):
            labels = [torch.tensor(seq) for seq in labels]
        if isinstance(attention_masks[0], list):
            attention_masks = [torch.tensor(seq) for seq in attention_masks]

        # padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        # truncation
        input_ids = input_ids[:, :self.max_length]
        labels = labels[:, :self.max_length]
        attention_masks = attention_masks[:, :self.max_length]

        # modify return type to satisfy parallelism requirements
        batch = {}
        batch["input_ids"]       = input_ids
        batch["graph"]           = graph_data
        batch["labels"]          = labels
        batch["attn_masks"]      = attention_masks
        batch["labels"].to(batch["input_ids"].device, non_blocking=True)        
        
        #* return must be `dict` data
        return batch