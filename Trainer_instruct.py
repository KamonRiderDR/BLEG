""" This Trainer is for instruction tuning for pretrained-LM.
    Use text data (QA) and graph data together. 
    Training stategy contains PEFT tuning && LoRA. We also train a GCN encoder here.
    
    [Options]:
        -- GPT2-base:    534M,
           GCN:          1024
        -- GPT2-large:   1.5B,
           GCN:          1600
"""
import os
import argparse
import numpy as np
import json
from itertools import product
import time
from copy import deepcopy
from utils.args import *


args = get_args()
device = ""
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device}'

import torch
from torch.utils.data import Subset, random_split
from torch_geometric.data import DenseDataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


from transformers import (
    BioGptForCausalLM, 
    BioGptTokenizer,
    BioGptForSequenceClassification,
    BioGptModel,
    BioGptForTokenClassification,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

from transformers import pipeline, set_seed
from transformers import (
    DataCollatorWithPadding
    )
from accelerate import (
    infer_auto_device_map, 
    load_checkpoint_in_model,
    Accelerator
)

from datasets import Dataset as DatasetLM

from peft import LoraConfig, get_peft_model, PeftModel

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score

from utils.dataloader import DataLoader as myDataLoader
from utils.utils import *
from utils.load_data import *

from models.BrainLLaVA import (
    BioGPTWithAdapter, 
    BrainAdapter, 
    BrainLLaVA
)
from models.GCN import GCN
from models.model_pretrain import BioGPTAdapter4InstructionTuning

from dataset_lm import (
    GraphTextDataset, 
    GraphTextDataCollator, 
    load_dataset_distill,
    load_dataset_distill_mm,
    GTDataCollator_instruct,
)



def preprocess_function(examples):
    """preprocess function for instruction tuning. (text only)

    Args:
        examples (dict?):{
            - Input (text)
            - Question (text)
            - Answer (text)
        }

    Returns:
        tokenized: TBD
    """
    text ="{}\n{}\n{}".format(
                            examples["Input"],
                            examples["Question"],
                            examples["Answer"])

    tokenized = tokenizer(text, 
                          truncation=True, 
                          padding="max_length", 
                          max_length=256,
                          return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()

    assert tokenized["labels"].shape == tokenized["input_ids"].shape
    return tokenized

def preprocess_function_instruct(examples):
    """ preprocess function for multi_modal instruction tuning.

    Args:
        examples (dict): contains 3 items
                - Input: pyg graph
                - Question: description of the input graph. Questions are sampled randomly.
                - Answer: the answer to the question.
    Returns:
        dict: 
            - input_ids:    input_ids for text data (q + a)
            - graph:        graph data (pyg)
            - labels:       labels for auto-regressive loss (a)
            - attn_mask:    attn_masks for text data (q + a)
    """
    inputs = []
    labels = []
    attn_masks = []
    for graph, question, answer in zip(examples["Input"], 
                                       examples["Question"], 
                                       examples["Answer"]):
        instruction_seq = f"<question>{question}</question><answer>{answer}</answer>"
        
        question = f"<question>{question}</question>"
        answer = f"<answer>{answer}</answer>"
        question_tokens = tokenizer(question, 
                                    truncation=True, 
                                    max_length=128,
                                    return_tensors="pt")
        answer_tokens = tokenizer(answer, 
                                  truncation=True,
                                  max_length=256, 
                                  return_tensors="pt")
        
        q_input_ids = question_tokens.input_ids[0]
        a_input_ids = answer_tokens.input_ids[0]
        q_attn_mask = question_tokens.attention_mask[0]
        a_attn_mask = answer_tokens.attention_mask[0]
        
        input_id = torch.cat([q_input_ids, a_input_ids], dim=0)
        inputs.append(input_id)
        
        attn_mask = torch.cat([q_attn_mask, a_attn_mask])
        attn_masks.append(attn_mask)

        label = torch.cat([
            torch.full_like(q_input_ids, -100),
            a_input_ids
        ])
        labels.append(label)

    # assert inputs.shape == labels.shape
    return {
            "input_ids":    inputs,
            "graph":        examples["Input"],
            "labels":       labels,
            "attn_masks":   attn_masks
    }



def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    return dist.get_rank()

#*================================================================================#
#*===================== This is main function ====================================#
#*================================================================================#




#* Init config parameterss


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

training_args = TrainingArguments(
    output_dir="../results/instruction_tuning_all_datasets",
    per_device_train_batch_size=64,
    num_train_epochs=5,
    logging_steps=10,
    logging_dir="../logs/LLM",
    gradient_checkpointing=True,  
    save_safetensors=False,
    save_steps=300,

)


#* Init model
model_name = "/home/dongrui/code/LLM4Brain/ckpt/BioGPT"
tokenizer = BioGptTokenizer.from_pretrained(model_name)
biogpt_model = BioGptForCausalLM.from_pretrained(model_name)

adapter = GCN(
    args, 
    input_dim=90,
    hidden_dim=256,
    output_dim=1024,
    act_fn="gelu"
)


model = BioGPTAdapter4InstructionTuning(
    args, adapter, biogpt_model, tokenizer)

#* Init dataset
graph_dataset = BrainDataset(root="{}/dataset/pretrain".format(args.root),
                             args=args)
json_dataset = load_json_dataset_pretrain(
    args,
    root_path="{}/{}".format(args.root, "text_data"),
    max_length=len(graph_dataset)
)
print(len(graph_dataset), len(json_dataset))
distill_dataset_list = load_dataset_distill_mm(
    args, graph_dataset, json_dataset)
distill_dataset = DatasetLM.from_list(distill_dataset_list)

processed_dataset = distill_dataset.map(preprocess_function_instruct,
                                        batched=True)

#* Start training.
trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset,
                tokenizer=tokenizer,
                data_collator=GTDataCollator_instruct(tokenizer),
            )
trainer.train()


#* ======================================================== *#
#* ================= Save finetuned model ================= *#
#* ======================================================== *#

base_path = f"{args.root}/ckpt/BioGPT"
save_path_ = f"{args.root}/ckpt/BioGPT_base_instruct_all_5"

model.biogpt_model.save_pretrained(save_path_)
tokenizer.save_pretrained(save_path_)
torch.save(adapter.state_dict(), f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth")
print("Save temp weights into {}".format(save_path_))
