""" Trainer_sft is for supervised-finetuning. (GCN classification)

    Model:
        - BioGPT: model path
            - BioGPT_base_instruct_all_5
            - BioGPT_base_instruct_all
            - BioGPT_instruct_large (for larger GPT2)
        
        - GCN: model path
            - GCN_pretrain_all_datasets.pth
            - GCN_pretrain_all_datasets_5.pth


    Experiments:
        1.  k_fold:         k fold training
        2.  few_shot:       train ratio * len(dataset)
        3.  k_way:          k way n shot learning
        4.  inference:      generate description for private dataset (TODO)

    Datasets:
        1.  public dataset:  [HCP, ABIDE, ADHD, MDD]
        2.  private dataset: zhongdaxinxiang

    Note: For private dataset, need to firstly generate logits from tuning GPT2
"""

from transformers import (
    BioGptForCausalLM, 
    BioGptTokenizer,
    BioGptForSequenceClassification,
    BioGptModel,
    BioGptForTokenClassification,
    GPT2LMHeadModel,
    Trainer
    )
from transformers import pipeline, set_seed
from accelerate import infer_auto_device_map, load_checkpoint_in_model

from peft import PeftModel, LoraConfig, get_peft_model

import argparse
import os
import numpy as np
import json
from itertools import product
import time
from copy import deepcopy

import torch
from torch.utils.data import Subset
from torch_geometric.data import DenseDataLoader, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score

from utils.dataloader import DataLoader as myDataLoader
from utils.utils import *
from utils.load_data import *
from utils.args import *

from models.BrainLLaVA import BioGPTWithAdapter, BrainAdapter, BrainLLaVA
from models.model_pretrain import BioGPTAdapter4InstructionTuning
from models.GCN import GCN
from models.model_distill import GCN_d

from dataset_lm import (
    GraphTextDataset, 
    GraphTextDataCollator, 
    load_dataset_distill,
    load_dataset_distill_mm,
    load_dataset_private_mm,
    GTDataCollator_instruct,
    dict2graph
)

device = ""
seeds = [25, 50, 100, 125, 150, 175, 200, 225, 250, 275]


#* ============================ SC-FC eval PART ========================#
def sensitivity_specificity(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)

    if tp == 0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity, specificity

def eval_FCSC(args, model, loader):
    global device

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for i, data in enumerate(loader):
        data = Batch.from_data_list(data)

        graph_input = data
        data = graph_input

        data.to(device)
        labels = data.y

        # output,_ = model(graph_input, text_input)
        output,_ = model(graph_input)

        pred = output.data.argmax(dim=1)
        correct += torch.sum(pred == labels.view(-1)).item()
        test_loss += F.cross_entropy(output, labels.view(-1)).item() * output.shape[0]

        pred_num = pred.cpu().numpy()
        y_num = labels.cpu().numpy()
        for num in range(len(pred)):
            Y_pred.append(pred_num[num])
            Y_test.append(y_num[num])

    test_acc = correct / len(loader.dataset)
    test_loss = test_loss / len(loader.dataset)
    test_sen, test_spe = sensitivity_specificity(Y_test, Y_pred)
    test_f1 = f1_score(Y_test, Y_pred)
    test_auc = roc_auc_score(Y_test, Y_pred)
    
    torch.cuda.empty_cache()
    
    return test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, Y_test, Y_pred



#* ============================ prompt template functions ======================#

"""Premitive version. Generate same text prompt for each samples."""
def get_prompt():
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



tokenizer=None
def preprocess_function_instruct_private(examples):
    """ preprocess function for multi_modal instruction tuning. (zhongdaxinxiang)

    Args:
        examples (dict): contains 2 items
                - Input: pyg graph
                - Question: description of the input graph. Questions are sampled randomly.
    Returns:
        dict: 
            - input_ids:    input_ids for text data (q + 2)
            - graph:        graph data (pyg)
            - attn_mask:    attn_masks for text data (q + 2)
    """
    # print(examples)
    inputs = []
    labels = []
    attn_masks = []
    data_list_ = []
    for i, example in enumerate(examples):
        graph, question = example["Input"], example["Question"]
        graph = dict2graph([graph])[0]
        question = f"<question>{question}</question>"
        answer = "<answer></answer>"
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
        # inputs.append(input_id)
        
        attn_mask = torch.cat([q_attn_mask, a_attn_mask])
        # attn_masks.append(attn_mask)
        
        data_list_.append({
            "input_ids":    input_id,
            "graph":        graph,
            "attn_masks":   attn_mask
        })

    return data_list_








def pretrain(
    args,
    adapter,
    gnn_encoder,
    biogpt_model,
    tokenizer,
    dataloader
):
    """ Pretrain for llm-based model. keep llm frozen and prefix encoder/adapter training.
        Use GNN as encoder. BioGPT as LLM model.
        ITC Loss is used for pretraining.

    Args:
        args (_type_): _description_
        adapter (_type_): _description_
        biogpt_model (_type_): _description_
        tokenizer (_type_): _description_
        dataloader (_type_): _description_
        
    Return:
        best_epoch:
        ckpt_GNN
    """
    global device    
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    early_stop = args.patience 
    max_acc = 0.
    patience = 0
    best_epoch = 0


    model = BrainLLaVA(args, 
                       gnn_encoder,
                       adapter,
                       biogpt_model,
                       tokenizer)
    model.to(device)
    ckpt_gnn = None
    
    for param in model.bio_gpt_model.parameters():
        param.require_grads = False

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    for epoch in range(args.epochs_pretrain):
        model.train()
        train_loss = 0.
        train_correct = 0
        
        for idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            data.to(device)

            text_input = get_prompt()
            graph_logits, text_logits = model.pretrain(data, text_input)
            itc_loss = contrastive_loss(graph_logits, text_logits)
            itc_loss.backward()
            optimizer.step()

            out = model(data, text_input)
            pred = out.data.argmax(dim=1)
            train_loss += itc_loss.item() * out.shape[0]
            train_correct += torch.sum(pred == data.y.view(-1)).item()
            torch.cuda.empty_cache()

        scheduler.step()
        
        train_loss = train_loss / len(dataloader.dataset)
        train_acc = train_correct / len(dataloader.dataset)
        if epoch % 20 == 0:
            print("Epoch: {:03d} \t train_loss: {:.6f} \t train_acc: {:.4f}".format(
                epoch, train_loss, train_acc
            ))
            
        if train_acc > max_acc:
            max_acc = train_acc
            best_epoch = epoch
            patience = 0   
            ckpt_gnn = deepcopy(model.gnn_encoder.state_dict())
        else:
            patience += 1
        if patience >= early_stop:
            break

    torch.cuda.empty_cache()
    return best_epoch, ckpt_gnn


def train(
    args, 
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    optimizer, 
    scheduler,
    max_patience=None,
    i_fold=0
):
    """whole training process within [1 fold (times)]
    
    Args:
        skip

    Return:
        [Best epoch] && [ckpt]
    """ 
    global device
    
    max_acc = 0.
    patience = 0
    best_epoch = 0
    ckpt = None
    early_stop = args.patience if max_patience is None else max_patience

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.
        train_correct = 0

        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = 0.
            
            #* model forward
            graph_input, text_input = data
            data = graph_input
            data.to(device)

            out = model(graph_input, text_input)

            #* backward process
            loss_ce = F.cross_entropy(out, data.y.view(-1))
            loss = loss_ce
            loss.backward()
            optimizer.step()

            pred = out.data.argmax(dim=1)
            train_loss += loss.item() * out.shape[0]
            train_correct += torch.sum(pred == data.y.view(-1)).item()
        # scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # val && test
        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = test(model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = test(model, test_loader)
        if epoch % 10 == 0:
            print("Epoch: {:03d} \t train_loss: {:.6f} \t train_acc: {:.4f}\t val_loss: {:.4f} \t val_acc: {:.4f} \t test_acc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc, test_acc
            ))

        if val_acc > max_acc:
            max_acc = val_acc
            best_epoch = epoch
            patience = 0   
            ckpt = deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= early_stop:
            break

    torch.cuda.empty_cache()
    return best_epoch, ckpt

def distill_unknown(
                args,
                teacher_model,
                student_model,
                train_loader,
                val_loader,
                test_loader,
                optimizer_t,
                optimizer_s,
                max_patience=None,
            ):
    """whole training process within [1 fold (times)].
        Tip: for unknown dataset like zhongdaxinxiang.
    
    Args:
        skip

    Return:
        [Best epoch] && [ckpt]
    """ 
    global device
    
    max_acc = 0.
    patience = 0
    best_epoch = 0
    ckpt = None
    early_stop = args.patience if max_patience is None else max_patience

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    for epoch in range(args.epochs):
        teacher_model.eval()
        student_model.train()
        train_loss = 0.
        train_correct = 0

        for idx, data in enumerate(train_loader):
            optimizer_t.zero_grad()
            optimizer_s.zero_grad()
            loss = 0.
            
            #* model forward
            graph_input, text_input = data
            data = graph_input
            data.to(device)

            tea_out, tea_logits = teacher_model.distill(graph_input, text_input)
            stu_out, stu_logits = student_model(graph_input)

            #* backward process
            loss_ce_tea = F.cross_entropy(tea_out, data.y.view(-1))
            loss_ce_stu = F.cross_entropy(stu_out, data.y.view(-1))
            loss_ce = loss_ce_stu

            f_kd = nn.MSELoss()
            # loss_kd = KL_loss(tea_logits, stu_logits)
            loss_kd = f_kd(stu_logits, tea_logits)

            loss = loss_ce * (1 - args.lambda_kd) + loss_kd * args.lambda_kd

            loss.backward()
            optimizer_s.step()

            pred = stu_out.data.argmax(dim=1)
            train_loss += loss.item() * stu_out.shape[0]
            train_correct += torch.sum(pred == data.y.view(-1)).item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # val && test
        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = test(student_model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = test(student_model, test_loader)
        if epoch % 10 == 0:
            print("Epoch: {:03d} \t train_loss: {:.6f} \t train_acc: {:.4f}\t val_loss: {:.4f} \t val_acc: {:.4f} \t test_acc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc, test_acc
            ))

        if val_acc > max_acc:
            max_acc = val_acc
            best_epoch = epoch
            patience = 0   
            ckpt = deepcopy(student_model.state_dict())
        else:
            patience += 1
        if patience >= early_stop:
            break

    torch.cuda.empty_cache()
    return best_epoch, ckpt


def distill(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    max_patience=None,
):
    """whole training process within [1 fold (times)]
    
    Args:
        skip

    Return:
        [Best epoch] && [ckpt]
    """ 
    global device
    
    max_acc = 0.
    patience = 0
    best_epoch = 0
    ckpt = None
    early_stop = args.patience if max_patience is None else max_patience

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.
        train_correct = 0

        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = 0.
            
            #* model forward
            data = Batch.from_data_list(data)
            graph_input = data
            data.to(device)

            stu_out, stu_logits = model(graph_input)
            tea_logits = data.logits

            #* backward process
            loss_ce_stu = F.cross_entropy(stu_out, data.y.view(-1))
            loss_ce = loss_ce_stu

            f_kd = nn.MSELoss()
            loss_kd = f_kd(stu_logits, tea_logits)

            loss = loss_ce * (1 - args.lambda_kd) + loss_kd * args.lambda_kd

            loss.backward()
            optimizer.step()

            pred = stu_out.data.argmax(dim=1)
            train_loss += loss.item() * stu_out.shape[0]
            train_correct += torch.sum(pred == data.y.view(-1)).item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # val && test
        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = test(model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = test(model, test_loader)
        if epoch % 10 == 0:
            print("Epoch: {:03d} \t train_loss: {:.6f} \t train_acc: {:.4f}\t val_loss: {:.4f} \t val_acc: {:.4f} \t test_acc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc, test_acc
            ))

        if val_acc > max_acc:
            max_acc = val_acc
            best_epoch = epoch
            patience = 0   
            ckpt = deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= early_stop:
            break

    torch.cuda.empty_cache()
    return best_epoch, ckpt


@torch.no_grad()
def test(model, test_loader):
    """_summary_

    Args:
        test_loader (_type_): _description_
    Return:
        [test_acc, test_loss, 
         test_sen, test_spe, 
         test_f1, test_auc,  
         y_test, y_pred] from {test_loader}
    """
    model.eval()
    y_test = []
    y_pred = []
    correct = 0.
    test_loss = 0.

    for i, data in enumerate(test_loader):       
        data = Batch.from_data_list(data)
        
        graph_input = data
        data = graph_input
        data.to(device)

        out, _ = model(graph_input)

        pred = out.argmax(dim=1)
        correct += torch.sum(pred == data.y.view(-1)).item()
        test_loss += F.cross_entropy(out, data.y.view(-1)) * out.shape[0]

        pred_num = pred.cpu().numpy()
        y_num = data.y.cpu().numpy()
        for num in range(len(pred)):
            y_pred.append(pred_num[num])
            y_test.append(y_num[num])

    test_acc = correct / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    test_sen, test_spe = sensitivity_specificity(y_test, y_pred)
    test_f1=f1_score(y_test, y_pred)
    test_auc=roc_auc_score(y_test, y_pred)
    
    torch.cuda.empty_cache()
    
    return test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, y_test, y_pred


def load_LM_logits(args,
                   model,
                   dataset,
                   save_path):
    """load all logits from npy file 

    Args:
        args (args): _description_
        model (model): biogpt + GNN, frozen
        dataset (Dataset): all the dataset (?)
        save_path (str): path for saving LM logits

    Returns:
        logits: numpy, N * d (all datasets)
    """

    global device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    if os.path.exists(save_path):
        print("load exist logits from: {}".format(save_path))
        logits = np.load(save_path)
        return logits

    print("load new logits!")
    datalist_ = []
    for data in dataset:
        graph_input, text_input = data
        _, logits_ = model.distill(graph_input, text_input)
        datalist_.append(logits_.detach().cpu().numpy())
    np.save(save_path, datalist_)
    print(f"logits saved to {save_path}")
    return datalist_



def k_way_train(args):
    """few shot training. k way shot training"""

    #* init parameters
    global device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    # load LM model
    model_name = f"{args.root}/ckpt/BioGPT_base_instruct_all_5"
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    biogpt_model = BioGptModel.from_pretrained(model_name)

    
    #* ======== 1. Dataset Preparation =======*#
    ##* 1.1. pretrain dataset 
    #@ v2.0
    dataset_pretrain = BrainDataset(root="{}/dataset/pretrain".format(args.root, args.dataset),
                                    args=args,
                                    type="pretrain")
    text_dataset_pretrain = load_text_dataset_pretrain(
                                        args,
                                        root_path=f"{args.root}/text_data/",
                                        max_length=len(dataset_pretrain)
                                    )
    multimodal_dataset = GraphTextDataset(data=dataset_pretrain,
                                          text_input=text_dataset_pretrain,
                                          tokenizer=tokenizer)
    dataloader = DataLoader(multimodal_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate)

    ##* load in model
    # load in adapter
    adapter = GCN(
                args,
                input_dim=args.input_dim,
                hidden_dim=256,
                output_dim=args.hidden_dim,
                act_fn="gelu"
            )
    adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
    model = BioGPTWithAdapter(
                            args, 
                            adapter,
                            biogpt_model, 
                            tokenizer
                        )
    for param in model.biogpt_model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = False

    model.to(device)

    ##* 1.2. fine-tune dataset
    logits_pretrain = load_LM_logits(args,
                                     model,
                                     dataloader,
                                     save_path=f"{args.root}/save/pretrain_all_datasets_5.npy")

    finetune_args = {
        "graph":            dataset_pretrain,
        "logits":           logits_pretrain,
        "dataset":          args.dataset
    }
    dataset_ft = BrainDataset(
        root="{}/dataset/{}_5".format(args.root, args.dataset),
        args=args,
        type="finetune",
        finetune_args=finetune_args
    )
    multimodal_dataset = dataset_ft


    #* ======== 2. Training Loop =======*#
    acc_iter = []
    sen_iter = []
    spe_iter = []
    f1_iter = []

    for k in range(args.times):
        train_split, valid_split, test_split = few_shot_k_way_split(multimodal_dataset, 
                                                                    args.k_way_k,
                                                                    args.k_way_q,
                                                                    args.seed)

        #* 2.0. load dataset
        collate_fn = default_collate
        train_subset, valid_subset, test_subset = Subset(multimodal_dataset, train_split.tolist()), \
                                                  Subset(multimodal_dataset, valid_split.tolist()), \
                                                  Subset(multimodal_dataset, test_split.tolist())

        train_loader    = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader      = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader     = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        #* 2.1. load model
        #@ v2.1
        adapter = GCN(args,
                      input_dim=args.input_dim,
                      hidden_dim=256,
                      output_dim=args.hidden_dim,
                      act_fn="gelu")
        adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
        model = GCN_d(
                    args,
                    encoder=adapter,
                    output_dim=args.hidden_dim,
                    dropout=args.dropout,
                )
        for param in model.encoder.parameters():
            param.requires_grad = False

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        #* 2.2. training function
        best_epoch, ckpt = distill(
                                args,
                                model,
                                train_loader,
                                val_loader,
                                test_loader, 
                                optimizer
                            )

        #* 2.3. save results
        model.load_state_dict(ckpt)
        test_acc, test_loss, \
            test_sen, test_spe, test_f1, test_auc,\
            y, pred = eval_FCSC(args, model, test_loader)
        torch.cuda.empty_cache()

        acc_iter.append(test_acc)
        sen_iter.append(test_sen)
        spe_iter.append(test_spe)
        f1_iter.append(test_f1)
        
        print('[{}] times k-way shot test set results, best_epoch = {:03d}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(
                k, best_epoch, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))

        logging_result(args, 
                       {
                            "fold": k, 
                            "best_epoch": best_epoch,
                            "test_acc": test_acc
                        }, 
                       args.result_path)

    print('---------------------------------------------------------------\n \
    [{}] times test set results, accuracy = {:.2f}±{:.2f}'.format(
    k, np.mean(acc_iter)*100, np.std(acc_iter)*100))
    
    # logging 10-times result
    result_path = args.result_path
    result_path_csv = f"{args.root}/logs/csv/{args.dataset}_{args.training_type}_{args.exp_type}.csv"
    result_dict = {
        # log hyper-parameters
        'seed'              : args.seed,
        'dropout'           : args.dropout,
        'lr'                : args.lr,
        "decay"             : args.weight_decay,   
        'lambda_kd'         : args.lambda_kd,
        'batch_size'        : args.batch_size,
        'k-way k'           : args.k_way_k,
        'k-way q'           : args.k_way_q,
        # log results
        'acc'               : "{:.2f} ± {:.2f}".format(np.mean(acc_iter) * 100, np.std(acc_iter) * 100),
        'sen'               : "{:.2f} ± {:.2f}".format(np.mean(sen_iter) * 100, np.std(sen_iter) * 100),
        'spe'               : "{:.2f} ± {:.2f}".format(np.mean(spe_iter) * 100, np.std(spe_iter) * 100),
        'f1'                : "{:.2f} ± {:.2f}".format(np.mean(f1_iter)  * 100, np.std(f1_iter) * 100)
    }
    logging_result(args, result_dict, result_path)
    save_as_csv(args, result_path_csv, result_dict)



def few_shot_train(args):
    """few shot training. train_ratio * len(dataset)"""
    #* init parameters
    global device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    # load LM model
    model_name = f"{args.root}/ckpt/BioGPT_base_instruct_all_5"
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    biogpt_model = BioGptModel.from_pretrained(model_name)

    
    #* ======== 1. Dataset Preparation =======*#
    ##* 1.1. pretrain dataset 
    #@ v2.0
    dataset_pretrain = BrainDataset(root="{}/dataset/pretrain".format(args.root, args.dataset),
                                    args=args,
                                    type="pretrain")
    text_dataset_pretrain = load_text_dataset_pretrain(
                                        args,
                                        root_path=f"{args.root}/text_data/",
                                        max_length=len(dataset_pretrain)
                                    )
    multimodal_dataset = GraphTextDataset(data=dataset_pretrain,
                                          text_input=text_dataset_pretrain,
                                          tokenizer=tokenizer)
    dataloader = DataLoader(multimodal_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate)

    ##* load in model
    # load in adapter
    adapter = GCN(
                args,
                input_dim=args.input_dim,
                hidden_dim=256,
                output_dim=args.hidden_dim,
                act_fn="gelu"
            )
    adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
    model = BioGPTWithAdapter(
                            args, 
                            adapter,
                            biogpt_model, 
                            tokenizer
                        )
    for param in model.biogpt_model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = False

    model.to(device)

    ##* 1.2. fine-tune dataset
    logits_pretrain = load_LM_logits(args,
                                     model,
                                     dataloader,
                                     save_path=f"{args.root}/save/pretrain_all_datasets_5.npy")

    finetune_args = {
        "graph":            dataset_pretrain,
        "logits":           logits_pretrain,
        "dataset":          args.dataset
    }
    dataset_ft = BrainDataset(
        root="{}/dataset/{}_5".format(args.root, args.dataset),
        args=args,
        type="finetune",
        finetune_args=finetune_args
    )
    multimodal_dataset = dataset_ft


    #* ======== 2. Training Loop =======*#
    acc_iter = []
    sen_iter = []
    spe_iter = []
    f1_iter = []

    for k in range(args.times):
        train_split, valid_split, test_split = few_shot_split(multimodal_dataset, 
                                                              args.few_shot_hyper,
                                                              args.seed)

        #* 2.0. load dataset
        collate_fn = default_collate
        train_subset, valid_subset, test_subset = Subset(multimodal_dataset, train_split.tolist()), \
                                                  Subset(multimodal_dataset, valid_split.tolist()), \
                                                  Subset(multimodal_dataset, test_split.tolist())

        train_loader    = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader      = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader     = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        #* 2.1. load model
        #@ v2.1
        adapter = GCN(args,
                      input_dim=args.input_dim,
                      hidden_dim=256,
                      output_dim=args.hidden_dim,
                      act_fn="gelu")
        adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
        model = GCN_d(
                    args,
                    encoder=adapter,
                    output_dim=args.hidden_dim,
                    dropout=args.dropout,
                )
        for param in model.encoder.parameters():
            param.requires_grad = False

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        #* 2.2. training function
        best_epoch, ckpt = distill(
                                args,
                                model,
                                train_loader,
                                val_loader,
                                test_loader, 
                                optimizer
                            )

        #* 2.3. save results
        model.load_state_dict(ckpt)
        test_acc, test_loss, \
            test_sen, test_spe, test_f1, test_auc,\
            y, pred = eval_FCSC(args, model, test_loader)
        torch.cuda.empty_cache()

        acc_iter.append(test_acc)
        sen_iter.append(test_sen)
        spe_iter.append(test_spe)
        f1_iter.append(test_f1)
        
        print('[{}] times few shot test set results, best_epoch = {:03d}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(
                k, best_epoch, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))

        logging_result(args, 
                       {
                            "fold": k, 
                            "best_epoch": best_epoch,
                            "test_acc": test_acc
                        }, 
                       args.result_path)

    print('---------------------------------------------------------------\n \
    [{}] times test set results, accuracy = {:.2f}±{:.2f}'.format(
    k, np.mean(acc_iter)*100, np.std(acc_iter)*100))
    
    # logging 10-fold result
    result_path = args.result_path
    result_path_csv = f"{args.root}/logs/csv/{args.dataset}_{args.training_type}_{args.exp_type}.csv"
    result_dict = {
        # log hyper-parameters
        'seed'              : args.seed,
        'dropout'           : args.dropout,
        'lr'                : args.lr,
        "decay"             : args.weight_decay,   
        'lambda_kd'         : args.lambda_kd,
        'batch_size'        : args.batch_size,
        'few_shot_hyper'    : args.few_shot_hyper,
        # log results
        'acc'               : "{:.2f} ± {:.2f}".format(np.mean(acc_iter) * 100, np.std(acc_iter) * 100),
        'sen'               : "{:.2f} ± {:.2f}".format(np.mean(sen_iter) * 100, np.std(sen_iter) * 100),
        'spe'               : "{:.2f} ± {:.2f}".format(np.mean(spe_iter) * 100, np.std(spe_iter) * 100),
        'f1'                : "{:.2f} ± {:.2f}".format(np.mean(f1_iter)  * 100, np.std(f1_iter)*100)
    }
    logging_result(args, result_dict, result_path)
    save_as_csv(args, result_path_csv, result_dict)



def t_times_train(args):
    """Traditional training methods. T times training for traditional 10-fold training."""

    #* init parameters
    global device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)

    # load LM model
    # model_name = "/home/dongrui/code/LLM4Brain/ckpt/BioGPT"
    # model_name = "/home/dongrui/code/LLM4Brain/ckpt/BioGPT_merged_distill"
    # model_name = f"{args.root}/ckpt/BioGPT_instruct_large"
    model_name = f"{args.root}/ckpt/BioGPT_base_instruct_all_5"
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    biogpt_model = BioGptModel.from_pretrained(model_name)


    #* ======== 1. Dataset Preparation =======*#
    ##* 1.1. pretrain dataset 

    #@ v2.0
    graph_dataset_pretrain = BrainDataset(
                                    root="{}/dataset/pretrain".format(args.root),
                                    args=args,
                                    type="pretrain"
                                    )
    text_dataset_pretrain = load_text_dataset_pretrain(
                                        args,
                                        root_path=f"{args.root}/text_data/",
                                        max_length=len(graph_dataset_pretrain)
                                    )
    multimodal_dataset = GraphTextDataset(data=graph_dataset_pretrain,
                                          text_input=text_dataset_pretrain,
                                          tokenizer=tokenizer)
    dataloader = DataLoader(multimodal_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate)

    ##* load in model
    # load in adapter
    adapter = GCN(
                args,
                input_dim=args.input_dim,
                hidden_dim=256,
                output_dim=args.hidden_dim,
                act_fn="gelu"
            )
    adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
    model = BioGPTWithAdapter(
                            args, 
                            adapter,
                            biogpt_model, 
                            tokenizer
                        )
    for param in model.biogpt_model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = False

    model.to(device)

    ##* 1.2. fine-tune dataset
    logits_pretrain = load_LM_logits(args,
                                     model,
                                     dataloader,
                                     save_path=f"{args.root}/save/pretrain_all_datasets_5.npy")

    finetune_args = {
        "graph":            graph_dataset_pretrain,
        "logits":           logits_pretrain,
        "dataset":          args.dataset
    }
    dataset_ft = BrainDataset(
        root="{}/dataset/{}_5".format(args.root, args.dataset),
        args=args,
        type="finetune",
        finetune_args=finetune_args
    )
    multimodal_dataset = dataset_ft


    #* ======== 2. Training Loop =======*#

    acc = []
    sen = []
    spe = []
    f1 = []
    
    #* t times loop
    for i in range(args.times):
        setup_seed(args.seed)
        
        acc_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []

        #* k fold loop
        for k, (train_split, valid_split, test_split) \
                in enumerate(cross_validate_multimodal(args.folds, multimodal_dataset)):
            #* 2.0. load dataset
            collate_fn = default_collate
            train_subset, valid_subset, test_subset = Subset(multimodal_dataset, train_split.tolist()), \
                                                    Subset(multimodal_dataset, valid_split.tolist()), \
                                                    Subset(multimodal_dataset, test_split.tolist())

            train_loader    = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
            val_loader      = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader     = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

            #* 2.1. load model
            #@ v2.1
            adapter = GCN(args,
                        input_dim=args.input_dim,
                        hidden_dim=256,
                        output_dim=args.hidden_dim,
                        act_fn="gelu")
            adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
            model = GCN_d(
                        args,
                        encoder=adapter,
                        output_dim=args.hidden_dim,
                        dropout=args.dropout,
                    )
            for param in model.encoder.parameters():
                param.requires_grad = False

            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

            #* 2.2. training function
            best_epoch, ckpt = distill(
                                    args,
                                    model,
                                    train_loader,
                                    val_loader,
                                    test_loader, 
                                    optimizer
                                )

            #* 2.3. save results
            model.load_state_dict(ckpt)
            test_acc, test_loss, \
                test_sen, test_spe, test_f1, test_auc,\
                y, pred = eval_FCSC(args, model, test_loader)
            torch.cuda.empty_cache()

            acc_iter.append(test_acc)
            sen_iter.append(test_sen)
            spe_iter.append(test_spe)
            f1_iter.append(test_f1)
            
            print('[{}] fold test set results, best_epoch = {:03d}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                    'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(
                    k, best_epoch, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))

            logging_result(args, 
                        {
                                "fold": k, 
                                "best_epoch": best_epoch,
                                "test_acc": test_acc
                            }, 
                        args.result_path)

        print('---------------------------------------------------------------\n \
        [{}] times test set results, accuracy = {:.2f}±{:.2f}'.format(
        i, np.mean(acc_iter)*100, np.std(acc_iter)*100))

        #* logging 1 time 10-fold result
        result_path = args.result_path
        result_path_csv = f"{args.root}/logs/csv/{args.dataset}_{args.training_type}_{args.exp_type}.csv"
        result_dict = {
            # log hyper-parameters
            'seed'          : args.seed,
            'dropout'       : args.dropout,
            'lr'            : args.lr,
            "decay"         : args.weight_decay,   
            'lambda_kd'     : args.lambda_kd,
            'batch_size'    : args.batch_size,
            # log results
            'acc'           : "{:.2f} ± {:.2f}".format(np.mean(acc_iter) * 100,
                                                    np.std(acc_iter) * 100),
            'sen'           : "{:.2f} ± {:.2f}".format(np.mean(sen_iter) * 100,
                                                    np.std(sen_iter) * 100),
            'spe'           : "{:.2f} ± {:.2f}".format(np.mean(spe_iter) * 100,
                                                    np.std(spe_iter) * 100),
            'f1'            : "{:.2f} ± {:.2f}".format(np.mean(f1_iter)*100,
                                                    np.std(f1_iter)*100)
        }
        logging_result(args, result_dict, result_path)
        save_as_csv(args, result_path_csv, result_dict)


        acc.append(np.mean(acc_iter))
        sen.append(np.mean(sen_iter))
        spe.append(np.mean(spe_iter))
        f1.append(np.mean(f1_iter))


    #* logging t times 10-fold result
    result_path = args.result_path
    result_path_csv = f"{args.root}/logs/csv/{args.dataset}_{args.training_type}_{args.exp_type}.csv"
    result_dict = {
        # log hyper-parameters
        'seed'          : args.seed,
        'dropout'       : args.dropout,
        'lr'            : args.lr,
        "decay"         : args.weight_decay,   
        'lambda_kd'     : args.lambda_kd,
        'batch_size'    : args.batch_size,
        # log results
        'acc'           : "{:.2f} ± {:.2f}".format(np.mean(acc) * 100,
                                                   np.std(acc) * 100),
        'sen'           : "{:.2f} ± {:.2f}".format(np.mean(sen) * 100,
                                                   np.std(sen) * 100),
        'spe'           : "{:.2f} ± {:.2f}".format(np.mean(spe) * 100,
                                                   np.std(spe) * 100),
        'f1'            : "{:.2f} ± {:.2f}".format(np.mean(f1) * 100,
                                                   np.std(f1) * 100)
    }
    logging_result(args, result_dict, result_path)
    save_as_csv(args, result_path_csv, result_dict)


def decode_dataset(args):
    global device, tokenizer
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    # load LM model
    model_name = f"{args.root}/ckpt/BioGPT_base_instruct_all_5"
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    biogpt_model = GPT2LMHeadModel.from_pretrained(model_name)

    # load in dataset (if needed)
    dataset = BrainDataset(
        root="{}/dataset/{}_compare".format(args.root, args.dataset),
        args=args,
        type="pretrain",
    )

    dataset_mm = load_dataset_private_mm(dataset)

    processed_dataset = preprocess_function_instruct_private(dataset_mm)


    ##* load in model
    adapter = GCN(
                args,
                input_dim=args.input_dim,
                hidden_dim=256,
                output_dim=args.hidden_dim,
                act_fn="gelu"
            )
    adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
    model = BioGPTAdapter4InstructionTuning(
                                        args,
                                        adapter,
                                        biogpt_model,
                                        tokenizer
                                    )
    for param in model.biogpt_model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = False

    model.to(device)
    result_path = f"{args.root}/logs/generate/{args.dataset}.txt"
    
    for idx, data in enumerate(processed_dataset):
        # print(len(data))
        
        input_ids, graph, attn_masks = data["input_ids"], data["graph"], data["attn_masks"]

        input_ids.to(device)
        graph.to(device)
        attn_masks.to(device)
        
        output = model.generate(input_ids, graph, attn_masks)

        with open(result_path, "a") as f:
            f.write(f"===================[{idx+1}]=====================\n")
            f.write(f"output: {output}\n")
            f.write("\n\n")

        print(output)
    


def t_times_train_private(args):
    """training for private dataset. Which means there is no text description.
            1. load graph dataset
            2. generate text dataset
            3. generate text logits
            4. save graph dataset

    Args:
        args (_type_): _description_

    Raises:
        ValueError: _description_
    """
    global device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    # load LM model
    model_name = f"{args.root}/ckpt/BioGPT_base_instruct_all_5"
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    biogpt_model = BioGptModel.from_pretrained(model_name)

    # load in dataset (if needed)
    dataset = BrainDataset(
        f"{args.root}/dataset/{args.dataset}_temp",
        args=args,
        type="finetune",
        finetune_args={"logits": None}
    )
    dataset_text = load_text_dataset_private(args,
                                             dataset)
    multimodal_dataset = GraphTextDataset(data=dataset,
                                          text_input=dataset_text,
                                          tokenizer=tokenizer)
    dataloader = DataLoader(multimodal_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate)

    ##* load in model
    adapter = GCN(
                args,
                input_dim=args.input_dim,
                hidden_dim=256,
                output_dim=args.hidden_dim,
                act_fn="gelu"
            )
    adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
    model = BioGPTWithAdapter(
                            args, 
                            adapter,
                            biogpt_model, 
                            tokenizer
                        )
    for param in model.biogpt_model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = False

    model.to(device)
    
    logits_pretrain = load_LM_logits(args,
                                     model,
                                     dataloader,
                                     save_path=f"{args.root}/save/{args.dataset}_5.npy")
    
    dataset_ft = BrainDataset(
        root="{}/dataset/{}_5".format(args.root, args.dataset),
        args=args,
        type="finetune",
        finetune_args={"logits": logits_pretrain}
    )
    multimodal_dataset = dataset_ft


    #* ======== 2. Training Loop =======*#
    acc_iter = []
    sen_iter = []
    spe_iter = []
    f1_iter = []
    auc_iter = []

    for k, (train_split, valid_split, test_split) \
            in enumerate(cross_validate_multimodal(args.folds, multimodal_dataset)):
        #* 2.0. load dataset
        collate_fn = default_collate
        train_subset, valid_subset, test_subset = Subset(multimodal_dataset, train_split.tolist()), \
                                                  Subset(multimodal_dataset, valid_split.tolist()), \
                                                  Subset(multimodal_dataset, test_split.tolist())

        train_loader    = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader      = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader     = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        #* 2.1. load model
        #@ v2.1
        adapter = GCN(args,
                      input_dim=args.input_dim,
                      hidden_dim=256,
                      output_dim=args.hidden_dim,
                      act_fn="gelu")
        adapter.load_state_dict(torch.load(f"{args.root}/ckpt/GCN_pretrain_all_datasets_5.pth"))
        model = GCN_d(
                    args,
                    encoder=adapter,
                    output_dim=args.hidden_dim,
                    dropout=args.dropout,
                )
        for param in model.encoder.parameters():
            param.requires_grad = False

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        #* 2.2. training function
        best_epoch, ckpt = distill(
                                args,
                                model,
                                train_loader,
                                val_loader,
                                test_loader, 
                                optimizer
                            )

        #* 2.3. save results
        model.load_state_dict(ckpt)
        test_acc, test_loss, \
            test_sen, test_spe, test_f1, test_auc,\
            y, pred = eval_FCSC(args, model, test_loader)
        torch.cuda.empty_cache()

        acc_iter.append(test_acc)
        sen_iter.append(test_sen)
        spe_iter.append(test_spe)
        f1_iter.append(test_f1)
        auc_iter.append(test_auc)
        
        print('[{}] fold test set results, best_epoch = {:03d}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(
                k, best_epoch, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))

        logging_result(args, 
                       {
                            "fold": k, 
                            "best_epoch": best_epoch,
                            "test_acc": test_acc
                        }, 
                       args.result_path)

    print('---------------------------------------------------------------\n \
    [{}] times test set results, accuracy = {:.2f}±{:.2f}'.format(
    1, np.mean(acc_iter)*100, np.std(acc_iter)*100))
    
    # logging 10-fold result
    result_path = args.result_path
    result_path_csv = f"{args.root}/logs/csv/{args.dataset}_{args.training_type}_{args.exp_type}.csv"
    result_dict = {
        # log hyper-parameters
        'seed'          : args.seed,
        'dropout'       : args.dropout,
        'lr'            : args.lr,
        "decay"         : args.weight_decay,   
        'lambda_kd'     : args.lambda_kd,
        'batch_size'    : args.batch_size,
        # log results
        'acc'           : "{:.2f} ± {:.2f}".format(np.mean(acc_iter) * 100,
                                                   np.std(acc_iter) * 100),
        'sen'           : "{:.2f} ± {:.2f}".format(np.mean(sen_iter) * 100,
                                                   np.std(sen_iter) * 100),
        'spe'           : "{:.2f} ± {:.2f}".format(np.mean(spe_iter) * 100,
                                                   np.std(spe_iter) * 100),
        'f1'            : "{:.2f} ± {:.2f}".format(np.mean(f1_iter) * 100,
                                                   np.std(f1_iter) * 100),
        'auc'           : "{:.2f} ± {:.2f}".format(np.mean(auc_iter) * 100,
                                                   np.std(auc_iter) * 100)
    }
    logging_result(args, result_dict, result_path)
    save_as_csv(args, result_path_csv, result_dict)



def main(args):
    if args.exp_type == "k_fold":
        # k-fold training
        param_names = ["seed", "lr", "lambda_kd", "dropout", "batch_size"]
        params = [
            seeds,
            [0.0005, 0.0001],
            [0.3, 0.4, 0.5, 0.6],
            [0.0, 0.1, 0.3, 0.5],
            [128, 64, 32]
        ]
        
        if args.dataset in args.dataset_pool:
            grid_search(args, param_names, params, t_times_train)
        else:
            grid_search(args, param_names, params, t_times_train_private)
    elif args.exp_type == "few_shot":    
        # few-shot-training
        train_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        json_path = f"{args.root}/config/best_hyper_{args.dataset}.json"
        json_dict = load_json_param(json_path)
        for key, val in json_dict.items():
            setattr(args, key, val)
            
        param_names = ["few_shot_hyper"]
        params = [train_ratio]

        grid_search(args, param_names, params, few_shot_train)

    elif args.exp_type == "k_way":
        # k-way training
        json_path = f"{args.root}/config/best_hyper_{args.dataset}.json"
        json_dict = load_json_param(json_path)
        for key, val in json_dict.items():
            setattr(args, key, val)

        # k_way_train(args)
        k_s = [1, 2, 5, 10]
        q_s = [50, 100, -1]
        param_names = ["k_way_k", "k_way_q", "seed"]
        params = [
            k_s,
            q_s,
            seeds
        ]
        grid_search(args, param_names, params, k_way_train)
    else:
        raise ValueError("Invalid experiment type.")



if __name__ == '__main__':
    args = get_args()
    main(args)

