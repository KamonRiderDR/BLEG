import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import unbatch

from torch_geometric.loader import DataLoader

from transformers import BioGptTokenizer, BioGptModel

from .batch import Batch

class FC(nn.Module):
    def __init__(self, input, output, dropout, norm=False):
        super(FC, self).__init__()
        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(input)
        self.fc = nn.Linear(input, output)
        self.act_fn = nn.GELU()
        self.norm = None if norm is False else nn.BatchNorm1d(output)
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.norm is None:
            return self.fc(x)
        else:
            return self.norm(self.fc(x))

class CLSToken(nn.Module):
    def __init__(self, hidden_size):
        super(CLSToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # 可学习的 [CLS] token

    def forward(self, batch_size):
        return self.cls_token.expand(batch_size, -1, -1)  # 形状: [batch_size, 1, hidden_size]


"""a simple two-layer FFN for transformation.
    input   -> hidden (256)
            -> output
"""
class BrainAdapter(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 dropout: float=0.1,
                 norm:str="batch",
                 act_fn="gelu",
                 args=None):
        super(BrainAdapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm0 = nn.BatchNorm1d(input_dim)
        self.gelu = nn.GELU() if act_fn == "gelu" else nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim) if norm == "layer" else nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.args = args

    def forward(self, data, type="data"):
        """_summary_

        Args:
            data (pyg): graph input

        Returns:
            _type_: _description_
        """
        # x = self.bn0(x)
        # if self.args:
        #     device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        #     x = data.fc_x.to(device)
        x = data.fc_x if type == "data" else data
        x = self.fc1(x)
        x = self.norm1(x)
        # x = self.relu(x)
        x = self.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout

        x = self.fc2(x)

        return x


class BrainLLaVA(nn.Module):
    def __init__(self,
                 args,
                 gnn_encoder,
                 adapter,
                 bio_gpt_model,
                 tokenizer,
                 training_type="pretrain"):
        """
            e.g.
            [text, Adapter(GNN(graph)), [cls] token] => [BioGPT] 
                                                     => classifier([cls])
                                                     => result
            Use `BioGPTModel` as model.

        Args:
            args (_type_): _description_
            gnn_encoder (_type_): _description_
            adapter (_type_): _description_
            bio_gpt_model (_type_): _description_
            tokenizer (_type_): _description_
        """
        super(BrainLLaVA, self).__init__()
        
        self.args = args
        self.gnn_encoder = gnn_encoder
        self.adapter = adapter
        self.bio_gpt_model = bio_gpt_model
        self.tokenizer = tokenizer
        self.cls_token = CLSToken(hidden_size=args.hidden_dim)
        self.classifier = nn.Linear(args.hidden_dim, args.num_classes)
        self.training_type = training_type

    def pretrain(self, graph_input, text_input):
        
        """ pretrain_stage

        Args:
            graph_input (data): pyg data
            text_input (string): text data
        
        Return:
            graph logits:   [b, hidden]
            text logits:    [b, hidden]
        """

        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        batch_size  = graph_input.batch.max().item() + 1
        
        for param in self.bio_gpt_model.parameters():
            param.require_grads = False

        graph_embeddings = self.gnn_encoder(graph_input)
        graph_embeddings = graph_embeddings.view(batch_size, 90, -1)

        text_inputs = self.tokenizer(text_input, 
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True).to(device)

        text_input_ids = text_inputs["input_ids"]
        text_attn_mask = text_inputs["attention_mask"]

        text_embeddings = self.bio_gpt_model(text_input_ids, 
                                             output_hidden_states=True).hidden_states[0]
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1) # [b, n_t, h]

        # [b, [graph + text], h]
        combined_input = torch.cat([graph_embeddings, text_embeddings], dim=1)
        text_attn_mask = text_attn_mask.repeat(batch_size, 1)
        combined_attn_mask = torch.cat([
            torch.ones((batch_size, 90)).to(text_attn_mask.device),
            text_attn_mask],
            dim=1)
        
        outputs = self.bio_gpt_model(inputs_embeds=combined_input,
                                     attention_mask=combined_attn_mask)
        
        text_embeddings_graph = outputs.last_hidden_state[:, 0:90, :]
        text_graph_mean = text_embeddings_graph.mean(dim=1)
        graph_mean = graph_embeddings.mean(dim=1)
        
        return graph_mean, text_graph_mean


    def forward(self, 
                graph_input, 
                text_input):
        """ skip.

        Args:
            graph_input (data): pyg format  [b * n_g, h_g]
            text_input (string):            [b, n_t, h_t]

        Returns:
            tensor: _description_
        """
        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')
        graph_input = Batch.from_data_list(graph_input).to(device)
        text_input = text_input[0]
        batch_size  = graph_input.batch.max().item() + 1


        #* frozen parameters
        if self.training_type == "pretrain":
            for param in self.bio_gpt_model.parameters():
                param.require_grads = False
            for param in self.gnn_encoder.parameters():
                param.require_grads = False

        #* get graph/text embeddings
        graph_embedding = self.gnn_encoder(graph_input)
        graph_embedding = self.adapter(graph_embedding)
        graph_embedding = graph_embedding.view(batch_size, 90, -1) # [b, 90, h]

        cls_tokens = self.cls_token(batch_size).to(device) # [b, 1, h]
        
        text_inputs = self.tokenizer(text_input, 
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True).to(device)
        text_input_ids = text_inputs["input_ids"]
        text_attn_mask = text_inputs["attention_mask"]
        text_embeddings = self.bio_gpt_model(text_input_ids, 
                                             output_hidden_states=True).hidden_states[0]
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1) # [b, n_t, h]

        # [batch, (graph + text + [cls]), hidden]
        combined_input = torch.cat([graph_embedding, text_embeddings, cls_tokens], dim=1)
        text_attn_mask = text_attn_mask.repeat(batch_size, 1)
        combined_attn_mask = torch.cat([
            torch.ones((batch_size, 90)).to(text_attn_mask.device),
            text_attn_mask,
            torch.ones((batch_size, 1)).to(text_attn_mask.device)],
            dim=1)

        outputs = self.bio_gpt_model(inputs_embeds=combined_input,
                                     attention_mask=combined_attn_mask)

        cls_embedding = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(cls_embedding)

        #* follow trainer template
        if self.training_type == "peft":
            # print(self.training_type)
            labels = graph_input.y
            loss = nn.CrossEntropyLoss()(logits, labels) 
            return {"loss": loss, "logits": logits}
        else:
            # print(self.training_type)
            return logits


class BioGPTWithAdapter(nn.Module):
    """ BioGPT + Adapter for classification.
        e.g.
            [Adapter(graph), text, [cls] token] => [BioGPT] => classification result.
        Use `BioGptForSequenceClassification` as GPT model.
    """
    def __init__(self, 
                 args,
                 adapter, 
                 biogpt_model,
                 tokenizer):
        super(BioGPTWithAdapter, self).__init__()
        
        self.args = args
        self.adapter = adapter
        self.biogpt_model = biogpt_model
        self.tokenizer = tokenizer
        self.cls_token = CLSToken(hidden_size=biogpt_model.config.hidden_size)
        self.norm_embedding = nn.BatchNorm1d(biogpt_model.config.hidden_size)
        # self.classifier = BrainAdapter(input_dim=biogpt_model.config.hidden_size,
        #                                hidden_dim=256,
        #                                output_dim=args.num_classes,
        #                                dropout=self.args.dropout)
        self.classifier = nn.Linear(biogpt_model.config.hidden_size, args.num_classes)


    def distill(self, sequence, text):
        """_summary_

        Args:
            sequence (_type_): _description_
            text (_type_): _description_

        Returns:
            logits (tensor):            [B, 2] for classification
            cls_embeddings (tensor):    [B, hidden_dim] for distill
        """
        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')

        # graph_input = Batch.from_data_list(sequence).to(device)
        graph_input = sequence
        graph_input.to(device)
        batch_size = graph_input.batch.max().item() + 1

        # graph_embedding = self.adapter(graph_input.fc_x)
        graph_embedding = self.adapter(graph_input)

        text_inputs = self.tokenizer(text, 
                                     return_tensors="pt",
                                     padding=True).to(device)
        intput_ids = text_inputs["input_ids"]
        text_attn_mask = text_inputs["attention_mask"]

        text_embedding = self.biogpt_model(intput_ids, 
                                            output_hidden_states=True).hidden_states[0]

        # expand text_inputs to [batch, n_t, hidden]
        graph_embedding = graph_embedding.view(batch_size, 90, -1)
        # text_embedding = text_embedding.repeat(batch_size, 1, 1)
        # text_attn_mask = text_attn_mask.repeat(batch_size, 1)

        cls_tokens = self.cls_token(batch_size).to(device) # [b, 1, h]

        combined_embedding = torch.cat([graph_embedding, text_embedding, cls_tokens], dim=1)
        combined_attn_mask = torch.cat([
            torch.ones((batch_size, 90)).to(text_attn_mask.device),
            text_attn_mask,
            torch.ones((batch_size, 1)).to(text_attn_mask.device)], 
            dim=1)

        outputs = self.biogpt_model(inputs_embeds=combined_embedding,
                                    attention_mask=combined_attn_mask)

        cls_embeddings = outputs.last_hidden_state[:, -1, :]
        # cls_embeddings = self.bn_embedding(cls_embeddings) # not use before
        logits = self.classifier(cls_embeddings)

        return logits, cls_embeddings

    def forward(self, sequence, text):
        """
        Args:
            sequence [pyg]: graph data as input
            text [string]: text data as input
        Returns:
            tensor: logits
        """
        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')

        # graph_input = Batch.from_data_list(sequence).to(device)
        graph_input = sequence
        batch_size = graph_input.batch.max().item() + 1

        # graph_embedding = self.adapter(graph_input.fc_x)
        graph_embedding = self.adapter(graph_input)

        text_inputs = self.tokenizer(text, 
                                     return_tensors="pt",
                                     padding=True).to(device)
        intput_ids = text_inputs["input_ids"]
        text_attn_mask = text_inputs["attention_mask"]

        text_embedding = self.biogpt_model(intput_ids, 
                                            output_hidden_states=True).hidden_states[0]

        # expand text_inputs to [batch, n_t, hidden]
        graph_embedding = graph_embedding.view(batch_size, 90, -1)
        # text_embedding = text_embedding.repeat(batch_size, 1, 1)
        # text_attn_mask = text_attn_mask.repeat(batch_size, 1)

        cls_tokens = self.cls_token(batch_size).to(device) # [b, 1, h]

        combined_embedding = torch.cat([graph_embedding, text_embedding, cls_tokens], dim=1)
        combined_attn_mask = torch.cat([
            torch.ones((batch_size, 90)).to(text_attn_mask.device),
            text_attn_mask,
            torch.ones((batch_size, 1)).to(text_attn_mask.device)], 
            dim=1)

        outputs = self.biogpt_model(inputs_embeds=combined_embedding,
                                    attention_mask=combined_attn_mask)

        # auto_regress_loss = outputs.loss

        cls_embeddings = outputs.last_hidden_state[:, -1, :]
        # cls_embeddings = self.bn_embedding(cls_embeddings) # not use before
        logits = self.classifier(cls_embeddings)

        # pooled_output = outputs.last_hidden_state[:, 0, :]
        # logits = self.classifier(pooled_output)

        # return logits, auto_regress_loss
        return logits, cls_embeddings