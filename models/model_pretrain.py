import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import unbatch
from torch_geometric.loader import DataLoader

from transformers import (
    BioGptTokenizer, 
    BioGptModel, 
    BioGptPreTrainedModel
)

from .batch import Batch


class BioGPTAdapter4InstructionTuning(nn.Module):
    """ BioGPT + Adapter for instruction tuning.
        e.g.
            [Adapter(graph), text, [cls] token] => [BioGPT] => classification result.
        Use `BioGptForSequenceClassification` as GPT model.
    """
    def __init__(self, 
                 args,
                 adapter, 
                 biogpt_model,
                 tokenizer):
        super(BioGPTAdapter4InstructionTuning, self).__init__()

        self.args = args
        self.adapter = adapter
        self.biogpt_model = biogpt_model
        self.tokenizer = tokenizer
        self.pre_norm = nn.BatchNorm1d(biogpt_model.config.hidden_size)

    # for larger model, save GPU memory
    def gradient_checkpointing_enable(self, **kwargs):
        self.biogpt_model.gradient_checkpointing_enable(**kwargs)


    def generate(self, 
                 input_ids,
                 graph,
                 attn_masks=None):
        device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')

        graph_input = graph.to(device)
        input_ids = input_ids.to(device)
        attn_masks = attn_masks.to(device)
        
        batch_size = 1
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # 变成 (1, seq_len)
        attn_masks = attn_masks.unsqueeze(0)
        
        # print(input_ids.device)
        
        text_embedding = self.biogpt_model(input_ids, 
                                           output_hidden_states=True).hidden_states[0]
        graph_embedding = self.adapter(graph_input)
        graph_embedding = graph_embedding.view(batch_size, 90, -1)

        text_attn_mask = attn_masks
        combined_embedding = torch.cat([graph_embedding, text_embedding], dim=1)
        combined_attn_mask = torch.cat([
            torch.ones((batch_size, 90)).to(text_attn_mask.device),
            text_attn_mask], 
            dim=1)

        # outputs = self.biogpt_model(inputs_embeds=combined_embedding,
        #                             attention_mask=combined_attn_mask)
        
        
        output_ids = self.biogpt_model.generate(
            inputs_embeds=combined_embedding,
            max_length=64,
            num_return_sequences=1,  # 生成几个序列
            no_repeat_ngram_size=2,  # 避免重复的n-gram
            early_stopping=True
        )

        decode_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        torch.cuda.empty_cache()
        return decode_text


    def forward(self, 
                input_ids,
                graph, 
                attn_masks=None,
                labels=None):
        """ Instruction tuning for adapter (GNN) and LM (BioGPTForCausualLM)

        Args:
            graph (pyg): _description_
            input_ids (_type_): _description_
            attn_masks (_type_, optional): _description_. Defaults to None.
            labels (_type_, optional): _description_. Defaults to None.

        Returns:
            Output: Output from GPT2
        """
        # device = torch.device('cuda:{}'.format(self.args.device) if torch.cuda.is_available() else 'cpu')

        # print(graph)
        # graph_input = Batch.from_data_list(graph).to(device)
        # graph_input = graph.to(device)
        # input_ids.to(device)
        # attn_masks.to(device)
        # assert graph.x.device == input_ids.device

        graph_input = graph
        batch_size = graph_input.batch.max().item() + 1

        text_embedding = self.biogpt_model(input_ids, 
                                           output_hidden_states=True).hidden_states[0]
        graph_embedding = self.adapter(graph_input)
        # graph_embedding = self.pre_norm(graph_embedding)
        graph_embedding = graph_embedding.view(batch_size, 90, -1)


        # expand text_inputs to [batch, n_t, hidden]
        # text_embedding = text_embedding.repeat(batch_size, 1, 1)
        # text_attn_mask = text_attn_mask.repeat(batch_size, 1)
        
        # print(graph_embedding.shape, text_embedding.shape)
        text_attn_mask = attn_masks
        combined_embedding = torch.cat([graph_embedding, text_embedding], dim=1)
        combined_attn_mask = torch.cat([
            torch.ones((batch_size, 90)).to(text_attn_mask.device),
            text_attn_mask], 
            dim=1)
        combined_labels = torch.cat([
            torch.full_like(torch.ones((batch_size, 90)), -100, dtype=labels.dtype).to(labels.device),
            labels], 
            dim=1)
        outputs = self.biogpt_model(inputs_embeds=combined_embedding,
                                    attention_mask=combined_attn_mask,
                                    labels=combined_labels)

        torch.cuda.empty_cache()
        return outputs