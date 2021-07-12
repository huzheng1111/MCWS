import math
import copy
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertConfig
from pytorch_pretrained_bert.modeling import BertEncoder
from torchcrf import CRF

class MCWS(BertPreTrainedModel):
    def __init__(self, config, audio_config, args, first_device, second_device):
        super().__init__(config)
        self.first_device=first_device
        self.second_device=second_device

        self.bert = BertModel(config).to(self.first_device)
        self.audio_encoder = BertEncoder(audio_config).to(self.first_device)
        self.classifier = nn.Linear(config.hidden_size + audio_config.hidden_size, 4).to(self.first_device)
        self.text_lstm_cell = nn.LSTMCell(config.hidden_size, config.hidden_size).to(self.first_device)
        self.audio_lstm_cell = nn.LSTMCell(audio_config.hidden_size, audio_config.hidden_size).to(self.first_device)
        self.crf = CRF(4).to(self.first_device)
        self_attention_layer = SelfAttention(1, 1 , args.attention_probs_dropout_prob)
        self.self_attention_layers = nn.ModuleList(
            [copy.deepcopy(self_attention_layer) for _ in range(args.num_multi_attention_layers)]).to(self.second_device)

    def forward(self, input_ids, attention_mask, token_type_ids, audio_feature, audio_mask, labels=None):

        Batch_size, sent_len, audio_hidden_size = audio_feature.size()
        text_sequence_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)
        text_sequence_output = text_sequence_output[:, 1:, :]
        _, _, text_hidden_size = text_sequence_output.size()

        text_hs_forward = torch.zeros((Batch_size, text_hidden_size))
        text_cs_forward = torch.zeros((Batch_size, text_hidden_size))
        text_hs_forward = torch.nn.init.xavier_uniform(text_hs_forward).to(self.first_device)
        text_cs_forward = torch.nn.init.xavier_uniform(text_cs_forward).to(self.first_device)

        audio_hs_forward = torch.zeros((Batch_size, audio_hidden_size))
        audio_cs_forward = torch.zeros((Batch_size, audio_hidden_size))
        audio_hs_forward = torch.nn.init.xavier_uniform(audio_hs_forward).to(self.first_device)
        audio_cs_forward = torch.nn.init.xavier_uniform(audio_cs_forward).to(self.first_device)

        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)

        extended_audio_mask = extended_audio_mask.to(dtype=torch.float32)
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0

        audio_sequence_output = self.audio_encoder(audio_feature, extended_audio_mask, output_all_encoded_layers=False)

        audio_sequence_output = audio_sequence_output[0]

        output = torch.zeros((Batch_size, sent_len, text_hidden_size + audio_hidden_size)).to(self.first_device)
        for i in range(sent_len):
            text_hs_forward, text_cs_forward = self.text_lstm_cell(text_sequence_output[:, i, :],
                                                                   (text_hs_forward, text_cs_forward))

            audio_hs_forward, audio_cs_forward = self.audio_lstm_cell(audio_sequence_output[:, i, :],
                                                                      (audio_hs_forward, audio_cs_forward))

            audio_text = torch.cat([text_hs_forward, audio_hs_forward], -1).to(self.second_device)

            hidden_state = torch.zeros(Batch_size, text_hidden_size + audio_hidden_size, 1).to(self.second_device)
            audio_text = audio_text.unsqueeze(-1)
            for j, layer_module in enumerate(self.self_attention_layers):
                hidden_state += layer_module(audio_text)
            hidden_state += audio_text
            text_hs_forward = hidden_state[:, :text_hidden_size, 0].to(self.first_device)
            audio_hs_forward = hidden_state[:, text_hidden_size:, 0].to(self.first_device)
            output[:, i, :] = (hidden_state[:, :, 0]).to(self.first_device)

        logits = self.classifier(output)
        logits = logits.transpose(1, 0)

        attention_mask = attention_mask[:, 1:].transpose(1, 0)
        attention_mask = attention_mask.type(torch.uint8)
        tag_seq = self.crf.decode(logits, attention_mask)
        if labels is not None:
            labels = labels.transpose(1, 0)
            total_loss = -self.crf(logits, labels, attention_mask)

        return total_loss, tag_seq

class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob,
                 output_attentions=False, keep_multihead_output=False):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_states.size(-1),)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.output_attentions:
            return attention_probs, context_layer
        return context_layer
