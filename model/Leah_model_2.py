import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.category_id_map import CATEGORY_ID_LIST
from transformers.models.bert.modeling_bert import BertModel, BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        bert_output_size = 768
        # tfidf 维度太大，暂时不加
        # tfidf_output_size = 128

        self.bert_fusion = TextConcatDenseSE(text_input_size=bert_output_size*2,
                                             hidden_size=args.bert_fusion_hidden_size,
                                             dropout=args.dropout,
                                             output_size=bert_output_size)

        self.text_vision_fusion = TextVisionConcatDenseSE(multimodal_hidden_size=args.vlad_hidden_size + bert_output_size,
                                                          hidden_size=args.bert_fusion_hidden_size,
                                                          dropout=args.dropout,
                                                          output_size=bert_output_size)

        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.act = nn.ReLU()
        self.MLP = nn.Linear(args.fc_size, args.Liner_hidden_size)
        self.vision_dense = nn.Linear(args.frame_embedding_size, args.frame_embedding_size)
        self.bert_config = self.bert.config
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        text_embedding = self.bert.embeddings(input_ids=inputs['text_input'],
                                              token_type_ids=inputs['text_token_type_ids']
                                              )
        vision_feature = self.act(self.vision_dense(inputs['frame_input']))
        vision_embedding = self.bert.embeddings(inputs_embeds=vision_feature)
        embedding = torch.cat([text_embedding, vision_embedding], dim=1)
        text_mask = inputs['text_mask']
        vision_mask = inputs['frame_mask']
        mask = torch.cat([text_mask, vision_mask], dim=1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        final_embedding = self.bert.encoder(embedding, attention_mask=mask).last_hidden_state.mean(1)
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()

        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class TextVisionConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, output_size, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_ouput = nn.Linear(hidden_size, output_size)
        # self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.fusion_ouput(embeddings)

        return embedding


class TextConcatDenseSE(nn.Module):
    def __init__(self, text_input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.text_fusion = nn.Linear(text_input_size, hidden_size)
        self.text_fusion_dropout = nn.Dropout(dropout)
        self.text_feature_output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        text_embedding = torch.cat(inputs, dim=1)
        text_embedding = self.text_fusion(text_embedding)
        text_embedding = self.text_fusion_dropout(text_embedding)
        text_embedding = self.text_feature_output(text_embedding)
        return text_embedding
