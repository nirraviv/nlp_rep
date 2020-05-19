from torch import nn
from .crf import CRF
from transformers import BertPreTrainedModel


class NERModel(nn.Module):
    def __init__(self, model, start_label_id, stop_label_id, num_labels, hidden_size=768, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(self.hidden_size, num_labels)
        self.crf = CRF(start_label_id, stop_label_id, num_labels)

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        seq_out, _ = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        seq_out = self.dropout(seq_out)
        feats = self.hidden2label(seq_out)
        out = self.crf(feats, labels)
        if self.training:
            return out, _
        return out

