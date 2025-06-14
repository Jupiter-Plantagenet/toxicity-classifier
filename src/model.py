import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel

class ToxicityClassifier(DistilBertPreTrainedModel):
    """
    DistilBERT model for toxicity classification.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1  # Binary classification
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Ground truth labels
            
        Returns:
            dict: Model outputs including loss and logits
        """
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': distilbert_output.hidden_states,
            'attentions': distilbert_output.attentions
        }
