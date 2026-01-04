import torch
import torch.nn as nn

class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Token-level A-V-T fusion
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.cmt = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: [batch, 3, 256] (A, V, and T tokens)
        fused = self.cmt(x)
        
        # Calculate Modality Attention Weights
        attn_weights = torch.softmax(torch.mean(torch.abs(fused), dim=-1), dim=-1)
        
        # Binary Sentiment Prediction (Raw Logit)
        logits = self.classifier(fused.mean(dim=1))
        return logits, attn_weights