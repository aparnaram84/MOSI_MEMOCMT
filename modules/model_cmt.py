import torch.nn as nn

class CrossModalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Token-level A-V-T fusion [cite: 134, 163]
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.cmt = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(256, 1) # Sentiment prediction [cite: 196]

    def forward(self, x):
        # x shape: [batch, 3, 256] (A, V, and T tokens)
        fused = self.cmt(x)
        return self.classifier(fused.mean(dim=1))