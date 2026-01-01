import torch
import torch.nn as nn

class CrossModalTransformerMulticlass(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, num_classes=7, dropout=0.4):
        super().__init__()
        
        # Normalization helps align BERT, HuBERT, and ResNet scales
        self.norm = nn.LayerNorm(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.cmt = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Deepened classification head to capture 7-class intensities
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Apply custom initialization to prevent majority class collapse
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [batch, 3, 256]
        x = self.norm(x) 
        fused = self.cmt(x)
        # Pool across the A-V-T tokens
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)