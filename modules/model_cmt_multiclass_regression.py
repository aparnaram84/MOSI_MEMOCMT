import torch
import torch.nn as nn

class CrossModalTransformerRegression(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, dropout=0.2):
        super().__init__()
        
        # Step 1: Modality Embeddings to help the transformer distinguish A, V, T tokens
        self.modality_embeddings = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
        # Step 2: Learnable CLS token to aggregate global sentiment info
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.cmt = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Step 3: Deep Regression Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Add modality information
        x = self.norm(x) + self.modality_embeddings
        
        # Prepend CLS token: [Batch, 4, 256]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Multi-modal Fusion via Transformer Attention
        fused = self.cmt(x)
        
        # Extract the CLS token (index 0) which now contains the fused representation
        pooled = fused[:, 0]
        
        # Scaled Output to match MOSI range [-3, 3]
        return torch.tanh(self.classifier(pooled)) * 3