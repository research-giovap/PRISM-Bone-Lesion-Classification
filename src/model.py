import torch
import torch.nn as nn

class TabularTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Learned positional embedding (matching the trained weights)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output_head(x[:, 0, :])

def extract_attention_scientifically(model, X_tensor):
    """
    Manually unrolls the transformer forward pass layer-by-layer,
    extracting raw multi-head attention weights at each layer before
    the residual connection and layer-norm are applied.

    Returns: list of tensors, one per layer, each [Batch, Heads, Seq, Seq]
    """
    model.eval()
    attentions = []

    with torch.no_grad():
        x = X_tensor.unsqueeze(-1)
        x = model.input_proj(x)
        cls = model.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + model.pos_embedding[:, :x.size(1), :]

        for layer in model.transformer.layers:
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
            attentions.append(attn_weights.cpu())

            # Complete the layer pass to correctly feed the next layer
            x = x + layer.dropout1(attn_output)
            x = layer.norm1(x)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + layer.dropout2(src2)
            x = layer.norm2(x)

    return attentions