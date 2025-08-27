import torch
import torch.nn as nn

class MultiHeadModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        
        self.feature_config = cfg["features"]
        self.heads_config = cfg["model"]["heads"]
        self.fusion_cfg = cfg["model"]["fusion"]

        # --- oe-encoded cat features ---
        self.cat_embeddings = nn.ModuleList()
        cat_out_dim = 0
        if "oe_cats" in self.feature_config:
            for card in self.feature_config["oe_cats"]["cardinalities"]:
                emb_dim = self.feature_config["oe_cats"].get("emb_dim", 16)
                self.cat_embeddings.append(nn.Embedding(card + 1, emb_dim)) # reserved N+1 emb for new classes
                cat_out_dim += emb_dim
                
        # --- binary cat features ---
        self.bin_cat_dim = self.feature_config.get("bin_cats", {}).get("input_dim", 0)
        cat_out_dim += self.bin_cat_dim
        
        # --- time features ---
        self.time_mlp = None
        time_out_dim = 0
        if "times" in self.feature_config:
            dims = self.feature_config["times"].get("mlp_dims", [32,16])
            layers = []
            in_dim = self.feature_config["times"]["input_dim"]
            for h in dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            
            self.time_mlp = nn.Sequential(*layers)
            time_out_dim = dims[-1]
            
        # --- text features ---
        self.text_mlp = None
        text_out_dim = 0
        if "text" in self.feature_config:
            dims = self.feature_config["text"].get("mlp_dims", [256, 128])
            dropout_p = self.feature_config["text"].get("dropout", 0.1)
            layers = []
            in_dim = self.feature_config["text"]["input_dim"]
            
            for i, h in enumerate(dims):
                linear = nn.Linear(in_dim, h)
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
                layers.append(linear)
                layers.append(nn.ReLU())
                # Only apply dropout after intermediate layers, not the last
                if i < len(dims) - 1:
                    layers.append(nn.Dropout(dropout_p))
                in_dim = h

            self.text_mlp = nn.Sequential(*layers)
            text_out_dim = dims[-1]

        # --- fusion MLP encoder ---
        fusion_input_dim = cat_out_dim + time_out_dim + text_out_dim
        layers = []
        dropout_p = self.fusion_cfg.get('dropout', 0.1)

        for i, h in enumerate(self.fusion_cfg["hidden_dims"]):
            linear = nn.Linear(fusion_input_dim, h)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            layers.append(linear)
            layers.append(nn.ReLU())
            # Only dropout on intermediate layers, not after last
            if i < len(self.fusion_cfg["hidden_dims"]) - 1:
                layers.append(nn.Dropout(dropout_p))
            fusion_input_dim = h
        self.fusion_mlp = nn.Sequential(*layers)

        # --- multi-head outputs ---
        self.heads = nn.ModuleDict()
        for name, cfg in self.heads_config.items():
            tower_dims = cfg.get("tower_dims")  # optional
            if tower_dims:
                layers = []
                in_dim = fusion_input_dim
                for h in tower_dims:
                    layers.append(nn.Linear(in_dim, h))
                    layers.append(nn.ReLU())
                    if 'dropout' in cfg:
                        layers.append(nn.Dropout(cfg['dropout']))
                    in_dim = h
                layers.append(nn.Linear(in_dim, cfg["out_dim"]))
                self.heads[name] = nn.Sequential(*layers)
                
                for m in self.heads[name]:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            else:
                self.heads[name] = nn.Linear(fusion_input_dim, cfg["out_dim"])
                            

    def forward(self, x: dict):
        fused_parts = []

        # --- oe + bin cats ---
        if len(self.cat_embeddings) > 0:
            cat_feats_oe = torch.cat([emb(x["oe_cats"][:, i]) 
                                    for i, emb in enumerate(self.cat_embeddings)], dim=-1)
        else:
            cat_feats_oe = None

        bin_cats = x.get("bin_cats")
        if cat_feats_oe is not None and bin_cats is not None:
            cat_feats = torch.cat([cat_feats_oe, bin_cats], dim=-1)
        elif cat_feats_oe is not None:
            cat_feats = cat_feats_oe
        elif bin_cats is not None:
            cat_feats = bin_cats
        else:
            cat_feats = None

        if cat_feats is not None:
            fused_parts.append(cat_feats)

        # --- time features ---
        if self.time_mlp is not None and "times" in x:
            fused_parts.append(self.time_mlp(x["times"]))

        # --- text features ---
        if self.text_mlp is not None and "text" in x:
            fused_parts.append(self.text_mlp(x["text"]))

        # --- fusion ---
        fused = torch.cat(fused_parts, dim=-1)
        fused = self.fusion_mlp(fused)

        # --- heads ---
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = torch.squeeze(head(fused), dim=-1)
        return outputs
