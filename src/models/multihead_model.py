import torch
import torch.nn as nn

class MultiHeadModel(nn.Module):
    def __init__(
        self,
        feature_config: dict,
        heads_config: dict,
        hidden_dims: list = [256, 128]
    ):
        super().__init__()

        # --- oe-encoded cat features ---
        self.cat_embeddings = nn.ModuleList()
        cat_out_dim = 0
        if "oe_cats" in feature_config:
            for card in feature_config["oe_cats"]["cardinalities"]:
                emb_dim = feature_config["oe_cats"].get("emb_dim", 16)
                self.cat_embeddings.append(nn.Embedding(card + 1, emb_dim)) # reserved N+1 emb for new classes
                cat_out_dim += emb_dim
                
        # --- binary cat features ---
        self.bin_cat_dim = feature_config.get("bin_cats", {}).get("input_dim", 0)
        cat_out_dim += self.bin_cat_dim
        
        # --- time features ---
        self.time_mlp = None
        time_out_dim = 0
        if "times" in feature_config:
            dims = feature_config["times"].get("mlp_dims", [32,16])
            layers = []
            in_dim = feature_config["times"]["input_dim"]
            for h in dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            
            self.time_mlp = nn.Sequential(*layers)
            time_out_dim = dims[-1]
            
        # --- text features ---
        self.text_mlp = None
        text_out_dim = 0
        if "text" in feature_config:
            dims = feature_config["text"].get("mlp_dims", [256, 128])
            layers = []
            in_dim = feature_config["text"]["input_dim"]
            for h in dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            
            self.text_mlp = nn.Sequential(*layers)
            text_out_dim = dims[-1]
            
        # --- fusion MLP encoder ---
        fusion_input_dim = cat_out_dim + time_out_dim + text_out_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(fusion_input_dim, h))
            layers.append(nn.ReLU())
            fusion_input_dim = h
        self.fusion_mlp = nn.Sequential(*layers)

        # --- multi-head outputs ---
        self.heads = nn.ModuleDict()
        for name, cfg in heads_config.items():
            tower_dims = cfg.get("tower_dims")  # optional
            if tower_dims:
                layers = []
                in_dim = fusion_input_dim
                for h in tower_dims:
                    layers.append(nn.Linear(in_dim, h))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.15))
                    in_dim = h
                layers.append(nn.Linear(in_dim, cfg["out_dim"]))
                self.heads[name] = nn.Sequential(*layers)
                
                for m in self.heads[name]:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            else:
                self.heads[name] = nn.Linear(fusion_input_dim, cfg["out_dim"])

        self.heads_config = heads_config
            

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

