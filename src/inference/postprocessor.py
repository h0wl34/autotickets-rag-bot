import numpy as np
import joblib
from typing import Dict, Any

class Postprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.head_mappings = {}
        
        for head, head_cfg in self.cfg.get("heads", {}).items():
            if "mapping_path" in head_cfg:
                self.head_mappings[head] = joblib.load(head_cfg["mapping_path"])
            elif "mapping" in head_cfg:
                self.head_mappings[head] = head_cfg["mapping"]
            else:
                self.head_mappings[head] = None

    def process_head(self, head_name: str, output: np.ndarray) -> Any:
        head_cfg = self.cfg["heads"][head_name]
        mapping = self.head_mappings.get(head_name)
        head_type = head_cfg["type"]

        if head_type == "classification":
            pred_idx = np.argmax(output, axis=-1)
            
            if np.isscalar(pred_idx):
                pred_idx = np.array([pred_idx])
                
            if mapping is not None:
                # If it's a LabelEncoder
                if hasattr(mapping, "inverse_transform"):
                    return mapping.inverse_transform(pred_idx)
                # If it's a manual dict
                return [mapping.get(i, i) for i in pred_idx]
            return pred_idx

        elif head_type == "binary":
            threshold = head_cfg.get("threshold", 0.5)
            pred_bin = (output.squeeze() >= threshold).astype(int)
            
            if np.isscalar(pred_bin):
                pred_bin = np.array([pred_bin])
        
            if mapping is not None:
                return [mapping.get(i, i) for i in pred_bin]
            return pred_bin

        elif head_type == "regression":
            return output.squeeze()

        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def process(self, outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        batch_size = None
        # Determine batch size from any head
        for out in outputs.values():
            batch_size = out.shape[0] if out.ndim > 0 else 1
            break
    
        processed_heads =  {h: self.process_head(h, out) for h, out in outputs.items()}
        
        if batch_size == 1:
            return {h: v[0] if isinstance(v, (list, np.ndarray)) else v
                    for h, v in processed_heads.items()}
        else:
            # For batch: return list of dicts
            return [
                {h: v[i] for h, v in processed_heads.items()} 
                for i in range(batch_size)
            ]
