from segment_anything import sam_model_registry, SamPredictor
import torch

model_type = "vit_h"
checkpoint_path = "path/to/sam_vit_h.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
predictor = SamPredictor(sam)