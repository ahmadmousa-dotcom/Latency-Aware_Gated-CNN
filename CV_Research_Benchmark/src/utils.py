import torch
import numpy as np
import random
from .config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_temperature(epoch):
    return max(Config.TEMP_MIN, Config.TEMP_START * (Config.TEMP_DECAY ** epoch))