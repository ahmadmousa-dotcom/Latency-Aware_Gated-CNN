import torch

class Config:
    PROJECT_NAME = "Latency_Aware_Benchmark_Suite"
    AUTHORS = ["Ahmad Mousa", "Mohamad Hafiz"]
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    
    # Training Settings (Optimized for Benchmark Speed)
    BATCH_SIZE = 128
    EPOCHS = 25           # Sufficient to show pruning convergence
    LEARNING_RATE = 0.05
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # Gating Hyperparameters
    LAMBDA_LATENCY = 1e-4
    TEMP_START = 5.0
    TEMP_MIN = 0.5
    TEMP_DECAY = 0.9