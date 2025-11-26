# --- Server Configuration ---
MODEL_NAME = "SimpleCNN"
AGGREGATION_STRATEGY = "FedAvgM" # Options: FedAvg, FedAvgM, FedAdam
SERVER_LEARNING_RATE = 1.0       # For FedAvgM/FedOpt (How fast the global model updates)
SERVER_MOMENTUM = 0.9            # For FedAvgM
DEVICE = "auto"
TOTAL_ROUNDS = 10
MIN_CLIENTS_PER_ROUND = 10
MIN_CLIENTS_FOR_AGGREGATION = 10
SAVED_MODEL_NAME = "final_global_model.pth"

# --- Client Configuration ---
TOTAL_CLIENTS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
POLL_INTERVAL = 10

# --- Data Configuration (Non-IID) ---
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 42

# --- Simulation of FL Conditions ---
CLIENT_DROPOUT_RATE = 0.0
ROUND_TIMEOUT_SEC = 300
# --- NEW: Slow Sender & Latency Simulation ---
SLOW_SENDER_RATE = 0.0
SLOW_SENDER_DELAY_SEC = 30
NETWORK_LATENCY_RATE = 0.0
NETWORK_LATENCY_DELAY_SEC = 5

# --- Robustness ---
FEDPROX_MU = 0.01  # Proximal term weight (0.0 = disabled). Try 0.01 - 1.0