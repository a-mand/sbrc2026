import logging
from flask import Flask, request, jsonify, send_file
from collections import OrderedDict
import os
import io
import signal
import threading
from threading import Timer, Lock
import random
import json

# --- ML Imports ---
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from shared.models import get_model
from server.strategies import get_strategy  # <--- NEW IMPORT
import config

app = Flask(__name__)

werkzeug_logger = logging.getLogger('werkzeug')

log_dir = os.path.join("fl_logs", "tensorboard")
tb_writer = SummaryWriter(log_dir=log_dir)

# 2. Define a custom filter
class StatusFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "GET /status" in msg or "POST /register" in msg:
            return False 
        return True

# 3. Add the filter to the logger
werkzeug_logger.addFilter(StatusFilter())

# --- Configuration ---
fl_state = {
    "status": "WAITING", 
    "current_round": 0,
    "client_updates": [],
    "registered_clients": set(),
    "aggregation_lock": Lock(), 
    "round_timer": None          
}

# We need a place to store the model (as tensors in memory)
global_models_by_round = {}

# --- NEW: Global strategy variable ---
fl_strategy = None
# ---

def setup_initial_model():
    """
    Creates the very first model for round 0 and stores it in memory.
    Initializes the aggregation strategy.
    """
    global global_models_by_round, fl_strategy
    
    # 1. Create the model instance first
    initial_model = get_model(config.MODEL_NAME)
    
    # 2. Initialize Strategy (NEW)
    fl_strategy = get_strategy(
        config.AGGREGATION_STRATEGY, 
        global_model=initial_model,  # Pass instance for FedOpt
        lr=config.SERVER_LEARNING_RATE,
        momentum=config.SERVER_MOMENTUM
    )
    
    # 3. Store the state_dict (tensors) directly
    global_models_by_round[0] = initial_model.state_dict()
    
    logger.info(f"Initial model for round 0 created and stored in memory.")
    logger.info(f"Strategy: {config.AGGREGATION_STRATEGY} initialized.")

def state_dict_to_bytes(data_payload):
    """
    Serializes a payload (Dict or OrderedDict) to bytes.
    """
    buffer = io.BytesIO()
    torch.save(data_payload, buffer)
    buffer.seek(0)
    return buffer.read()

# --- Helper Functions ---

def load_test_data():
    """Loads the CIFAR-10 test dataset."""
    logger.info("Loading CIFAR-10 test dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=2)
    logger.info("Test dataset loaded.")
    return test_loader

def evaluate_model(model_state_tensors, test_loader):
    """Evaluates the global model on the test dataset."""
    
    # --- Check for GPU and set device ---
    forced_device = config.DEVICE.lower()
    if forced_device == "cpu":
        device = torch.device("cpu")
    elif forced_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available! Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Evaluating on device: {device}")
    
    model = get_model(config.MODEL_NAME)
    model.load_state_dict(model_state_tensors)
    model.to(device) 
    model.eval() 
    
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad(): 
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss

def check_and_aggregate(test_loader):
    if fl_state["status"] == "AGGREGATING":
        return
        
    if fl_state["round_timer"]:
        fl_state["round_timer"].cancel()
        fl_state["round_timer"] = None
        
    fl_state["status"] = "AGGREGATING"
    current_round = fl_state["current_round"]
    
    if len(fl_state["client_updates"]) == 0:
        logger.warning(f"No updates received for round {current_round}. Skipping aggregation.")
    else:
        logger.info(f"--- Aggregating {len(fl_state['client_updates'])} updates for round {current_round} ---")
        
        # 1. Aggregate
        aggregation_result = fl_strategy.aggregate(fl_state["client_updates"])
        
        # 2. Unwrap Payload (Handle Scaffold vs Standard)
        if isinstance(aggregation_result, dict) and "model_state" in aggregation_result:
            # Complex Strategy (Scaffold)
            new_global_model_tensors = aggregation_result["model_state"]
            # We save the WHOLE result (including global_c) to send to clients
            payload_to_save = aggregation_result 
        else:
            # Simple Strategy (FedAvg)
            new_global_model_tensors = aggregation_result
            payload_to_save = aggregation_result

        # 3. Evaluate (Use only the model weights)
        if new_global_model_tensors:
            accuracy, loss = evaluate_model(new_global_model_tensors, test_loader)
            
            # Log Metrics
            tb_writer.add_scalar("Global/Accuracy", accuracy, current_round)
            tb_writer.add_scalar("Global/Loss", loss, current_round)
            tb_writer.flush()
            
            logger.info(f"--- Round {current_round} Complete. Accuracy: {accuracy:.2f}%, Loss: {loss:.4f} ---")
            
            # 4. Save State for Next Round
            # We save the wrapped payload so download_model() sends the right data
            global_models_by_round[current_round + 1] = payload_to_save
            
            if current_round == config.TOTAL_ROUNDS - 1:
                try:
                    save_path = os.path.join("fl_logs", config.SAVED_MODEL_NAME)
                    os.makedirs("fl_logs", exist_ok=True)
                    # For the final file, usually just save the weights
                    torch.save(new_global_model_tensors, save_path)
                    logger.info(f"--- Final model saved to {save_path} ---")
                except Exception as e:
                    logger.error(f"Failed to save final model: {e}", exc_info=True)

    fl_state["current_round"] += 1
    fl_state["client_updates"] = [] # Clear updates for next round
    
    if fl_state["current_round"] >= config.TOTAL_ROUNDS:
        logger.info(f"--- All {config.TOTAL_ROUNDS} rounds complete. ---")
        fl_state["status"] = "TRAINING_COMPLETE"
        threading.Timer(2.0, shutdown_server).start()
    else:
        start_next_round_timer(test_loader)

def trigger_aggregation(test_loader):
    """Callback function for the round timer."""
    with fl_state["aggregation_lock"]:
        if fl_state["status"] == "WAITING": 
            logger.info(
                f"--- ROUND {fl_state['current_round']} TIMEOUT ---"
                f" Received {len(fl_state['client_updates'])} updates."
            )
            check_and_aggregate(test_loader) 

def start_next_round_timer(test_loader): 
    """Starts the timer for the current round."""
    global fl_state
    
    round_num = fl_state["current_round"]
    logger.info(f"--- Starting Round {round_num}. Timeout: {config.ROUND_TIMEOUT_SEC}s ---")
    fl_state["status"] = "WAITING"
    fl_state["client_updates"] = []
    
    fl_state["round_timer"] = Timer(
        config.ROUND_TIMEOUT_SEC, 
        trigger_aggregation, 
        args=[test_loader] 
    )
    fl_state["round_timer"].start()

def shutdown_server():
    logger.info("--- Training complete. Sending shutdown signal. ---")
    os.kill(os.getpid(), signal.SIGINT)

# --- Flask Routes ---

logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    client_id = data.get("client_id")
    
    with fl_state["aggregation_lock"]:
        if client_id not in fl_state["registered_clients"]:
            fl_state["registered_clients"].add(client_id)
            logger.info(f"Client {client_id} registered. Total clients: {len(fl_state['registered_clients'])}")
        
        if fl_state["current_round"] == 0 and fl_state["round_timer"] is None:
            logger.info("First client registered. Starting Round 0 timer.")
            with app.app_context():
                test_loader = app.config['TEST_LOADER'] 
                start_next_round_timer(test_loader)    
        
    return jsonify({"status": fl_state["status"]})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": fl_state["status"],
        "current_round": fl_state["current_round"]
    })

@app.route('/download_model', methods=['GET'])
def download_model():
    global global_models_by_round
    
    try:
        requested_round = request.args.get('round', type=int)
        if requested_round is None:
            return jsonify({"error": "'round' parameter is required"}), 400
            
        state_dict = global_models_by_round.get(requested_round)
        
        if state_dict:
            logger.info(f"Serving model for round {requested_round} to a client.")
            model_bytes = state_dict_to_bytes(state_dict)
            return send_file(
                io.BytesIO(model_bytes),
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=f'model_round_{requested_round}.pth'
            )
        else:
            logger.warning(f"Client requested model for round {requested_round}, which is not found.")
            return jsonify({"error": f"Model for round {requested_round} not found."}), 404
            
    except Exception as e:
        logger.error(f"Error in /download_model: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/submit_update', methods=['POST'])
def submit_update():
    if fl_state["status"] != "WAITING":
        return jsonify({"error": "Server is not accepting updates right now."}), 400
    
    if 'model' not in request.files or 'json' not in request.form:
        return jsonify({"error": "Missing model or metadata"}), 400

    try:
        # 1. Parse JSON Metadata
        metadata = json.loads(request.form['json'])
        client_id = metadata.get('client_id')
        num_samples = metadata.get('num_samples')
        metrics = metadata.get('metrics', {}) 
        
        if not all([client_id, num_samples is not None]):
            return jsonify({"error": "Missing required metadata"}), 400
        
        # Client Dropout Simulation
        if config.CLIENT_DROPOUT_RATE > 0 and random.random() < config.CLIENT_DROPOUT_RATE:
            logger.warning(f"SIMULATING DROPOUT: Ignoring update from {client_id}.")
            return jsonify({"status": "Update received"})
        
        # 2. Parse Binary Data (The Model + Tensors)
        file_bytes = request.files['model'].read()
        binary_data = torch.load(io.BytesIO(file_bytes), map_location='cpu')
        
        # Check if it is a Composite Payload (Model + Extra Tensors)
        if isinstance(binary_data, dict) and "model_state" in binary_data:
            client_state_dict = binary_data["model_state"]
            # Extract the tensor metrics (e.g. scaffold_delta_c) and merge them back into metrics
            if "tensor_metrics" in binary_data:
                metrics.update(binary_data["tensor_metrics"])
        else:
            # It's just a standard model state_dict (Old Client / FedAvg)
            client_state_dict = binary_data
        
    except Exception as e:
        logger.error(f"Error parsing client update: {e}", exc_info=True)
        return jsonify({"error": "Failed to parse update"}), 400
    
    with fl_state["aggregation_lock"]:
        if fl_state["status"] != "WAITING":
            return jsonify({"error": "Server is not accepting updates right now (locked)."}), 400
        
        for update in fl_state["client_updates"]:
            if update["client_id"] == client_id:
                logger.warning(f"Client {client_id} tried to submit a second update.")
                return jsonify({"error": "Already received update for this round"}), 400
        
        # Store everything
        fl_state["client_updates"].append({
            "client_id": client_id,
            "num_samples": num_samples,
            "model_update": client_state_dict,
            "metrics": metrics # Now contains both JSON info AND SCAFFOLD Tensors
        })
        
        logger.info(
            f"Update from {client_id} (Round {fl_state['current_round']}): "
            f"Upload: {metrics.get('payload_size_mb', 0):.2f}MB "
            f"({len(fl_state['client_updates'])}/{config.MIN_CLIENTS_PER_ROUND})"
        )
        
        if len(fl_state["client_updates"]) >= config.MIN_CLIENTS_FOR_AGGREGATION:
            logger.info(f"Quorum met. Triggering aggregation early.")
            check_and_aggregate(app.config['TEST_LOADER'])
        
    return jsonify({"status": "Update received"})

if __name__ == '__main__':
    try:
        test_loader = load_test_data()
        app.config['TEST_LOADER'] = test_loader
    except Exception as e:
        logger.error(f"Failed to load test data: {e}. Exiting.")
        exit(1)
        
    setup_initial_model() 
    
    logger.info(f"--- FL Server starting... ---")
    app.run(host='0.0.0.0', port=5000, debug=False)