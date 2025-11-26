import torch
import io
import logging

logger = logging.getLogger(__name__)

def get_model_bytes(model):
    """
    Serializes the model state_dict to a binary BytesIO buffer.
    """
    logger.debug("Serializing model to bytes...")
    buffer = io.BytesIO()
    # Save just the state_dict (weights), not the whole class
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

def set_model_from_bytes(model, model_bytes):
    """
    Loads a binary stream of weights into the model.
    """
    logger.debug("Deserializing model from bytes...")
    buffer = io.BytesIO(model_bytes)
    try:
        state_dict = torch.load(buffer, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.debug("Successfully loaded new model state.")
    except Exception as e:
        logger.error(f"Failed to load state_dict from bytes: {e}")
        raise