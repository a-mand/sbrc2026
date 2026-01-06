import torch
import io
import logging

logger = logging.getLogger(__name__)

def get_model_bytes(payload):
    """
    Serializes a payload (model dict or composite dict) to bytes.
    """
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return buffer.read()

def set_model_from_bytes(model, model_bytes):
    """
    Carrega os bytes. Se for um Modelo Leve ou SCAFFOLD, extrai as partes corretas.
    Retorna o 'extra_payload' (como estratégia ou variates de controle).
    """
    buffer = io.BytesIO(model_bytes)
    try:
        # Carrega os dados para a CPU primeiro por segurança
        data = torch.load(buffer, map_location='cpu')
        
        # 1. Caso seja um Payload Composto (Usado para SCAFFOLD ou Estratégias Proativas)
        if isinstance(data, dict) and "model_state" in data:
            state_dict = data["model_state"]
            
            # Verificamos se é um update parcial (Modelo Leve)
            # Se o state_dict tiver menos chaves que o modelo original, carregamos com strict=False
            current_model_dict = model.state_dict()
            is_partial = len(state_dict.keys()) < len(current_model_dict.keys())
            
            if is_partial:
                logger.info("Modelo Leve detectado: Carregando apenas camadas parciais.")
                current_model_dict.update(state_dict)
                model.load_state_dict(current_model_dict)
            else:
                model.load_state_dict(state_dict)
                
            return data.get("extra_payload") 

        # 2. Caso seja um state_dict padrão
        else:
            model.load_state_dict(data)
            return None
            
    except Exception as e:
        logger.error(f"Falha ao carregar o state_dict: {e}")
        raise