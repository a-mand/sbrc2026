#!/bin/bash
# Script para rodar o servidor e 3 veículos simultaneamente

echo "Lançando Servidor Central..."
python3 app.py &  # O '&' faz rodar em segundo plano
sleep 3           # Espera o servidor subir

echo "Iniciando Frota de Veículos..."
CLIENT_ID=client_001 python3 client.py &
CLIENT_ID=client_002 python3 client.py &
CLIENT_ID=client_003 python3 client.py &

echo "Simulação em andamento. Use 'pkill python3' para parar tudo."