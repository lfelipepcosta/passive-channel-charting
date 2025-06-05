# Use uma imagem base do TensorFlow 2.15.0 otimizada para GPU.
# Esta versão é projetada para usar CUDA 12.x e cuDNN 8.9,
# o que deve ser mais compatível com sua RTX 4070 Super e drivers recentes.
FROM tensorflow/tensorflow:2.15.0-gpu

# Define o diretório de trabalho principal dentro do contêiner.
WORKDIR /app

# Instala as dependências Python adicionais necessárias.
RUN pip install --no-cache-dir \
    tqdm \
    numpy \
    scipy \
    scikit-learn \
    matplotlib