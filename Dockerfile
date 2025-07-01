FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

RUN pip install --no-cache-dir \
    tqdm \
    numpy \
    scipy \
    scikit-learn \
    matplotlib