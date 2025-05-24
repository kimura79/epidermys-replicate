from main import predict

def run(inputs):
    image = inputs["image"]  # bytes dell'immagine caricata
    return predict(image)
