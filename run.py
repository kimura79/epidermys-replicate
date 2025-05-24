from main import predict

def run(inputs):
    image = inputs["image"]
    eta = inputs.get("eta", None)  # opzionale, ma richiesto dal tuo flusso
    return predict(image, eta)
