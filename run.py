from predict import predict

def run(inputs):
    image = inputs["image"]
    eta = inputs.get("eta", None)
    return predict(image, eta)
