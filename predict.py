from cog import BasePredictor, Input, Path
from PIL import Image
import numpy as np

class Predictor(BasePredictor):
    def predict(self, image: Path = Input(description="Foto del volto")) -> str:
        # Carica immagine come array numpy
        img = Image.open(image).convert("RGB")
        img_array = np.array(img)

        # Messaggio di log (verrà stampato durante il push)
        print(f"✅ Immagine ricevuta. Dimensioni: {img_array.shape}")

        # Restituisce una stringa di esempio
        return "✅ Analisi completata correttamente"
