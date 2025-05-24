from cog import BasePredictor, Input, Path
from PIL import Image
import numpy as np

class Predictor(BasePredictor):
    def predict(self, image: Path = Input(description="Foto del volto")) -> str:
        # Carica immagine come array
        img = Image.open(image).convert("RGB")
        img_array = np.array(img)

        # QUI puoi aggiungere la tua analisi (es. analisi dermatologica, metadati ecc.)
        # Per esempio, ritorni un messaggio fittizio:
        print(f"✅ Immagine ricevuta. Dimensioni: {img_array.shape}")
        
        return "Analisi completata"
