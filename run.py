from cog import BasePredictor, Input, Path
from PIL import Image
from predict import Predictor

class PredictorCog(BasePredictor):
    def setup(self):
        self.predictor = Predictor()

    def predict(
        self,
        image: Path = Input(description="Immagine del volto"),
        eta_input: str = Input(description="Età stimata utente", default=None)
    ) -> dict:
        image = Image.open(image).convert("RGB")
        return self.predictor.predict(image, eta_input)
