import cv2
import numpy as np
from maschera_frontale_completa import genera_maschera_frontale
from skimage.color import rgb2lab
import mediapipe as mp

class Predictor:
    def predict(self, image, eta_input=None):
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Calcolo posa con landmark naso-mento
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return {"errore": "Nessun volto rilevato"}

            landmarks = results.multi_face_landmarks[0].landmark
            naso = np.array([landmarks[1].x * w, landmarks[1].y * h])
            mento = np.array([landmarks[152].x * w, landmarks[152].y * h])
            angolo = np.arctan2((mento - naso)[0], (mento - naso)[1]) * 180 / np.pi
            posa = "frontale" if abs(angolo) < 8 else "non_frontale"

        if posa != "frontale":
            return {"posa": posa, "errore": "La foto non è frontale, analisi non effettuata"}

        # Maschera pelle
        maschera = genera_maschera_frontale(image_rgb)

        # Conversione CIELab
        lab = rgb2lab(image_rgb)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Colore medio pelle
        L_pelle = np.mean(L[maschera])
        A_pelle = np.mean(A[maschera])
        B_pelle = np.mean(B[maschera])

        # Fototipo
        if L_pelle >= 80:
            fototipo = "1"
        elif L_pelle >= 65:
            fototipo = "2"
        elif L_pelle >= 53:
            fototipo = "3"
        elif L_pelle >= 45:
            fototipo = "4"
        elif L_pelle >= 38:
            fototipo = "5"
        else:
            fototipo = "6"

        return {
            "posa": posa,
            "fototipo": fototipo,
            "L*": round(L_pelle, 2),
            "a*": round(A_pelle, 2),
            "b*": round(B_pelle, 2),
            "eta_stimata": eta_input if eta_input else "non_specificata"
        }
