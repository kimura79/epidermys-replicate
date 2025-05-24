import numpy as np
import cv2
from PIL import Image
import io
from skimage.color import rgb2lab
import mediapipe as mp
from maschera_frontale_con_esclusioni import genera_maschera_frontale
from deepface import DeepFace

def predict(image: bytes, eta: int = None) -> dict:
    # Carica immagine
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Maschera viso con esclusione occhi, narici, labbra
    mask_rgba = genera_maschera_frontale(image_bgr)
    mask = mask_rgba[:, :, 3] > 0

    # LAB conversione e fototipo grezzo
    lab = rgb2lab(image_np)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    L_global = np.mean(L[mask])

    if L_global > 80: ft_grezzo = "I"
    elif L_global > 70: ft_grezzo = "II"
    elif L_global > 60: ft_grezzo = "III"
    elif L_global > 50: ft_grezzo = "IV"
    elif L_global > 40: ft_grezzo = "V"
    else: ft_grezzo = "VI"

    # Area neutra senza macchie/rossori
    if ft_grezzo in ["I", "II", "III"]:
        maschera_neutra = (A < 18) & (B > 5) & (B < 25) & (L > 45) & (L < 90) & mask
    else:
        maschera_neutra = (A < 20) & (L < 90) & mask

    if np.sum(maschera_neutra) < 5000:
        maschera_neutra = mask

    L_mean = np.mean(L[maschera_neutra])
    A_mean = np.mean(A[maschera_neutra])
    B_mean = np.mean(B[maschera_neutra])

    def stima_fototipo(L_val):
        if L_val > 80: return "Tipo I (molto chiara)"
        elif L_val > 70: return "Tipo II (chiara)"
        elif L_val > 60: return "Tipo III (medio-chiara)"
        elif L_val > 50: return "Tipo IV (olivastra)"
        elif L_val > 40: return "Tipo V (marrone)"
        else: return "Tipo VI (molto scura)"

    fototipo = stima_fototipo(L_mean)

    # Stima posa con angolo naso-mento
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(image_np)

    posa = "non rilevata"
    if results.multi_face_landmarks:
        h, w = image_np.shape[:2]
        lm = results.multi_face_landmarks[0].landmark
        naso = np.array([lm[1].x * w, lm[1].y * h])
        mento = np.array([lm[152].x * w, lm[152].y * h])
        angolo = np.degrees(np.arctan2(mento[0] - naso[0], mento[1] - naso[1]))

        if abs(angolo) < 15: posa = "frontale"
        elif 15 <= angolo < 40: posa = "3/4 destro"
        elif -40 < angolo <= -15: posa = "3/4 sinistro"
        elif angolo >= 40: posa = "profilo destro"
        elif angolo <= -40: posa = "profilo sinistro"

    # Stima età solo se frontale e non già fornita
    eta_stimata = eta
    if eta is None and posa == "frontale":
        try:
            analisi = DeepFace.analyze(img_path=image_np, actions=["age"], enforce_detection=False)
            eta_stimata = int(round(analisi[0]["age"]))
        except:
            eta_stimata = "non stimabile"

    return {
        "fototipo": fototipo,
        "posa": posa,
        "L*": round(L_mean, 1),
        "a*": round(A_mean, 1),
        "b*": round(B_mean, 1),
        "fototipo_grezzo": ft_grezzo,
        "età": eta_stimata
    }
