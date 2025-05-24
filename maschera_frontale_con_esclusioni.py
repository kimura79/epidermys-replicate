
import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import splprep, splev

def genera_maschera_frontale(image_rgb):
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return np.zeros((h, w, 4), dtype=np.uint8)

    landmarks = results.multi_face_landmarks[0].landmark
    mediapipe_points = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    all_points = {idx: pt for idx, pt in enumerate(mediapipe_points)}

    corrispondenze = {
       500: (54, 60),   
    501: (67, 40),  
    502: (109, 40),  
    503: (10, 40),   
    504: (338, 40),   
    505: (297, 40),   
    506: (284, 60),   
    507: (103, 90),   
    510: (67, 105),  
    511: (109, 106),   
    512: (10, 110),    
    513: (338, 106),  
    514: (297, 100),  
    515: (332, 90),
    }
    for new_id, (ref_id, offset_y) in corrispondenze.items():
        x_ref, y_ref = mediapipe_points[ref_id]
        all_points[new_id] = (x_ref, y_ref - offset_y)

    face_outline_ids = [
        500, 510, 511, 512, 513, 514, 515, 506,
        284, 389, 366, 288, 379, 175, 150,
        58, 137, 139, 54, 507
    ]
    face_outline = np.array([all_points[i] for i in face_outline_ids], dtype=np.int32)

    tck, u = splprep(face_outline.reshape(-1, 2).T, s=5.0, per=True)
    unew = np.linspace(0, 1, 600)
    smoothed_pts = np.array(splev(unew, tck)).T.astype(np.int32)
    cv2.fillPoly(mask, [smoothed_pts], 255)

    def exclude(ids):
        return np.array([mediapipe_points[i] for i in ids], dtype=np.int32)

    aree_da_escludere = [
        [33, 160, 158, 133, 153, 144],
        [362, 385, 387, 263, 373, 380],
        [57,39,37,0,267,269,287,321,405,314,17,84,181,91,146],
        [113, 161, 468, 190, 26, 472, 110],
        [414, 473, 263, 446, 339, 477, 463],
        [218, 59, 60, 141, 44],
        [461, 457, 438, 439, 460],
        [70, 63, 105, 66, 107],
        [336, 296, 334, 293, 300],
    ]

    for region in aree_da_escludere:
        cv2.fillPoly(mask, [exclude(region)], 0)

    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[:, :, 3] = mask
    return mask_rgba
