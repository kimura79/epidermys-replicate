# Epidermys Replicate Model

Questo modello AI è progettato per analizzare una foto del volto umano e restituire **metadati dermatologici chiave**, tra cui:

- **Fototipo Fitzpatrick**
- **Posa del volto** (frontale, 3/4, profilo)
- **Valori cromatici medi (L*, a*, b*)**
- **Fototipo grezzo preliminare**

È ottimizzato per eseguire inferenza tramite [Replicate.com](https://replicate.com) e integrarsi facilmente con la Progressive Web App (PWA) di Epidermys.

---

## Come funziona

Il modello:
1. Applica una **maschera facciale completa** generata via `mediapipe` con esclusione di occhi, narici e labbra.
2. Calcola il colore medio in **zona neutra** per evitare falsi dati dovuti a melasmi o rossori.
3. Stima il **fototipo Fitzpatrick** in base alla luminanza L*.
4. Determina la **posa del volto** usando l’angolo tra naso e mento.

---

## Input

- Un'immagine del volto (JPG o PNG), passata come campo `image` nei bytes (non URL).

---

## Output (esempio)

```json
{
  "fototipo": "Tipo III (medio-chiara)",
  "posa": "frontale",
  "L*": 68.7,
  "a*": 14.5,
  "b*": 18.2,
  "fototipo_grezzo": "III"
}
