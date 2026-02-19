import cv2
import pytesseract
import os
import pandas as pd
import numpy as np
from skimage.filters import threshold_sauvola

pytesseract.pytesseract.tesseract_cmd = r"tesseract"

BASE_DIR = "pilot_dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
GT_DIR = os.path.join(BASE_DIR, "ground_truth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULT_DIR = os.path.join(BASE_DIR, "results")



RAW_DIR = os.path.join(OUTPUT_DIR, "raw_ocr")
NLM_DIR = os.path.join(OUTPUT_DIR, "nlm")
CLAHE_DIR = os.path.join(OUTPUT_DIR, "clahe_sauvola")
GAMMA_CLAHE_DIR = os.path.join(OUTPUT_DIR, "gamma_clahe_sauvola")
MEDIAN_DIR = os.path.join(OUTPUT_DIR, "median_adaptive")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(NLM_DIR, exist_ok=True)
os.makedirs(CLAHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MEDIAN_DIR, exist_ok=True)

os.makedirs(GAMMA_CLAHE_DIR, exist_ok=True)
# ------------------------------------
# METRİKLER
# ------------------------------------
def cer(gt, pred):
    import Levenshtein
    return Levenshtein.distance(gt, pred) / max(len(gt),1)

def wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    import Levenshtein
    return Levenshtein.distance(gt_words, pred_words) / max(len(gt_words),1)

# ------------------------------------
# ÖN İŞLEME
# ------------------------------------
def preprocess_nlm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    return denoised

def preprocess_clahe_sauvola(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    thresh = threshold_sauvola(contrast, window_size=25)
    binary = (contrast > thresh).astype(np.uint8) * 255

    return binary

def preprocess_median_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)

    th = cv2.adaptiveThreshold(
        median,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )
    return th

def adjust_gamma(image, gamma=1.4):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_gamma_clahe_sauvola(img, gamma=1.4):
    # 1) gamma düzeltme
    adj = adjust_gamma(img, gamma=gamma)

    # 2) gri ton
    gray = cv2.cvtColor(adj, cv2.COLOR_BGR2GRAY)

    # 3) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 4) Sauvola
    thresh = threshold_sauvola(enhanced, window_size=25)
    binary = (enhanced > thresh).astype(np.uint8) * 255

    return binary

# ------------------------------------
# OCR
# ------------------------------------
def run_ocr(img):
    config = r'-l tur --oem 3 --psm 6'
    return pytesseract.image_to_string(img, config=config)

# ------------------------------------
# MAIN
# ------------------------------------
results = []

def change_ext(name, new_ext):
    return os.path.splitext(name)[0] + new_ext


for img_name in sorted(os.listdir(IMAGE_DIR)):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(IMAGE_DIR, img_name)
        gt_path = os.path.join(GT_DIR, change_ext(img_name, ".txt"))

        image = cv2.imread(img_path)
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()

        # ----------------------
        # A: RAW OCR
        # ----------------------
        raw_text = run_ocr(image)

        with open(os.path.join(RAW_DIR, change_ext(img_name, ".txt")), "w", encoding="utf-8") as f:
             f.write(raw_text)

        # ----------------------
        # B: NLM
        # ----------------------
        nlm_img = preprocess_nlm(image)
        cv2.imwrite(os.path.join(NLM_DIR, change_ext(img_name, ".png")), nlm_img)

        nlm_text = run_ocr(nlm_img)
        with open(os.path.join(NLM_DIR, change_ext(img_name, ".txt")), "w", encoding="utf-8") as f:
            f.write(nlm_text)

        # ----------------------
        # C: CLAHE + SAUVOLA
        # ----------------------
        cs_img = preprocess_clahe_sauvola(image)
        cv2.imwrite(os.path.join(CLAHE_DIR, change_ext(img_name, ".png")), cs_img)

        cs_text = run_ocr(cs_img)
        with open(os.path.join(CLAHE_DIR, change_ext(img_name, ".txt")), "w", encoding="utf-8") as f:
            f.write(cs_text)

        # ----------------------
        # D: MEDIAN ADAPTIVE
        # ----------------------
        ma_img = preprocess_median_adaptive(image)
        cv2.imwrite(os.path.join(MEDIAN_DIR, change_ext(img_name, ".png")), ma_img)

        ma_text = run_ocr(ma_img)
        with open(os.path.join(MEDIAN_DIR, change_ext(img_name, ".txt")), "w", encoding="utf-8") as f:
            f.write(ma_text)

        # ----------------------
        # D: Gamma + CLAHE + Sauvola
        # ----------------------
        gcs_img = preprocess_gamma_clahe_sauvola(image, gamma=1.4)
        cv2.imwrite(os.path.join(GAMMA_CLAHE_DIR, change_ext(img_name, ".png")), gcs_img)

        gcs_text = run_ocr(gcs_img)
        with open(os.path.join(GAMMA_CLAHE_DIR, change_ext(img_name, ".txt")), "w", encoding="utf-8") as f:
            f.write(gcs_text)


        # ----------------------
        # METRİKLER
        # ----------------------
        results.append([img_name, "RAW", cer(gt_text, raw_text), wer(gt_text, raw_text)])
        results.append([img_name, "NLM", cer(gt_text, nlm_text), wer(gt_text, nlm_text)])
        results.append([img_name, "CLAHE+Sauvola", cer(gt_text, cs_text), wer(gt_text, cs_text)])
        results.append([img_name, "Median+Adaptive", cer(gt_text, ma_text), wer(gt_text, ma_text)])
        results.append([img_name, "Gamma+CLAHE+Sauvola", cer(gt_text, gcs_text), wer(gt_text, gcs_text)])
    # ------------------------------------
    # RAPOR
    # ------------------------------------
    df = pd.DataFrame(results, columns=["Image", "Scenario", "CER", "WER"])
df.to_csv(os.path.join(RESULT_DIR, "ocr_results.csv"), index=False, encoding="utf-8")

print("\n✅ Tüm işlemler başarıyla tamamlandı!\n")
print(df.groupby("Scenario")[["CER", "WER"]].mean())
