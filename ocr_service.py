# ocr_service.py
import re
import io
import cv2
import numpy as np
import pandas as pd
import easyocr

_reader = None

def get_reader(gpu=False):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=gpu)
    return _reader

def preprocess_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return img, gray

def run_ocr_on_image(image_bytes, gpu=False):
    reader = get_reader(gpu=gpu)
    img, gray = preprocess_image_bytes(image_bytes)
    results = reader.readtext(gray, detail=1)
    # return DataFrame rows with bbox, text, conf
    rows = []
    for bbox, text, conf in results:
        rows.append({"bbox": bbox, "text": text, "conf": float(conf)})
    return pd.DataFrame(rows)

# Robust number extractor
def extract_number_robust(s):
    s = str(s).replace(",", "")
    m = re.search(r'\d+\.?\d*', s)
    if not m:
        return None
    v = m.group(0)
    return float(v) if '.' in v else int(v)

# Parse marks from OCR DataFrame into Subject / Maximum / Obtained
def parse_marks_from_ocr_df(ocr_df):
    text_list = list(ocr_df['text'].astype(str).values)
    subjects = []
    maximum = []
    obtained = []

    # Very similar logic you had, but simplified and robust
    i = 0
    while i < len(text_list):
        t = text_list[i].strip()
        if not t or len(t) < 2:
            i += 1
            continue
        # heuristics for subject lines: contain alpha and not "MARKS" header
        if re.search(r'[A-Za-z]{2,}', t) and "MARK" not in t.upper():
            # scan next 5 tokens for numbers
            nums = []
            j = i+1
            scanned = 0
            while j < len(text_list) and scanned < 6 and len(nums) < 2:
                n = extract_number_robust(text_list[j])
                if n is not None:
                    nums.append(n)
                j += 1
                scanned += 1
            if len(nums) >= 2:
                subjects.append(t)
                maximum.append(int(nums[0]))
                obtained.append(int(nums[1]))
                i = j
                continue
        i += 1

    df = pd.DataFrame({"Subject": subjects, "Maximum": maximum, "Obtained": obtained})
    return df
