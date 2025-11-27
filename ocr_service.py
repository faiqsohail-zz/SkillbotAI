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
    """
    Returns DataFrame with columns: bbox, text, conf
    """
    reader = get_reader(gpu=gpu)
    img, gray = preprocess_image_bytes(image_bytes)
    results = reader.readtext(gray, detail=1)
    rows = []
    for bbox, text, conf in results:
        rows.append({"bbox": bbox, "text": text, "conf": float(conf)})
    return pd.DataFrame(rows)

def extract_number_robust(s):
    s = str(s).replace(",", "")
    m = re.search(r'\d+\.?\d*', s)
    if not m:
        return None
    v = m.group(0)
    return float(v) if '.' in v else int(v)

def parse_marks_from_ocr_df(ocr_df):
    """
    Heuristic parser: scans text items for plausible Subject names and following numbers.
    Returns DataFrame with Subject | Maximum | Obtained
    """
    text_list = list(ocr_df['text'].astype(str).values)
    subjects = []
    maximum = []
    obtained = []
    i = 0
    while i < len(text_list):
        t = text_list[i].strip()
        if not t or len(t) < 2:
            i += 1
            continue
        # plausible subject heuristic
        if re.search(r'[A-Za-z]{2,}', t) and "MARK" not in t.upper():
            nums = []
            j = i + 1
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
    if not subjects:
        # fallback: try any "Subject - Obtained" pattern
        for t in text_list:
            parts = re.split(r'[:\-]', t)
            if len(parts) >= 2:
                # try extract numbers
                nums = [extract_number_robust(p) for p in parts]
                nums = [n for n in nums if n is not None]
                if len(nums) >= 1:
                    subjects.append(parts[0].strip())
                    maximum.append(int(nums[0]) if nums else None)
                    obtained.append(int(nums[1]) if len(nums) > 1 else None)
    df = pd.DataFrame({"Subject": subjects, "Maximum": maximum, "Obtained": obtained})
    return df
