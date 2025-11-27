# recommender.py
import pandas as pd
import math

SUBJECT_KEYWORDS = {
    "math": ["MATH", "MATHEMATICS"],
    "physics": ["PHYSICS", "PHYS"],
    "chemistry": ["CHEMISTRY", "CHEM"],
    "biology": ["BIOLOGY", "BIO"],
    "computer": ["COMPUTER", "COMPUTER SCIENCE", "IT", "CS"],
    "english": ["ENGLISH", "ENG"],
    "urdu": ["URDU"],
    "islamiat": ["ISLAMIYAT"],
    "pakstudies": ["PAKISTAN STUDIES", "PAK STUDIES", "PAK STUD"]
}

SUBFIELDS = {
    "STEM": ["Engineering", "Computer Science", "Pharmacy", "Medicine"],
    "ARTS": ["Design", "Fine Arts", "Journalism", "Languages"],
    "COMMERCE": ["Finance", "Accounting", "Business", "Economics"]
}

def extract_subject_scores(marks_df: pd.DataFrame):
    scores = {}
    if marks_df is None or marks_df.empty:
        return {k: None for k in SUBJECT_KEYWORDS.keys()}

    # try case-insensitive match
    for sk, tokens in SUBJECT_KEYWORDS.items():
        found = None
        for tk in tokens:
            mask = marks_df['Subject'].str.contains(tk, case=False, na=False)
            if mask.any():
                found = marks_df[mask].iloc[0]
                break
        if found is not None:
            try:
                obt = float(found.get('Obtained') or 0)
                mx = float(found.get('Maximum') or 100)
                scores[sk] = obt/mx if mx > 0 else obt
            except:
                scores[sk] = None
        else:
            scores[sk] = None
    return scores

def normalize_personality(personality: dict):
    normalized = {}
    for k, v in (personality or {}).items():
        try:
            f = float(v)
            if f <= 5:
                normalized[k] = f/5.0
            elif f <= 100:
                normalized[k] = f/100.0
            else:
                normalized[k] = min(1.0, f/100.0)
        except:
            normalized[k] = 0.0
    return normalized

def calculate_best_fit(scores_dict: dict, personality: dict):
    s = {k: (scores_dict.get(k) or 0.0) for k in ['math','physics','chemistry','biology','computer','english','urdu','islamiat','pakstudies']}
    p = normalize_personality(personality)
    stem_score = (s['math']*0.35 + s['physics']*0.25 + s['chemistry']*0.15 + s['computer']*0.25) * (0.7 + 0.3*p.get('openness', 0))
    arts_score = (s['english']*0.5 + s['urdu']*0.3 + s['biology']*0.2) * (0.6 + 0.4*p.get('openness', 0))
    commerce_score = (s['math']*0.5 + s['english']*0.3 + s['computer']*0.2) * (0.6 + 0.4*p.get('conscientiousness', 0))
    scores = {'STEM': max(0, stem_score), 'ARTS': max(0, arts_score), 'COMMERCE': max(0, commerce_score)}
    total = sum(scores.values()) or 1.0
    normalized_scores = {k: round(v/total, 3) for k,v in scores.items()}
    best_field = max(normalized_scores, key=normalized_scores.get)
    return {
        "best_field": best_field,
        "best_subfields": SUBFIELDS.get(best_field, []),
        "scores": normalized_scores
    }

def recommend_field_for_student(marks_df, personality_record: dict):
    marks_scores = extract_subject_scores(marks_df)
    rec = calculate_best_fit(marks_scores, personality_record or {})
    return rec
