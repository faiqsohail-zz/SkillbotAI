# recommender.py
import math
import pandas as pd

# Example mapping: canonical names to search tokens (you can extend)
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
    "COMMERCE": ["Finance", "Accounting", "Business", "Economics"],
    # add more as needed
}

def extract_subject_scores(marks_df: pd.DataFrame):
    """
    Expect marks_df with columns 'Subject' and 'Obtained' or similar.
    Returns normalized dict of subject -> score (0..1 relative to max if available)
    """
    scores = {}
    if marks_df is None or marks_df.empty:
        return {k: None for k in SUBJECT_KEYWORDS.keys()}

    # prepare subjects text to search
    for subj_key, tokens in SUBJECT_KEYWORDS.items():
        # search for any token in Subject column
        found = None
        for token in tokens:
            mask = marks_df['Subject'].str.contains(token, case=False, na=False)
            if mask.any():
                row = marks_df[mask].iloc[0]
                found = row
                break
        if found is not None:
            obt = found.get('Obtained') if 'Obtained' in found else found.get('marks') if 'marks' in found else None
            mx = found.get('Maximum') if 'Maximum' in found else None
            try:
                obt_n = float(obt) if obt is not None else None
                mx_n = float(mx) if mx is not None else None
                if obt_n is not None and mx_n and mx_n>0:
                    scores[subj_key] = obt_n/mx_n
                elif obt_n is not None:
                    scores[subj_key] = obt_n
                else:
                    scores[subj_key] = None
            except:
                scores[subj_key] = None
        else:
            scores[subj_key] = None
    return scores

def normalize_personality(personality: dict):
    # Example: if values are 0-5 map to 0-1; if 0-100 map to 0-1
    normalized = {}
    for k, v in (personality or {}).items():
        try:
            if v is None:
                normalized[k] = 0.0
            else:
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
    """
    returns: dict with 'best_field', 'best_subfields', 'scores' (field->score)
    This is a heuristic rule-based aggregator; replace with ML later.
    """
    # Basic buckets (example):
    # STEM weight: math+physics+chemistry+computer
    # ARTS weight: english + urdu + creativity (from personality)
    # COMMERCE: math + english + conscientiousness
    s = {k: (scores_dict.get(k) or 0.0) for k in ['math','physics','chemistry','biology','computer','english','urdu','islamiat','pakstudies']}
    p = normalize_personality(personality)

    stem_score = (s['math']*0.35 + s['physics']*0.25 + s['chemistry']*0.15 + s['computer']*0.25) * (0.7 + 0.3*p.get('openness', 0))
    arts_score = (s['english']*0.5 + s['urdu']*0.3 + s['biology']*0.2) * (0.6 + 0.4*p.get('openness', 0))
    commerce_score = (s['math']*0.5 + s['english']*0.3 + s['computer']*0.2) * (0.6 + 0.4*p.get('conscientiousness', 0))

    scores = {'STEM': max(0, stem_score), 'ARTS': max(0, arts_score), 'COMMERCE': max(0, commerce_score)}
    total = sum(scores.values()) or 1.0
    # normalize to probabilities
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
