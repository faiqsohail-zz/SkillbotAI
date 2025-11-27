# streamlit_app.py
import os
import io
import uuid
import time
import json
import pandas as pd
import streamlit as st
from supabase import create_client, Client

from ocr_service import run_ocr_on_image, parse_marks_from_ocr_df
from recommender import recommend_field_for_student

st.set_page_config(page_title="CareerMate — Tests → Personalize → Predict", layout="wide")

# ---------------- Supabase config (Streamlit Cloud secrets or env vars) ----------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase secrets. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Helpers ----------------
def current_user_id():
    return st.session_state.get("user_id")

def create_profile_in_db(user_id, profile):
    try:
        payload = {**profile, "user_id": user_id}
        supabase.table("profiles").upsert(payload, on_conflict="user_id").execute()
        return True
    except Exception as e:
        st.error(f"Could not save profile: {e}")
        return False

def save_test_results(user_id, riasec_dict, tci_dict):
    try:
        payload = {
            "user_id": user_id,
            "riasec_R": riasec_dict.get("R"),
            "riasec_I": riasec_dict.get("I"),
            "riasec_A": riasec_dict.get("A"),
            "riasec_S": riasec_dict.get("S"),
            "riasec_E": riasec_dict.get("E"),
            "riasec_C": riasec_dict.get("C"),
            "tci_data": json.dumps(tci_dict)
        }
        supabase.table("test_results").insert(payload).execute()
        st.success("Test results saved.")
    except Exception as e:
        st.error(f"Failed to save test results: {e}")

def upload_file_to_supabase(bucket: str, file_bytes: bytes, dest_path: str):
    try:
        res = supabase.storage.from_(bucket).upload(dest_path, io.BytesIO(file_bytes))
        url_obj = supabase.storage.from_(bucket).get_public_url(dest_path)
        if isinstance(url_obj, dict):
            # Supabase python client sometimes returns dict
            return url_obj.get("publicUrl") or url_obj.get("public_url") or url_obj.get("publicURL")
        return url_obj
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def save_ocr_log(user_id, marks_csv_url, confidences, raw_texts):
    try:
        payload = {
            "user_id": user_id,
            "marks_csv_url": marks_csv_url,
            "ocr_confidences": json.dumps(confidences),
            "raw_texts": json.dumps(raw_texts)
        }
        supabase.table("ocr_logs").insert(payload).execute()
    except Exception as e:
        st.warning(f"Could not save ocr log: {e}")

def save_recommendation(user_id, rec):
    try:
        payload = {
            "user_id": user_id,
            "best_field": rec.get("best_field"),
            "scores": json.dumps(rec.get("scores")),
            "subfields": json.dumps(rec.get("best_subfields"))
        }
        supabase.table("recommendations").insert(payload).execute()
    except Exception as e:
        st.warning(f"Could not save recommendation: {e}")

# ---------------- UI: Authentication (lightweight) ----------------
st.header("CareerMate — Find recommended careers")

with st.sidebar:
    st.subheader("Account")
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign up (create user id)"):
            uid = st.text_input("Choose a unique user id (email or uid)", key="signup_uid")
            if uid:
                st.session_state["user_id"] = uid
                st.success("User id set in session. Create profile next.")
    with col2:
        if st.button("Use existing user id"):
            uid = st.text_input("Enter your user id", key="login_uid")
            if uid:
                st.session_state["user_id"] = uid
                st.success("User id set in session.")

    st.markdown("---")
    st.write("Current user id:")
    st.code(str(st.session_state.get("user_id")))

# ---------------- Step 1: User takes tests ----------------
st.subheader("Step 1 — Take tests (RIASEC & TCI)")

colA, colB = st.columns(2)
with colA:
    st.markdown("**RIASEC Test**")
    # Load questions from repo (questions.csv must be in repo root)
    try:
        qdf = pd.read_csv("questions.csv")
    except Exception:
        # fallback empty
        qdf = pd.DataFrame()
    riasec_answers = {}
    if not qdf.empty:
        for idx, row in qdf.iterrows():
            qid = f"riasec_{idx}"
            ans = st.slider(row.get("question_text", f"Q{idx}"), 0, 5, 2, key=qid)
            # assume each row has mapping to R/I/A/S/E/C columns; for simplicity store as dict
            riasec_answers[qid] = ans
    else:
        st.info("questions.csv not found in repo. Add it to present RIASEC questions.")

with colB:
    st.markdown("**TCI Test**")
    try:
        tdf = pd.read_csv("tci_questions.csv")
    except Exception:
        tdf = pd.DataFrame()
    tci_answers = {}
    if not tdf.empty:
        for idx, row in tdf.iterrows():
            qid = f"tci_{idx}"
            ans = st.slider(row.get("question_text", f"Q{idx}"), 0, 5, 2, key=qid)
            tci_answers[qid] = ans
    else:
        st.info("tci_questions.csv not found in repo. Add it to present TCI questions.")

if st.button("Submit test answers"):
    if not st.session_state.get("user_id"):
        st.error("Set your user id in sidebar before submitting tests.")
    else:
        # Aggregate tests into simplified RIASEC/TCl usable form
        # (here we assume questions.csv/tci_questions.csv contain mapping columns)
        # We'll store raw answers, and later your processing code can map to factors
        user_id = st.session_state["user_id"]
        st.session_state["latest_riasec_answers"] = riasec_answers
        st.session_state["latest_tci_answers"] = tci_answers
        save_test_results(user_id, {"R": None, "I": None, "A": None, "S": None, "E": None, "C": None}, {})
        st.success("Answers saved temporarily to session. We saved a record in DB (raw mapping deferred).")

# ---------------- Step 2: Ask if user wants personalized results ----------------
st.subheader("Step 2 — Personalization (optional)")

if st.session_state.get("latest_riasec_answers") or st.session_state.get("latest_tci_answers"):
    want_personal = st.radio("Would you like a more personalized prediction using your profile & marksheet?", ("No (quick)", "Yes (personalized)"))
else:
    want_personal = None
    st.info("Complete tests first to see personalization option.")

# ---------------- Step 3: Signup / Create Profile ----------------
st.subheader("Step 3 — Create or update your profile (required for personalized result)")

if not st.session_state.get("user_id"):
    st.warning("Create or set your user id in the sidebar first.")
else:
    with st.form("profile_form"):
        name = st.text_input("Full name")
        age = st.number_input("Age", min_value=10, max_value=120, value=18)
        gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        qualification = st.text_input("Highest qualification")
        submit_profile = st.form_submit_button("Save profile")
        if submit_profile:
            profile = {"full_name": name, "age": int(age), "gender": gender, "qualification": qualification}
            ok = create_profile_in_db(st.session_state["user_id"], profile)
            if ok:
                st.success("Profile saved.")

# ---------------- Step 4: Upload latest marksheet ----------------
st.subheader("Step 4 — Upload latest marksheet (image/pdf)")

uploaded = st.file_uploader("Upload marksheet now (we'll extract subjects and marks). This is required for personalized prediction.", type=["jpg","jpeg","png","pdf"])
if uploaded and st.session_state.get("user_id"):
    user_id = st.session_state["user_id"]
    file_bytes = uploaded.read()
    dest = f"{user_id}/{uuid.uuid4()}_{uploaded.name}"
    url = upload_file_to_supabase("marksheets", file_bytes, dest)
    if url:
        st.success("File uploaded to storage. Running OCR & extracting marks...")
        # Run OCR (synchronous)
        ocr_df = run_ocr_on_image(file_bytes)
        st.write("Raw OCR rows (sample):")
        st.dataframe(ocr_df.head(30))
        marks_df = parse_marks_from_ocr_df(ocr_df)
        st.write("Parsed marks (Subject | Maximum | Obtained):")
        st.dataframe(marks_df)
        # Save CSV to storage
        csv_bytes = marks_df.to_csv(index=False).encode()
        csv_path = f"{user_id}/{uuid.uuid4()}_marks.csv"
        csv_url = upload_file_to_supabase("marksheets", csv_bytes, csv_path)
        # Save ocr log
        confidences = list(ocr_df['conf'].astype(float).values) if 'conf' in ocr_df.columns else []
        raw_texts = list(ocr_df['text'].astype(str).values)
        save_ocr_log(user_id, csv_url, confidences, raw_texts)
        st.success("Marks parsed and saved.")
        st.session_state["latest_marks_df"] = marks_df

# ---------------- Step 5: Final prediction (only if user agreed) ----------------
st.subheader("Step 5 — Get career prediction")

if want_personal == "Yes (personalized)":
    if not st.session_state.get("user_id"):
        st.error("Set user id first.")
    elif "latest_marks_df" not in st.session_state:
        st.warning("Upload marksheet to get personalized prediction.")
    else:
        if st.button("Generate personalized prediction"):
            user_id = st.session_state["user_id"]
            marks_df = st.session_state["latest_marks_df"]
            # fetch latest saved test_results for this user
            res = supabase.table("test_results").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
            personality_record = {}
            if res.data and len(res.data) > 0:
                personality_record = res.data[0]
            rec = recommend_field_for_student(marks_df, personality_record)
            st.success("Personalized prediction ready:")
            st.json(rec)
            save_recommendation(user_id, rec)
else:
    if st.button("Generate quick prediction (no marksheet)"):
        # quick prediction uses only tests (if available) or default
        user_id = st.session_state.get("user_id")
        marks_df = st.DataFrame([{"Subject":"default","Maximum":100,"Obtained":50}])  # fallback
        res = supabase.table("test_results").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        personality_record = {}
        if res.data and len(res.data) > 0:
            personality_record = res.data[0]
        rec = recommend_field_for_student(marks_df, personality_record)
        st.json(rec)
        if user_id:
            save_recommendation(user_id, rec)
