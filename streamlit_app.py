# streamlit_app.py
import os
import io
import json
import uuid
import math
import pandas as pd
import streamlit as st
from supabase import create_client, Client

# Import your OCR and recommender modules (must exist in repo)
from ocr_service import run_ocr_on_image, parse_marks_from_ocr_df
from recommender import recommend_field_for_student

st.set_page_config(page_title="CareerMate — Tests First → Personalize", layout="wide")

# ------------------ Supabase init (use Streamlit secrets) ------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY. Add them to Streamlit Secrets and redeploy.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------ Helpers ------------------
def load_riasec_questions(path="questions.csv"):
    df = pd.read_csv(path)
    # Expect columns: id, question, category
    # Normalize names
    df.columns = [c.strip() for c in df.columns]
    if "question" not in [c.lower() for c in df.columns]:
        # try lowercase match
        raise ValueError("questions.csv must include a 'question' column.")
    # map to known column names
    # find actual column names (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    question_col = col_map.get("question")
    category_col = col_map.get("category") if "category" in col_map else None
    id_col = col_map.get("id") if "id" in col_map else None
    return df, id_col, question_col, category_col

def load_tci_questions(path="tci_questions.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    question_col = col_map.get("question") or list(df.columns)[0]
    trait_col = col_map.get("trait") or (list(df.columns)[1] if len(df.columns) > 1 else None)
    if not question_col:
        raise ValueError("tci_questions.csv must include a question column.")
    return df, question_col, trait_col

def aggregate_riasec(riasec_answers, questions_df, qid_prefix="riasec_"):
    """
    riasec_answers: dict keyed by qid like 'riasec_<id or idx>' mapped to 0..5 values
    questions_df: original questions.csv dataframe (contains category column like R/I/A/S/E/C)
    Returns dict of aggregated R,I,A,S,E,C mean scores (0..1)
    """
    # build mapping from qid -> category
    categories = {}
    # detect id or index mapping
    # our ui used qid = f"riasec_{row['id']}" if id exists else riasec_{idx}
    id_col = next((c for c in questions_df.columns if c.strip().lower() == "id"), None)
    q_col = next((c for c in questions_df.columns if c.strip().lower() == "question"), None)
    cat_col = next((c for c in questions_df.columns if c.strip().lower() == "category"), None)
    # iterate rows to map qid used by UI to categories
    for idx, row in questions_df.iterrows():
        unique = str(row[id_col]) if id_col else str(idx)
        qid = f"{qid_prefix}{unique}"
        cat = None
        if cat_col:
            cat = str(row[cat_col]).strip().upper()
        categories[qid] = cat
    # collect answers per category
    cat_vals = {}
    for qid, val in riasec_answers.items():
        cat = categories.get(qid)
        if not cat:
            continue
        try:
            v = float(val)
        except:
            continue
        cat_vals.setdefault(cat, []).append(v)
    # compute mean normalized to 0..1 (answers were 0..5)
    agg = {}
    for cat, vals in cat_vals.items():
        if vals:
            agg[cat] = sum(vals) / (len(vals) * 5.0)
    # ensure all R/I/A/S/E/C present (0 if missing)
    for c in ["R","I","A","S","E","C"]:
        agg.setdefault(c, 0.0)
    return agg

def aggregate_tci(tci_answers, tci_df, qid_prefix="tci_"):
    """
    tci_answers: dict keyed by qid 'tci_<idx>' -> 0..5
    tci_df: tci_questions.csv with column trait (e.g. 'Novelty Seeking', 'Harm Avoidance')
    returns dict trait -> mean(0..1)
    """
    trait_map = {}
    # map indices/ids to trait names
    # our UI used qid = f"tci_{i}" where i is row index in tci_df
    for i, row in tci_df.iterrows():
        trait = None
        # choose trait column
        for c in tci_df.columns:
            if c.strip().lower() == "trait":
                trait = row[c]
                break
        if not trait:
            # fallback: try second column
            cols = list(tci_df.columns)
            trait = row[cols[1]] if len(cols) > 1 else "trait"
        qid = f"{qid_prefix}{i}"
        trait_map[qid] = str(trait).strip()
    # aggregate answers
    t_vals = {}
    for qid, val in tci_answers.items():
        trait = trait_map.get(qid)
        if not trait:
            continue
        try:
            v = float(val)
        except:
            continue
        t_vals.setdefault(trait, []).append(v)
    agg = {}
    for trait, vals in t_vals.items():
        if vals:
            agg[trait] = sum(vals) / (len(vals) * 5.0)
    return agg

def save_test_results_to_db(user_id, riasec_agg, tci_agg, raw_riasec, raw_tci):
    payload = {
        "user_id": user_id,
        "riasec_R": float(riasec_agg.get("R", 0.0)),
        "riasec_I": float(riasec_agg.get("I", 0.0)),
        "riasec_A": float(riasec_agg.get("A", 0.0)),
        "riasec_S": float(riasec_agg.get("S", 0.0)),
        "riasec_E": float(riasec_agg.get("E", 0.0)),
        "riasec_C": float(riasec_agg.get("C", 0.0)),
        "riasec_raw": json.dumps(raw_riasec or {}),
        "tci_agg": json.dumps(tci_agg or {}),
        "tci_raw": json.dumps(raw_tci or {})
    }
    supabase.table("test_results").insert(payload).execute()

def upload_bytes_to_bucket(bucket: str, path: str, bytes_data: bytes):
    try:
        res = supabase.storage.from_(bucket).upload(path, io.BytesIO(bytes_data))
    except Exception as e:
        # upload might raise if file exists; try replace via update (client behavior varies)
        try:
            supabase.storage.from_(bucket).remove([path])
            supabase.storage.from_(bucket).upload(path, io.BytesIO(bytes_data))
        except Exception:
            raise
    # get public url result handling dict/string variants
    url_obj = supabase.storage.from_(bucket).get_public_url(path)
    if isinstance(url_obj, dict):
        return url_obj.get("publicUrl") or url_obj.get("public_url") or url_obj.get("publicURL")
    return url_obj

# ------------------ State initialization ------------------
if "riasec_answers" not in st.session_state:
    st.session_state["riasec_answers"] = {}
if "tci_answers" not in st.session_state:
    st.session_state["tci_answers"] = {}
if "want_personal" not in st.session_state:
    st.session_state["want_personal"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None  # will contain supabase user dict (email, tokens)
if "marks_df" not in st.session_state:
    st.session_state["marks_df"] = None

# ------------------ UI: Intro ------------------
st.title("CareerMate — Tests First, Then Personalize")

st.markdown(
    """
    **Flow:** 1) Take RIASEC & TCI tests → 2) Do you want a *personalized* recommendation? → 
    3) If yes: Sign up / Login + create profile → 4) Upload marksheet → 5) Get final career prediction.
    """
)

# ------------------ Step 1: Tests (anonymous) ------------------
st.header("Step 1 — Take the tests (no signup required)")

# Load questions safely
try:
    riasec_df, riasec_id_col, riasec_qcol, riasec_cat_col = load_riasec_questions("questions.csv")
except Exception as e:
    st.error(f"Could not load RIASEC questions: {e}")
    st.stop()

try:
    tci_df, tci_qcol, tci_trait_col = load_tci_questions("tci_questions.csv")
except Exception as e:
    st.error(f"Could not load TCI questions: {e}")
    st.stop()

# Render RIASEC
with st.expander("RIASEC Test (personality dimensions: Realistic, Investigative, Artistic, Social, Enterprising, Conventional)", expanded=True):
    for idx, row in riasec_df.iterrows():
        # qid uses id column if present, else use index
        unique = str(row[riasec_id_col]) if riasec_id_col else str(idx)
        qid = f"riasec_{unique}"
        prompt = str(row[riasec_qcol])
        # read previous answer from session if available
        prev = st.session_state["riasec_answers"].get(qid, 2)
        st.session_state["riasec_answers"][qid] = st.slider(prompt, 0, 5, int(prev), key=qid)

# Render TCI
with st.expander("TCI Test (temperament/character traits)", expanded=False):
    for idx, row in tci_df.iterrows():
        qid = f"tci_{idx}"
        prompt = str(row[tci_qcol])
        prev = st.session_state["tci_answers"].get(qid, 2)
        st.session_state["tci_answers"][qid] = st.slider(prompt, 0, 5, int(prev), key=qid)

# Submit test answers (store in session, compute aggregated summary shown)
if st.button("Submit test answers"):
    # compute aggregated scores
    riasec_agg = aggregate_riasec(st.session_state["riasec_answers"], riasec_df)
    tci_agg = aggregate_tci(st.session_state["tci_answers"], tci_df)
    st.session_state["latest_riasec_agg"] = riasec_agg
    st.session_state["latest_tci_agg"] = tci_agg

    st.success("Test answers recorded in session.")
    st.write("RIASEC aggregated (0..1):", riasec_agg)
    st.write("TCI aggregated (0..1):", tci_agg)

    # ask personalization choice next
    st.session_state["want_personal"] = st.radio("Would you like a personalized recommendation using profile and marksheet?", ("No — quick result", "Yes — personalized (requires signup)"))

# If user didn't yet submit, give an option to continue quickly
if "latest_riasec_agg" not in st.session_state:
    st.info("Submit test answers to see the personalization option or generate a quick result.")

# ------------------ Quick (non-personalized) result ------------------
st.markdown("---")
st.header("Quick result (no signup)")
if st.button("Get quick recommendation (uses only tests)"):
    # Use aggregated tests if available, else use raw defaults
    riasec_agg = st.session_state.get("latest_riasec_agg", {c:0.0 for c in ["R","I","A","S","E","C"]})
    tci_agg = st.session_state.get("latest_tci_agg", {})
    # Build a minimal personality record that recommender expects (keys will vary)
    personality_record = {**riasec_agg, **tci_agg}
    # fallback minimal marks_df (no marks provided)
    marks_df = pd.DataFrame([{"Subject":"_default_","Maximum":100,"Obtained":50}])
    rec = recommend_field_for_student(marks_df, personality_record)
    st.subheader("Quick Recommendation")
    st.json(rec)

# ------------------ If user opted for personalization, require auth now ------------------
st.markdown("---")
st.header("Personalized Recommendation (signup required)")

# Auth UI (Sign up / Login)
if st.session_state.get("want_personal") == "Yes — personalized (requires signup)":
    # simple auth flow using Supabase Auth (email/password)
    if not st.session_state.get("user"):
        st.info("To get a personalized result please sign up or log in.")

        auth_col1, auth_col2 = st.columns(2)

        with auth_col1:
            st.subheader("Sign up")
            su_email = st.text_input("Email (sign up)", key="su_email")
            su_password = st.text_input("Password (sign up)", type="password", key="su_pass")
            if st.button("Create account", key="signup_btn"):
                try:
                    supabase.auth.sign_up({"email": su_email, "password": su_password})
                    st.success("Account created. Please log in using the login form.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")

        with auth_col2:
            st.subheader("Log in")
            li_email = st.text_input("Email (login)", key="li_email")
            li_password = st.text_input("Password (login)", type="password", key="li_pass")
            if st.button("Log in", key="login_btn"):
                try:
                    auth_res = supabase.auth.sign_in_with_password({"email": li_email, "password": li_password})
                    # on success, store user minimal info in session
                    user_info = {
                        "email": li_email,
                        "access_token": getattr(auth_res.session, "access_token", None) if auth_res else None,
                        "refresh_token": getattr(auth_res.session, "refresh_token", None) if auth_res else None
                    }
                    st.session_state["user"] = user_info
                    st.success("Logged in.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
    else:
        st.success(f"Signed in as: {st.session_state['user'].get('email')}")

    # After login, allow profile creation and upload flow
    if st.session_state.get("user"):
        user_email = st.session_state["user"].get("email")
        st.subheader("Profile (create / update)")
        with st.form("profile_form"):
            full_name = st.text_input("Full name")
            age = st.number_input("Age", 10, 120, 18)
            gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
            qualification = st.text_input("Qualification")
            save_profile = st.form_submit_button("Save profile")
            if save_profile:
                payload = {"user_id": user_email, "full_name": full_name, "age": int(age), "gender": gender, "qualification": qualification}
                try:
                    supabase.table("profiles").upsert(payload, on_conflict="user_id").execute()
                    st.success("Profile saved.")
                except Exception as e:
                    st.error(f"Could not save profile: {e}")

        st.markdown("---")
        st.subheader("Upload your latest marksheet (image/PDF)")
        uploaded = st.file_uploader("Upload marksheet file", type=["jpg","jpeg","png","pdf"])
        if uploaded:
            file_bytes = uploaded.read()
            # store raw image/pdf to storage under user folder
            dest_path = f"{user_email}/{uuid.uuid4()}_{uploaded.name}"
            try:
                csv_url = upload_bytes_to_bucket("marksheets", dest_path, file_bytes)
            except Exception as e:
                st.error(f"Upload failed: {e}")
                csv_url = None

            st.success("Uploaded to storage. Running OCR...")
            # run OCR and parse marks
            try:
                ocr_df = run_ocr_on_image(file_bytes)
                parsed_marks_df = parse_marks_from_ocr_df(ocr_df)
                st.subheader("Parsed marks (preview)")
                st.dataframe(parsed_marks_df)
                # save parsed CSV back to storage
                csv_bytes = parsed_marks_df.to_csv(index=False).encode()
                parsed_path = f"{user_email}/{uuid.uuid4()}_parsed_marks.csv"
                parsed_url = upload_bytes_to_bucket("marksheets", parsed_path, csv_bytes)
                st.success("Parsed CSV saved to storage.")
                # record OCR log
                try:
                    supabase.table("ocr_logs").insert({
                        "user_id": user_email,
                        "marks_csv_url": parsed_url,
                        "ocr_confidences": json.dumps(list(ocr_df["conf"].astype(float).values)) if "conf" in ocr_df.columns else json.dumps([]),
                        "raw_texts": json.dumps(list(ocr_df["text"].astype(str).values))
                    }).execute()
                except Exception as e:
                    st.warning(f"Could not insert ocr_log: {e}")
                # save parsed marks in session for final prediction
                st.session_state["marks_df"] = parsed_marks_df
            except Exception as e:
                st.error(f"OCR / parsing failed: {e}")

        st.markdown("---")
        st.subheader("Generate final personalized recommendation")
        if st.button("Generate personalized recommendation"):
            # Ensure tests are aggregated and saved
            riasec_agg = st.session_state.get("latest_riasec_agg") or aggregate_riasec(st.session_state["riasec_answers"], riasec_df)
            tci_agg = st.session_state.get("latest_tci_agg") or aggregate_tci(st.session_state["tci_answers"], tci_df)
            # Persist test results to DB
            try:
                save_test_results_to_db(user_email, riasec_agg, tci_agg, st.session_state.get("riasec_answers"), st.session_state.get("tci_answers"))
            except Exception as e:
                st.warning(f"Could not save test results to DB: {e}")
            # prepare personality record for recommender (flatten)
            personality_record = {**riasec_agg}
            personality_record.update(tci_agg)
            # marks_df either from session (uploaded) or fallback minimal
            marks_df = st.session_state.get("marks_df") or pd.DataFrame([{"Subject":"_default_","Maximum":100,"Obtained":50}])
            rec = recommend_field_for_student(marks_df, personality_record)
            st.subheader("Personalized Recommendation")
            st.json(rec)
            # save recommendation to DB
            try:
                supabase.table("recommendations").insert({
                    "user_id": user_email,
                    "best_field": rec.get("best_field"),
                    "scores": json.dumps(rec.get("scores")),
                    "subfields": json.dumps(rec.get("best_subfields"))
                }).execute()
                st.success("Recommendation saved.")
            except Exception as e:
                st.warning(f"Could not save recommendation to DB: {e}")

# ------------------ End of app ------------------
