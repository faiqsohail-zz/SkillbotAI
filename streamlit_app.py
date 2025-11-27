import os
import io
import time
import json
import uuid
import pandas as pd
import streamlit as st
from supabase import create_client, Client

from ocr_service import run_ocr_on_image, parse_marks_from_ocr_df
from recommender import recommend_field_for_student

# ---------------- Streamlit Layout ----------------
st.set_page_config(page_title="CareerMate - Personalized Career Guidance", layout="wide")

# ---------------- Supabase Init -------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Session Helpers -------------------
def set_session(user):
    st.session_state["user"] = user
    st.session_state["logged_in"] = True
    st.session_state["email"] = user.get("email")
    st.session_state["access_token"] = user.get("access_token")
    st.session_state["refresh_token"] = user.get("refresh_token")

def clear_session():
    for k in ["user", "logged_in", "email", "access_token", "refresh_token"]:
        st.session_state.pop(k, None)

def require_auth():
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to continue.")
        st.stop()
    return st.session_state["user"]

# ---------------- Authentication UI -------------------
st.sidebar.title("Account")

if not st.session_state.get("logged_in"):
    tab_login, tab_signup = st.sidebar.tabs(["Login", "Sign Up"])

    with tab_login:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            try:
                auth_res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if auth_res and auth_res.user:
                    set_session({
                        "email": email,
                        "access_token": auth_res.session.access_token,
                        "refresh_token": auth_res.session.refresh_token
                    })
                    st.success("Logged in successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_signup:
        st.subheader("Sign Up")
        email = st.text_input("New Email", key="signup_email")
        password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Create Account"):
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("Account created! Please login now.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

else:
    st.sidebar.success(f"Logged in as: {st.session_state.get('email')}")
    if st.sidebar.button("Logout"):
        clear_session()
        st.rerun()

# ---------------- MAIN PIPELINE -------------------
st.title("CareerMate – AI-Powered Personalized Career Guidance")

# ------------- Step 1: RIASEC + TCI Tests ------------------
require_auth()

st.header("Step 1 — Take Personality Tests")

# Load question files from repo
try:
    riasec_df = pd.read_csv("questions.csv")
    tci_df = pd.read_csv("tci_questions.csv")
except:
    st.error("questions.csv or tci_questions.csv missing in repo.")
    st.stop()

riasec_answers = {}
tci_answers = {}

with st.expander("RIASEC Test Questions", expanded=True):
    for i, row in riasec_df.iterrows():
        qid = f"riasec_{i}"
        riasec_answers[qid] = st.slider(row["question_text"], 0, 5, 2)

with st.expander("TCI Test Questions", expanded=False):
    for i, row in tci_df.iterrows():
        qid = f"tci_{i}"
        tci_answers[qid] = st.slider(row["question_text"], 0, 5, 2)

if st.button("Submit Test Results"):
    user = st.session_state["email"]

    payload = {
        "user_id": user,
        "riasec_raw": json.dumps(riasec_answers),
        "tci_raw": json.dumps(tci_answers),
    }

    supabase.table("test_results").insert(payload).execute()

    st.success("Test results saved!")

st.markdown("---")

# ------------- Step 2: Ask Personalization ------------------
st.header("Step 2 — Personalization Option")

choice = st.radio(
    "Do you want a personalized recommendation?",
    ["No (quick result)", "Yes (use my profile & marksheet)"]
)

st.markdown("---")

# ------------- Step 3: Profile Creation ------------------
if choice == "Yes (use my profile & marksheet)":
    st.header("Step 3 — Create Your Profile")

    with st.form("profile_form"):
        full_name = st.text_input("Full Name")
        age = st.number_input("Age", 10, 120, 18)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        qualification = st.text_input("Qualification")

        submit_btn = st.form_submit_button("Save Profile")

    if submit_btn:
        user = st.session_state["email"]
        profile_data = {
            "user_id": user,
            "full_name": full_name,
            "age": age,
            "gender": gender,
            "qualification": qualification,
        }
        supabase.table("profiles").upsert(profile_data).execute()
        st.success("Profile saved!")

# ------------- Step 4: Marksheet Upload & OCR ------------------
if choice == "Yes (use my profile & marksheet)":
    st.header("Step 4 — Upload Your Marksheet")

    upload = st.file_uploader("Upload marksheet (jpg, png, pdf)", type=["jpg", "png", "jpeg", "pdf"])

    if upload:
        file_bytes = upload.read()
        live_user = st.session_state["email"]

        # Upload to supabase
        path = f"{live_user}/{uuid.uuid4()}_{upload.name}"
        supabase.storage.from_("marksheets").upload(path, io.BytesIO(file_bytes))
        public_url = supabase.storage.from_("marksheets").get_public_url(path)

        st.success("Uploaded. Running OCR...")

        # OCR
        ocr_df = run_ocr_on_image(file_bytes)
        marks_df = parse_marks_from_ocr_df(ocr_df)

        st.subheader("Extracted Marks")
        st.dataframe(marks_df)

        # Save CSV to storage
        csv_bytes = marks_df.to_csv(index=False).encode()
        csv_path = f"{live_user}/{uuid.uuid4()}_parsed_marks.csv"
        supabase.storage.from_("marksheets").upload(csv_path, io.BytesIO(csv_bytes))
        csv_url = supabase.storage.from_("marksheets").get_public_url(csv_path)

        # Log OCR
        supabase.table("ocr_logs").insert({
            "user_id": live_user,
            "marks_csv_url": csv_url,
            "ocr_confidences": json.dumps(list(ocr_df["conf"].values)),
            "raw_texts": json.dumps(list(ocr_df["text"].values))
        }).execute()

        st.success("OCR Saved!")

        st.session_state["marks_df"] = marks_df

# ------------- Step 5: Final Recommendation ------------------
st.header("Step 5 — Your AI-Powered Career Recommendation")

if choice == "No (quick result)":
    if st.button("Generate Quick Recommendation"):
        personality = {}  # Simplified version
        marks_df = pd.DataFrame([{"Subject": "Default", "Maximum": 100, "Obtained": 50}])
        rec = recommend_field_for_student(marks_df, personality)
        st.json(rec)

else:
    if st.button("Generate Personalized Recommendation"):
        if "marks_df" not in st.session_state:
            st.error("Please upload your marksheet first!")
        else:
            user = st.session_state["email"]

            # Fetch test results
            res = supabase.table("test_results").select("*").eq("user_id", user).order("created_at", desc=True).limit(1).execute()
            personality = res.data[0] if res.data else {}

            rec = recommend_field_for_student(st.session_state["marks_df"], personality)
            st.json(rec)

            supabase.table("recommendations").insert({
                "user_id": user,
                "best_field": rec["best_field"],
                "scores": json.dumps(rec["scores"]),
                "subfields": json.dumps(rec["best_subfields"])
            }).execute()

            st.success("Saved personalized prediction!")
