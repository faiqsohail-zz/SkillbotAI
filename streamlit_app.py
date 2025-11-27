# streamlit_app.py
import os
import io
import time
import uuid
import streamlit as st
import pandas as pd

from supabase import create_client, Client
# Import the refactored modules (we provide these in this reply)
from ocr_service import run_ocr_on_image, parse_marks_from_ocr_df
from recommender import recommend_field_for_student

st.set_page_config(page_title="Marks+Profile App", layout="wide")

# ---------- Supabase config (use st.secrets or env vars) ----------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Helpers ----------
def upload_file_to_supabase(bucket: str, file_bytes: bytes, dest_path: str):
    # dest_path e.g "marksheets/userid_filename.jpg"
    try:
        res = supabase.storage.from_(bucket).upload(dest_path, io.BytesIO(file_bytes))
        # get_public_url may return a dict or str depending on client - handle both
        url_obj = supabase.storage.from_(bucket).get_public_url(dest_path)
        if isinstance(url_obj, dict) and "publicUrl" in url_obj:
            return url_obj["publicUrl"]
        if isinstance(url_obj, dict) and "public_url" in url_obj:
            return url_obj["public_url"]
        return url_obj  # may be string
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def save_recommendation_to_db(user_id, rec):
    try:
        payload = {
            "user_id": user_id,
            "best_field": rec.get("best_field"),
            "scores": rec.get("scores"),
            "subfields": rec.get("subfields"),
            "created_at": pd.Timestamp.now().isoformat()
        }
        supabase.table("recommendations").insert(payload).execute()
    except Exception as e:
        st.warning(f"Could not save recommendation to DB: {e}")


# ---------- UI ----------
st.title("Marks + Profile — Upload & Get Recommendation")

with st.expander("Upload marksheet image"):
    uploaded = st.file_uploader("Upload marksheet (jpg/png/pdf). We'll extract marks", type=["jpg","jpeg","png","pdf"])
    user_id = st.text_input("User ID (your auth id or email)", value="", help="Your supabase auth user_id or identifier")
    if uploaded and user_id:
        file_bytes = uploaded.read()
        # create a deterministic destination
        filename = f"{user_id}/{uuid.uuid4()}_{uploaded.name}"
        url = upload_file_to_supabase("marksheets", file_bytes, filename)
        if url:
            st.success("File uploaded. Running OCR...")
            # Synchronous OCR for now (ideal: push to background job)
            try:
                ocr_df = run_ocr_on_image(file_bytes)  # returns DataFrame with 'Subject','Maximum','Obtained'
                st.write("OCR result preview:")
                st.dataframe(ocr_df)
                # Save CSV to supabase storage
                csv_bytes = ocr_df.to_csv(index=False).encode()
                csv_path = f"{user_id}/{uuid.uuid4()}_marks.csv"
                csv_url = upload_file_to_supabase("marksheets", csv_bytes, csv_path)
                if csv_url:
                    st.success("CSV saved to storage.")
                    # Run recommender
                    # Load personality/test results for user from supabase (example)
                    res = supabase.table("test_results").select("*").eq("user_id", user_id).execute()
                    if res.data and len(res.data) > 0:
                        personality = res.data[-1]  # pick latest
                    else:
                        personality = {}
                    rec = recommend_field_for_student(ocr_df, personality)
                    st.write("Recommendation:")
                    st.json(rec)
                    save_recommendation_to_db(user_id, {
                        "best_field": rec.get("best_field"),
                        "scores": rec.get("scores"),
                        "subfields": rec.get("best_subfields")
                    })
            except Exception as e:
                st.error(f"OCR or recommendation failed: {e}")
