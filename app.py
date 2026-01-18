import threading
import signal
import os
import re
import json
import streamlit as st
from datetime import datetime
from src.models import PaperAnalysis
from src.utils import extract_paper_sections

# ==========================================
# 0. THE MONKEY PATCH (Fixes Threading Crash)
# ==========================================
# Store the original signal function
_original_signal = signal.signal

# Define a safe version that ignores errors if we aren't in the main thread
def _safe_signal_handler(signalnum, handler):
    try:
        # Only try to register signals if we are in the main thread
        if threading.current_thread() is threading.main_thread():
            return _original_signal(signalnum, handler)
    except ValueError:
        # If we get "signal only works in main thread", just swallow the error
        pass

# Apply the patch
signal.signal = _safe_signal_handler

# Also try to disable telemetry via Env Var as a backup
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

from src.utils import process_zip_file, generate_docx_report
from src.crew import create_nutrition_crew
from src.models import PaperAnalysis # Import the model for validation



# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Nutrition Science Analyzer", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.text_input("Ollama Model Name", value="qwen3")
    base_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    st.info("Ensure Ollama is running.")

# ==========================================
# 2. MAIN UI LOGIC
# ==========================================

st.title("🧬 Nutrition Science Validator")
st.markdown("""
**Upload a ZIP file containing PDF scientific papers.** 
This AI Crew will unpack them, analyze them for study design flaws, funding bias, and statistical tricks.
""")

# Initialize Session State
if "results" not in st.session_state:
    st.session_state.results = []

uploaded_zip = st.file_uploader("Upload ZIP of PDFs", type="zip")
start_analysis = st.button("Start Analysis")

if uploaded_zip and start_analysis:

    # --- FIX 1: CLEAR PREVIOUS RESULTS ---
    st.session_state.results = []

    
    # Step 1: Unpack
    with st.spinner("Unpacking and parsing PDFs..."):
        papers = process_zip_file(uploaded_zip)
        st.success(f"Found {len(papers)} papers.")

    # Step 2: Analyze
    progress_bar = st.progress(0)
    
    for idx, paper in enumerate(papers):

        st.info(f"Analyzing: {paper['filename']}...")

        sections = paper["sections"]
        if sections["methods"] is None:
            st.warning(f"{paper['filename']}: Methods section missing — evidence automatically downgraded.")
        
        # Create Crew with dynamic user config
        crew = create_nutrition_crew(sections, model_name, base_url)

        
        try:
            crew_output = crew.kickoff()
            
            # Extract Pydantic model safely
            if hasattr(crew_output, 'pydantic') and crew_output.pydantic:
                analysis_result = crew_output.pydantic
            elif hasattr(crew_output, 'raw'):
                 st.warning(f"Structured data extraction failed for {paper['filename']}. Raw output: {crew_output.raw[:100]}...")
                 continue
            else:
                analysis_result = crew_output

            st.session_state.results.append(analysis_result)
        
        except Exception as e:
            st.error(f"Error analyzing {paper['filename']}: {e}")

        progress_bar.progress((idx + 1) / len(papers))

# ==========================================
# 3. DASHBOARD DISPLAY
# ==========================================

if st.session_state.results:
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Analysis Report")
    with col2:
        # Generate the DOCX file in memory
        docx_file = generate_docx_report(st.session_state.results)
        
        # Generate timestamped filename (e.g., nutrition_report_2026-01-07_14-30-05.docx)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"nutrition_report_{timestamp}.docx"
        
        st.download_button(
            label="📥 Download Report (.docx)",
            data=docx_file,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    # ------------------------------------

    for res in st.session_state.results:
        # Dynamic color coding for trust score
        score_color = "green" if res.trust_score >= 8 else "orange" if res.trust_score >= 5 else "red"
        
        with st.expander(f"{res.title} (Trust Score: {res.trust_score}/10)"):
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Evidence Level", value=res.evidence_level)
                st.metric(label="Trust Score", value=f"{res.trust_score}/10")
                if res.has_conflict_of_interest:
                    st.error(f"⚠️ Conflict Detected: {res.funding_source}")
                    st.caption(res.coi_notes)
                else:
                    st.success("No Obvious COI Detected")
            
            with col2:
                st.markdown(f"**Type:** {res.paper_type}")
                st.markdown(f"**Conclusion:** {res.conclusion_summary}")
                
                st.markdown("---")
                st.markdown("### 🔬 Methodology Check")
                st.write(f"**Control Group:** {res.control_group_quality}")
                st.write(f"**Intervention:** {res.intervention_details}")
                st.write(f"**Confounders:** {res.confounding_factors}")
                
                st.markdown("---")
                st.markdown("### 📈 Statistical Check")
                st.write(f"**Endpoints:** {res.endpoints}")
                st.write(f"**Risk Reporting:** {res.risk_type_reported}")
                st.write(f"**Significance:** {res.statistical_significance}")
                
                st.markdown("---")
                st.info(f"**Final Verdict:** {res.final_verdict}")