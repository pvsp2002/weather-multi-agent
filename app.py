# ----------------------------
# app.py — Streamlit Multi-Agent Weather (Phi PDF Agent)
# ----------------------------

# --- Step 0: Load environment variables ---
from dotenv import load_dotenv
import os
from pathlib import Path
import re

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env!")

if not os.getenv("WEATHER_API_KEY"):
    raise ValueError("WEATHER_API_KEY not found in .env!")

# --- Step 1: Imports ---
import streamlit as st
import nbformat

# --- Step 2: Helper function to run notebooks ---
def run_notebook(notebook_path: str):
    """
    Executes a Jupyter notebook and returns its global namespace.
    """
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    notebook_globals = {}

    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, notebook_globals)

    return notebook_globals

# --- Step 3: Load agents from notebooks ---
weather_ns = run_notebook("agents/weather_agent.ipynb")
weather_agent = weather_ns["weather_agent"]

content_ns = run_notebook("agents/content_agent.ipynb")
content_agent = content_ns["content_agent"]

pdf_ns = run_notebook("agents/pdf_agent.ipynb")
pdf_agent = pdf_ns["pdf_agent"]

# ----------------------------
# --- Step 4: Streamlit UI ---
# ----------------------------
st.set_page_config(
    page_title="Weather Multi-Agent App",
    page_icon="☀️",
    layout="centered"
)

st.title("🌤️ Weather Multi-Agent System")
st.write("Enter a city to get the weather report and download it as a PDF.")

city = st.text_input("City Name", value="Delhi")

if st.button("Get Weather Report") and city:
    with st.spinner("Running agents..."):
        try:
            # 1️⃣ Weather Agent
            weather_output = weather_agent.run(
                f"Get current weather for {city}"
            ).content

            # 2️⃣ Content Agent
            paragraph_output = content_agent.run(weather_output).content

            st.subheader("Weather Summary")
            st.write(paragraph_output)

            # 3️⃣ PDF Agent (Phi)
            run_response = pdf_agent.run(
                paragraph_output,
                city_name=city
            )

            raw_output = run_response.content.strip()

            # ---- IMPORTANT PART ----
            # Phi wraps tool output in text → extract the actual PDF path
            match = re.search(
                r"(weather_reports[\\/].+?\.pdf)",
                raw_output
            )

            if not match:
                raise ValueError(f"Could not extract PDF path from:\n{raw_output}")

            pdf_file_path = Path(match.group(1)).resolve()

            # Offer download
            if pdf_file_path.exists():
                with pdf_file_path.open("rb") as f:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=f,
                        file_name=pdf_file_path.name,
                        mime="application/pdf"
                    )
                st.success("Done! PDF generated successfully.")
            else:
                st.error(f"PDF file not found at: {pdf_file_path}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
