import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load API keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize Streamlit page
st.set_page_config(page_title="Compliance Assistant", layout="wide")

# Custom CSS for a sleek design
st.markdown(
    """
    <style>
    body { background-color: #f7f9fc; }
    .stButton>button {
        background-color: #4caf50; color: white; border: none;
        border-radius: 5px; padding: 8px 15px; font-size: 14px; margin: 5px; cursor: pointer;
    }
    .stButton>button:hover { background-color: #45a049; }
    .response-box {
        background-color: #ffffff; border-radius: 10px; padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif; font-size: 16px; color: #333333;
        line-height: 1.6; margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Settings
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768", "gemma2-9b-it","llama-3.1-8b-instant"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 25000, 3000)

##############################
# Section 1: Guidelines Upload
##############################
st.header("Step 1: Upload Guideline PDFs")
guidelines_files = st.file_uploader("Upload Guideline PDF(s):", type="pdf", accept_multiple_files=True, key="guidelines")

if guidelines_files:
    st.subheader("Processing Guidelines...")
    all_guidelines_text = ""
    for file in guidelines_files:
        try:
            pdf_reader = PdfReader(file)
            file_text = "".join([page.extract_text() for page in pdf_reader.pages])
            all_guidelines_text += "\n" + file_text
            st.success(f"Processed guideline: {file.name}")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    # Split guideline text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    guideline_chunks = text_splitter.split_text(all_guidelines_text)
    
    # Create a vector store from the guideline chunks using OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    guidelines_vector_store = FAISS.from_texts(guideline_chunks, embeddings)
    
    # Store the guidelines vector store in session state for future use
    st.session_state["guidelines_vector_store"] = guidelines_vector_store
else:
    st.info("Please upload your guideline PDFs. They will be stored for future compliance checks.")

#####################################
# Section 2: CSV/Excel Data Upload
#####################################
st.header("Step 2: Upload Daily CSV/Excel File for Compliance Check")
data_file = st.file_uploader("Upload Data File (CSV or Excel):", type=["csv", "xlsx", "xls"], key="data")

if data_file:
    try:
        file_name = data_file.name.lower()
        if file_name.endswith(".csv"):
            df_data = pd.read_csv(data_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df_data = pd.read_excel(data_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
        csv_text = df_data.to_string(index=False)
        st.success(f"Processed data file: {data_file.name}")
    except Exception as e:
        st.error(f"Error processing data file: {str(e)}")

#############################
# Compliance Check and LLM Call
#############################
if st.button("Check Compliance"):
    if "guidelines_vector_store" not in st.session_state:
        st.error("Please upload and process guideline PDFs first.")
    elif not data_file:
        st.error("Please upload a CSV/Excel file for compliance check.")
    else:
        # Retrieve the stored guidelines vector store
        guidelines_vector_store = st.session_state["guidelines_vector_store"]
        
        # Use the CSV/Excel content as a query to retrieve relevant guideline chunks
        relevant_guideline_chunks = guidelines_vector_store.similarity_search(csv_text, k=3)
        guidelines_context = " ".join([chunk.page_content for chunk in relevant_guideline_chunks])
        if len(guidelines_context) > max_context_length:
            guidelines_context = guidelines_context[:max_context_length]
        
        # Build the prompt for the LLM
        system_message = {
            "role": "system",
            "content": (
                "You are a compliance officer at Bank of Baroda. Evaluate the uploaded CSV/Excel data "
                "against the provided guidelines context. Return 'Pass' if the data complies with all guidelines, "
                "or 'Fail' if it does not, along with a brief explanation."
            )
        }
        user_message = {
            "role": "user",
            "content": (
                f"Guidelines Context:\n{guidelines_context}\n\n"
                f"CSV/Excel Data:\n{csv_text}\n\n"
                "Based on the above, assess compliance and provide a pass/fail result with a brief explanation."
            )
        }
        
        try:
            llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
            response = llm.invoke([system_message, user_message], temperature=temperature)
            response_text = response.content.strip()
            if not response_text.lower().startswith("hi"):
                response_text = f"Hi, {response_text}"
            st.markdown(f"<div class='response-box'><b>Response:</b><br>{response_text}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating compliance response: {str(e)}")

