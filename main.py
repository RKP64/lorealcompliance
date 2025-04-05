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

# Set up the Streamlit page
st.set_page_config(page_title="Data Cleansing Compliance", layout="wide")

# Custom CSS for styling
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

# Sidebar: Settings, model, and platform selection
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 8000, 3000)

# Choose the e-commerce platform
platform = st.sidebar.selectbox(
    "Select E-commerce Platform:",
    ["Amazon", "Flipkart", "Nykaa"]
)

#########################################################
# Hardcoded Data-Cleansing Guidelines for Each Platform
#########################################################
# These guidelines reflect your screenshots. Adjust as needed.
guidelines_dict = {
    "Amazon": (
        "Naming Conventions:\n"
        "• Weekly: Ecom_Sales_W_IN_CPD_Amazon_YYYYMMDD\n"
        "• Monthly: Ecom_Sales_M_IN_CPD_Amazon_YYYYMMDD\n\n"
        "Data Cleansing Steps:\n"
        "1) Change the file name as advised.\n"
        "2) Make sure there are 3 sheets in the file: FBA, Pantry, Fresh.\n"
        "3) The date/time columns must be present, and the date should appear in the top rows.\n"
        "4) The date/time format must be MM/DD/YYYY HH:MM:SS AM/PM.\n"
        "5) Provide 'Pass' if all steps are met, otherwise 'Fail' with a brief explanation."
    ),
    "Flipkart": (
        "Naming Conventions:\n"
        "• Weekly:\n"
        "   Ecom_Sales_W_IN_CPD_Flipkart-Grooming_YYYYMMDD\n"
        "   Ecom_Sales_W_IN_CPD_Flipkart-Makeup_YYYYMMDD\n"
        "   Ecom_Sales_W_IN_CPD_Flipkart_YYYYMMDD\n"
        "• Monthly:\n"
        "   Ecom_Sales_M_IN_CPD_Flipkart-Grooming_YYYYMMDD\n"
        "   Ecom_Sales_M_IN_CPD_Flipkart-Makeup_YYYYMMDD\n"
        "   Ecom_Sales_M_IN_CPD_Flipkart_YYYYMMDD\n\n"
        "Data Cleansing Steps:\n"
        "1) Change the file name as advised.\n"
        "2) Make sure there are 3 sheets in the file: Grooming, Makeup, Fresh.\n"
        "3) The columns must include: mp_name, brand_name, product_id, asin, mrp, sale_price.\n"
        "4) The date/time format must be MM/DD/YYYY (e.g., 12/01/2021).\n"
        "5) Provide 'Pass' if all steps are met, otherwise 'Fail' with a brief explanation."
    ),
    "Nykaa": (
        "Naming Conventions:\n"
        "• Weekly: Ecom_Sales_W_IN_CPD_Nykaa_YYYYMMDD\n"
        "• Monthly: Ecom_Sales_M_IN_CPD_Nykaa_YYYYMMDD\n\n"
        "Data Cleansing Steps:\n"
        "1) Change the file name as advised.\n"
        "2) Make sure the top rows contain the correct date/time or headers.\n"
        "3) The file may have multiple sheets, but ensure columns are present: sku_code, brand_name, final_sku_code, final_sku_code2, circulated, asin.\n"
        "4) The date/time format should be MM/DD/YYYY (e.g., 12/01/2021).\n"
        "5) Provide 'Pass' if all steps are met, otherwise 'Fail' with a brief explanation."
    )
}

selected_guidelines = guidelines_dict.get(platform, "No guidelines available for the selected platform.")

##############################################
# Section 1 (Optional): Upload Guideline PDFs
##############################################
st.header("Step 1 (Optional): Upload Additional Guideline")
st.write(
    "If you have more detailed PDFs for this platform, upload them here. "
    "."
)
guidelines_files = st.file_uploader("Upload Guideline PDF(s):", type="pdf", accept_multiple_files=True, key="guidelines")

if guidelines_files:
    st.subheader("Processing PDFs...")
    all_guidelines_text = ""
    for file in guidelines_files:
        try:
            pdf_reader = PdfReader(file)
            file_text = "".join([page.extract_text() for page in pdf_reader.pages])
            all_guidelines_text += "\n" + file_text
            st.success(f"Processed guideline PDF: {file.name}")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    # Split PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    guideline_chunks = text_splitter.split_text(all_guidelines_text)
    
    # Create a vector store from the PDF chunks using OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    guidelines_vector_store = FAISS.from_texts(guideline_chunks, embeddings)
    
    st.session_state["guidelines_vector_store"] = guidelines_vector_store
else:
    st.info("No additional PDFs uploaded.")

###############################################################
# Section 2: Upload CSV/Excel & Sample Data for Compliance Check
###############################################################
st.header("Step 2: Upload CSV/Excel for Data Cleansing Check")
data_file = st.file_uploader("Upload Data File (CSV or Excel):", type=["csv", "xlsx", "xls"], key="data")

# Let the user choose how many rows to sample for the compliance check
sample_rows = st.number_input("Number of rows to sample:", min_value=1, value=5, key="sample_rows")

sheet_names_str = "N/A"
if data_file:
    try:
        file_name = data_file.name  # We'll pass the exact file name to the LLM
        file_ext = file_name.lower()
        
        if file_ext.endswith(".csv"):
            df_data = pd.read_csv(data_file)
            sheet_names_str = "CSV file (single sheet)."
        elif file_ext.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(data_file)
            sheet_names = xls.sheet_names
            sheet_names_str = ", ".join(sheet_names)
            # By default, parse the first sheet for sampling
            df_data = xls.parse(sheet_names[0])
        else:
            st.error("Unsupported file type.")
            st.stop()
        
        sample_df = df_data.head(sample_rows)
        csv_text = sample_df.to_string(index=False)
        
        st.success(
            f"Processed data file: {file_name}\n"
            f"Sheet(s): {sheet_names_str}\n"
            f"Showing top {sample_rows} rows below."
        )
        st.dataframe(sample_df)
        
    except Exception as e:
        st.error(f"Error processing data file: {str(e)}")
else:
    st.stop()

###############################################################
# Section 3: Perform Data Cleansing Check (LLM) with Guidelines
###############################################################
if st.button("Check Data Cleansing Compliance"):
    # Optional: retrieve additional context from any uploaded PDFs
    pdf_guidelines_context = ""
    if "guidelines_vector_store" in st.session_state:
        guidelines_vector_store = st.session_state["guidelines_vector_store"]
        retrieval_query = "data cleansing steps, file naming, required sheets, date/time format"
        relevant_chunks = guidelines_vector_store.similarity_search(retrieval_query, k=3)
        pdf_guidelines_context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        # Truncate if too large
        if len(pdf_guidelines_context) > max_context_length:
            pdf_guidelines_context = pdf_guidelines_context[:max_context_length]
    
    # Combine the hardcoded guidelines for the selected platform with PDF guidelines
    combined_guidelines = (
        f"DATA-CLEANSING STEPS FOR {platform.upper()}:\n"
        f"{selected_guidelines}\n\n"
        "ADDITIONAL PDF GUIDELINES (IF ANY):\n"
        f"{pdf_guidelines_context}"
    )
    
    # Build the system prompt
    system_message = {
        "role": "system",
        "content": (
            "You are a compliance officer evaluating data files for an e-commerce platform. "
            "Check if the file adheres to the provided data-cleansing guidelines. "
            "Return 'Pass' if all requirements are met, or 'Fail' with a brief explanation."
        )
    }
    
    # Build the user prompt, including file details and sampled data
    user_message = {
        "role": "user",
        "content": (
            f"Guidelines:\n{combined_guidelines}\n\n"
            f"File Name: {file_name}\n"
            f"Sheet Names: {sheet_names_str}\n"
            f"Sample Rows (top {sample_rows} rows):\n{csv_text}\n\n"
            "Does this file meet all the data-cleansing requirements? "
            "Please respond with Pass or Fail and a short explanation."
        )
    }
    
    try:
        llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
        response = llm.invoke([system_message, user_message], temperature=temperature)
        response_text = response.content.strip()
        
        # Optionally add a greeting
        if not response_text.lower().startswith("hi"):
            response_text = f"Hi, {response_text}"
        
        st.markdown(
            f"<div class='response-box'><b>Response:</b><br>{response_text}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error generating compliance response: {str(e)}")
