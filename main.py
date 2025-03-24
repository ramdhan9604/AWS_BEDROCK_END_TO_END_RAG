import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import base64
load_dotenv()



aws_access_key_id = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = os.getenv("REGION_NAME")



prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

bedrock_embeddings = BedrockEmbeddings(
    model_id = "amazon.titan-embed-text-v2:0",
    client = bedrock
)

def get_documents():
    loader=PyPDFDirectoryLoader("pdf_data")
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=500)
    
    docs=text_splitter.split_documents(documents)
    return docs




def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")



def get_llm():
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,
                model_kwargs={'max_tokens':512})
    
    return llm


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']



def set_background():
    """Sets a background image for the Streamlit app."""
    image_url = "https://source.unsplash.com/1600x900/?technology,ai"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_background()
    
    # Stylish Header for Light Mode
    st.markdown(
        "<h1 style='text-align: center; color: #2C3E50;'>🤖 AI-Powered RAG Application</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<h3 style='text-align: center; color: #E67E22;'>Ask Questions from Your PDF Files with AI</h3>",
        unsafe_allow_html=True
    )
    
    # User Input Section
    st.markdown("### 📌 Ask Your Question")
    user_question = st.text_input("Type your question here...", 
                                  placeholder="E.g., What is the summary of the document?", 
                                  help="Enter a query related to the uploaded PDF files 📄")

    with st.sidebar:
        st.markdown("<h3 style='color: #2C3E50;'>⚙️ Update or Create Vector Store</h3>", unsafe_allow_html=True)

        if st.button("📂 Store Vector", use_container_width=True):
            with st.spinner("Processing... ⏳"):
                docs = get_documents()
                get_vector_store(docs)
                st.success("✅ Vector Store Updated Successfully!")

    # Creating two columns for better alignment
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        # Stylish Send Button
        send_btn = st.button("🚀 Generate Answer", use_container_width=True)

    if send_btn:
        with st.spinner("Processing your request... ⏳"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm()
            response = get_response_llm(llm, faiss_index, user_question)

        # Display Output in a Styled Box
        st.markdown("### ✨ AI Response:")
        st.markdown(
            f"""
            <div style="background-color: #ECF0F1; padding: 15px; border-radius: 10px; 
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                <p style="font-size: 18px; color: #2C3E50;">{response}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
