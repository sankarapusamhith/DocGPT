import streamlit as st
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings , AlephAlphaAsymmetricSemanticEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
import fitz
import spacy

st.set_page_config(
    page_title="DocGPT",
    page_icon="",
)

st.title("LLM Based Conversational System ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def load_pdf(file):   # To load document

    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    return text

def txt_splitter_rcts(file):  # Document Transformers (Text splitter) method using RecursiveCharacterTextSplitter

  text=load_pdf(file)
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap  = 20,
      length_function = len,
  )
  texts = text_splitter.split_text(text=text)

  return texts

def hembed(texts):

  embeddings = HuggingFaceEmbeddings()
  embed=embeddings.embed_documents(texts)
  return embed

def faiss_hembed(texts): # Vector Stores (FAISS Integration) with HuggingFaceEmbed

  embed = FAISS.from_texts(texts, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

  return embed

def sim_scr_spacy(df,pdf_name):

  nlp =  spacy.load("en_core_web_lg")
  for i in df['id'][df.topic_name==pdf_name]:
    doc1 = nlp(list(df["answer"][df.id==i])[0].lower())
    doc2 = nlp(list(df["pred_ans"][df.id==i])[0].lower())
    df.loc[df.id==i,"score"]=doc1.similarity(doc2)

  return df[df.topic_name==pdf_name]

def huggingface_eval(docs,question): # Final output Model for HuggingFace

  repo_id = "google/flan-t5-xxl"
  llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1})

  chain = load_qa_chain(llm=llm, chain_type="stuff")
  res = chain.run(input_documents=docs, question=question)

  return res


uploaded_file = st.file_uploader("Choose an Input ( .pdf ) file",type="pdf")

if uploaded_file is not None:
    hrf_texts=txt_splitter_rcts(uploaded_file)
    hrf_embed=faiss_hembed(hrf_texts)

    with st.form('my_form'):
        query = st.text_area('prompt:', '',
        placeholder='')
        submitted = st.form_submit_button('Submit')
        if submitted:
            hrf_docs=hrf_embed.similarity_search(query,k=3)
            res=huggingface_eval(hrf_docs,query)
            res


st.write("")
st.write("")
st.write("")
st.write("")

st.write("#### Limitations")
st.markdown("- In some scenarios it may provide trimmed form of actual answer ")
st.markdown("- Math computations ")
st.markdown("- Token limit issues are bound to arise from time to time ")
st.markdown("- Input data format confined to pdf ")

