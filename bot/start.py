import os
from collections import namedtuple
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
# from llama_index.llms.perplexity import Perplexity
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# Begrüßung
st.title("DNB-ChatBot")

@st.cache_resource(show_spinner=True)
def load_model(model, system_prompt):
    # Laden des Modells
    # pplx_api_key = os.environ["PERPLEXITY_API_KEY"]
    # llm = Perplexity(api_key=pplx_api_key, model=model, temperature=temperature, system_prompt=system_prompt)
    llm = Ollama(model=model, request_timeout=120.0, system_prompt=system_prompt)

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    return llm

@st.cache_resource(show_spinner=True)
def load_data(docs_path, persist_dir):
    # Erstellen eines Indexes der lokalen Dateien
    if not os.path.exists(persist_dir):
        st.write(f"Indiziere die Dokumente im Ordner {docs_path}. Das dauert ein paar Augenblicke...")
        documents = SimpleDirectoryReader(docs_path).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        st.write(f"Lade die indizierten Dokumente im Ordner {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index


st.info("Wählen Sie ein Nutzungszenario in der Sidebar!")
st.write("Die Dokumente liegen auf einem Server der DNB. Vector Embeddings sind erzeugt.")

