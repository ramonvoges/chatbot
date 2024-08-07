import os
from collections import namedtuple
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st


# Begrüßung
st.title("DNB-ChatBot")

def delete_chat():
    for key in st.session_state.keys():
        del st.session_state[key]


@st.cache_resource(show_spinner=True)
def load_model(model, system_prompt):
    # Laden des Modells
    llm = Ollama(model=model, request_timeout=300.0, system_prompt=system_prompt)

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


# Test-Corpora
Corpus = namedtuple('TestCorpus', ['description', 'docs_path', 'storage_path', 'system_prompt'])
c = Corpus("Sci-Fi-Literatur", "./Prototyp/data", "./Prototyp/storage", "Du bist ein Experte für den Science Fiction- und Fantasy-Literatur. Du hilfst Nutzerinnen und Nutzern dabei, Informationen über die Dokumente zu erhalten. Du beantwortest Fragen zum Inhalt der Dokumente. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Dokumente. Wenn Du eine Information in den Dokumenten nicht findest, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")

# Angaben zum Modell
model_options = [
    "llama3",
    "mistral",
    "aya",      
    "phi3"
]

# Einstellungen sichtbar machen
with st.sidebar:
    st.button("Chat-Verlauf zurücksetzen", on_click=delete_chat)
    st.write(f"Corpus: {c.description}")
    model = st.selectbox("Modell", options=model_options)
    # temperature = st.slider("Temperature", min_value=0.1, max_value=0.9, step=0.1, value=0.5)
    system_prompt = st.text_area("Charakterisierung", value=c.system_prompt)
    st.write(f"Dateien: {os.listdir(c.docs_path)}")

# Laden des Modells
llm = load_model(model=model, system_prompt=system_prompt)

# Konvertieren des Indexes in eine Abfrage-Engine
index = load_data(docs_path=c.docs_path, persist_dir=c.storage_path)
chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, verbose=True)

# Chat-Verlauf initialisieren
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "bot", "content": "Was möchten Sie über den Vorlass erfahren?"}
    ]

# Interaktive Frage-Antwort-Schleife basierend auf den indexierten Dateien
if prompt := st.chat_input("Ihre Frage"):
     st.session_state.messages.append({"role": "user", "content": prompt})

# Verlauf der Unterhaltung anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Neue Antwort generieren, wenn die message nicht vom Bot ist
if st.session_state.messages[-1]["role"] != "bot":
    with st.chat_message("bot"):
        with st.spinner("Ich denke nach..."):
            response = chat_engine.stream_chat(prompt)
            placeholder = st.empty()
            full_response = ""
            for item in response.response_gen:
                full_response += item
                placeholder.write(full_response)
            placeholder.write(full_response)
            # st.write(response.response)
            # Quellenangaben ausgeben
            # for node in response.source_nodes[:5]:
            #     st.write(f"S. {node.metadata['page_label']}, Datei {node.metadata['file_path']}")
            # An Verlauf anhängen
            message = {"role": "bot", "content": response.response}
            st.session_state.messages.append(message)
