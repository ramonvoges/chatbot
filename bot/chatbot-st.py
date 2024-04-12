import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.perplexity import Perplexity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# Begrüßung
st.title("DNB-ChatBot")
# st.subheader("Hallo! Ich beantworte Fragen zur Nutzung der Deutschen Nationalbibliothek. Wie kann ich weiterhelfen?")

# Laden des Perplexity API-Key (in der shell anlegen mit: export PERPLEXITY_API_KEY=...)
pplx_api_key = os.environ["PERPLEXITY_API_KEY"]

# Erstellen einer Instanz von Perplexity LLM
llm = Perplexity(api_key=pplx_api_key, model="mistral-7b-instruct", temperature=0.5, system_prompt="Du bist ein Experte für die Deutsche Nationalbibliothek. Du hilfst Nutzerinnen und Nutzern dabei, die Bibliothek zu benutzen. Du beantwortest Fragen zum Ausleihbetrieb und den verfügbaren Services. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Bibliothek, die nicht auf Fakten basieren. Wenn Du eine Information über die Bibliothek nicht hast, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Pfad zu den lokalen Dokumenten
docs_path = "./bot/data"
persist_dir = "./bot/storage"

@st.cache_resource(show_spinner=True)
def load_data():
    # Erstellen eines Indexes der lokalen Dateien
    if not os.path.exists(persist_dir):
        st.write('Indiziere die Dokumente. Das dauert ein paar Augenblicke...')
        documents = SimpleDirectoryReader(docs_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        st.write('Lade die indizierten Dokumente...')
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index

# Konvertieren des Indexes in eine Abfrage-Engine
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, verbose=True)

# Chat-Verlauf initialisieren
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "bot", "content": "Was möchten Sie über die Nutzung der DNB erfahren?"}
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
            response = chat_engine.chat(prompt)
            st.write(response.response)
            # An Verlauf anhängen
            message = {"role": "bot", "content": response.response}
            st.session_state.messages.append(message)
