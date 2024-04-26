import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.perplexity import Perplexity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# Begrüßung
st.title("DNB-ChatBot")

# Laden des Perplexity API-Key (in der shell anlegen mit: export PERPLEXITY_API_KEY=...)
pplx_api_key = os.environ["PERPLEXITY_API_KEY"]

# Erstellen einer Instanz von Perplexity LLM
# llm = Perplexity(api_key=pplx_api_key, model="sonar-small-chat", temperature=0.5, system_prompt="Du bist ein Experte für die Deutsche Nationalbibliothek. Du hilfst Nutzerinnen und Nutzern dabei, die Bibliothek zu benutzen. Du beantwortest Fragen zum Ausleihbetrieb und den verfügbaren Services. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Bibliothek, die nicht auf Fakten basieren. Wenn Du eine Information über die Bibliothek nicht hast, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")
# llm = Perplexity(api_key=pplx_api_key, model="sonar-small-chat", temperature=0.5, system_prompt="Du bist ein Experte für die Briefe von Christian Felix Weiße, die von Mark Lehmstedt unter Mitarbeit von Katrin Löffler herausgegeben und ediert wurden. Du hilfst Nutzerinnen und Nutzern dabei, Informationen über diese Briefe zu erhalten. Du beantwortest Fragen zum Inhalt der Briefe. Jeder Brief ist nummeriert. Zu Beginn jedes Eintrags steht, an wen der Brief Adressiert ist, und, wenn möglich, wann und wo er von Weiße geschrieben wurde. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Briefe, die nicht auf Fakten basieren. Wenn Du eine Information in den Briefen nicht findest, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")
llm = Perplexity(api_key=pplx_api_key, model="sonar-small-chat", temperature=0.5, system_prompt="Du bist ein Experte für die Erschließung von Schlagworten. Du hilfst den Erschließenden dabei, die richtigen Schlagworte zu finden. Du beantwortest dafür Fragen auf der Grundlage der Regeln für die Schlagwortkategorisierung. Deine Antworten sollen auf den Regeln basieren. Halluziniere keine Informationen über die Schlagwortkategorisierung, die nicht auf den Regeln basieren. Wenn Du eine Information über die Schlagwortkategorisierung nicht hast, sage den Erschließenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Pfad zu den lokalen Dokumenten
# docs_path = "./bot/data"
# persist_dir = "./bot/storage"
# docs_path = "./bot/data_bb"
# persist_dir = "./bot/storage_bb"
# docs_path = "./bot/data_lehmstedt"
# persist_dir = "./bot/storage_lehmstedt"
docs_path = "./bot/data_rswk"
persist_dir = "./bot/storage_rswk"


@st.cache_resource(show_spinner=True)
def load_data():
    # Erstellen eines Indexes der lokalen Dateien
    if not os.path.exists(persist_dir):
        st.write('Indiziere die Dokumente. Das dauert ein paar Augenblicke...')
        documents = SimpleDirectoryReader(docs_path).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
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
        {"role": "bot", "content": "Was möchten Sie erfahren?"}
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

            # Quellenangaben ausgeben
            for node in response.source_nodes:
                st.write(f"S. {node.metadata['page_label']}, Datei {node.metadata['file_path']}")
            
            # An Verlauf anhängen
            message = {"role": "bot", "content": response.response}
            st.session_state.messages.append(message)
