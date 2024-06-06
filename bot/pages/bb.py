import os
from collections import namedtuple
import streamlit as st

from start import load_model
from start import load_data

# Test-Corpora
Corpus = namedtuple('TestCorpus', ['description', 'docs_path', 'storage_path', 'system_prompt'])
c = Corpus("Dokumente aus der Benutzung", "./bot/data_bb", "./bot/storage_bb", "Du bist ein Experte für die Deutsche Nationalbibliothek. Du hilfst Nutzerinnen und Nutzern dabei, die Bibliothek zu benutzen. Du beantwortest Fragen zum Ausleihbetrieb und den verfügbaren Services. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Bibliothek. Wenn Du eine Information über die Bibliothek nicht hast, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")

# Angaben zum Modell
# model_options = ["sonar-small-chat", "mixtral-8x7b-instruct"]
model_options = [
    "llama3",
    "mixtral",
    "aya",      
    "phi3"
]
# temperature = 0.2

# Einstellungen sichtbar machen
with st.sidebar:
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
            # for node in response.source_nodes[:5]:
                # st.write(f"S. {node.metadata['page_label']}, Datei {node.metadata['file_path']}")
            
            # An Verlauf anhängen
            message = {"role": "bot", "content": response.response}
            st.session_state.messages.append(message)
