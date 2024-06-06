import os
from collections import namedtuple
import streamlit as st

from start import load_model
from start import load_data


# Begrüßung
st.title("DNB-ChatBot")

# Test-Corpora
Corpus = namedtuple('TestCorpus', ['description', 'docs_path', 'storage_path', 'system_prompt'])
c = Corpus("Regelwerk für Schlagwortkatalogisierung", "./bot/data_rswk", "./bot/storage_rswk", "Du bist ein Experte für die Erschließung von Schlagworten. Du hilfst den Erschließenden dabei, die richtigen Schlagworte zu finden. Du beantwortest dafür Fragen auf der Grundlage der Regeln für die Schlagwortkategorisierung. Deine Antworten sollen auf den Regeln basieren. Halluziniere keine Informationen über die Schlagwortkategorisierung, die nicht auf den Regeln basieren. Wenn Du eine Information über die Schlagwortkategorisierung nicht hast, sage den Erschließenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")

# Angaben zum Modell
model_options = [
    "llama3",
    "mistral",
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
# TODO: Session state ggf. löschen, wenn Seiten gewechselt werden.
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
