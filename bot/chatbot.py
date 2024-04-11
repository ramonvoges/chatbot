import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.perplexity import Perplexity
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Begrüßung
print("Hallo! Ich beantworte Fragen zur Nutzung der Deutschen Nationalbibliothek.")
print("Wie kann ich weiterhelfen? Zum Beenden 'exit' eingeben.")

# Laden des Perplexity API-Key (in der shell anlegen mit: export PERPLEXITY_API_KEY=...)
pplx_api_key = os.environ["PERPLEXITY_API_KEY"]

# Erstellen einer Instanz von Perplexity LLM
print('Lade die Modelle zur Indexierug und stelle eine Verbinung zu Perplexity.ai her.')
llm = Perplexity(api_key=pplx_api_key, model="mistral-7b-instruct", temperature=0.5, system_prompt="Du bist ein Experte für die Deutsche Nationalbibliothek. Du hilfst Nutzerinnen und Nutzern dabei, die Bibliothek zu benutzen. Du beantwortest Fragen zum Ausleihbetrieb und den verfügbaren Services. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Bibliothek, die nicht auf Fakten basieren. Wenn Du eine Information über die Bibliothek nicht hast, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Pfad zu den lokalen Dokumenten
docs_path = "./bot/data"
persist_dir = "./bot/storage"

# Erstellen eines Indexes der lokalen Dateien
if not os.path.exists(persist_dir):
    print('Indexiere die Dokumente...')
    documents = SimpleDirectoryReader(docs_path).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    print('Lade die indexierten Dokumente...')
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

# Konvertieren des Indexes in eine Abfrage-Engine
query_engine = index.as_query_engine(llm=llm)

# Frage stellen und Antwort erhalten
# response = query_engine.query("Darf ich eine Tasche in den Lesesaal mitnehmen? Bitte antworte auf Deutsch.")
# print(response)

# Interaktive Frage-Antwort-Schleife basierend auf den indexierten Dateien
while True:
    text_input = input("Frage: ")
    if text_input == "exit":
        break
    response = query_engine.query(text_input)
    print(f"Bot: {response}")
