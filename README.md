# Chatbot für die DNB - Ein Protoyp

## Die Idee

In einem Verzeichnis `data` liegen die Benutzungsordnung, Leitfäden und Hinweise für die Benutzenden. Über einen Chatbot lassen sich einfache Fragen zur Nutzung der Bibliothek stellen und die dazugehörigen Antworten erhalten.

## Vorbereitungen

```shell
poetry new bot
poetry add llama-index llama-index-embeddings-huggingface llama-index-llms-ollama streamlit
```

## Aufruf des ChatBots

```shell
cd bot
poetry run streamlit run bot/start.py
```

Beim ersten Aufruf werden die Vektoren lokal gesichert, Modelle von z.B. `pytorch` installiert und außerdem Embeddings von [HuggingFace](https://huggingface.co) heruntergeladen. Das dauert eine Weile...
