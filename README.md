# Chatbot für die DNB - Ein Protoyp

## Die Idee

In einem Verzeichnis `data` liegen die Benutzungsordnung, Leitfäden und Hinweise für die Benutzenden. Über einen Chatbot lassen sich einfache Fragen zur Nutzung der Bibliothek stellen und die dazugehörigen Antworten erhalten.

## Vorbereitungen

```shell
poetry new bot
poetry add llama-index llama-index-embeddings-huggingface llama-index-llms-perplexity llama-index-embeddings-huggingface streamlit
```

Zusätzlich ist ein Pro-Account bei [perplexity.ai](https://www.perplexity.ai/) notwendig. Über die Einstellung -> API kann ein API Key erzeugt werden. Dieser muss als Umgebungsvariable im Terminal vor Aufruf des Skripts hinterlegt werden:

```shell
export PERPLEXITY_API_KEY=...
```

## Aufruf des ChatBots

```shell
cd bot
poetry run streamlit run bot/chatbot-st.py
```

Beim ersten Aufruf werden die Vektoren lokal gesichert, Modelle von z.B. `pytorch` installiert und außerdem Embeddings von [HuggingFace](https://huggingface.co) heruntergeladen. Das dauert eine Weile...
