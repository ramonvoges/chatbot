import os
from collections import namedtuple

import streamlit as st
import ollama


st.title("Zusammenfassung")

# Test-Corpora
Corpus = namedtuple('TestCorpus', ['description', 'docs_path', 'storage_path', 'system_prompt'])
c = Corpus("Vorlass Altenhein", "./bot/data_altenhein", "./bot/storage_altenhein", "Du bist ein Experte für den Vorlass von Herrn Altenhein und den Börsenverein des Deutschen Buchhandels. Du hilfst Nutzerinnen und Nutzern dabei, Informationen über die Dokumente aus dem Vorlass von Herrn Altenhein zu erhalten. Du beantwortest Fragen zum Inhalt der Dokumente. Jedes Dokument ist in einer eigenen Datei gespeichert. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Dokumente, die nicht auf Fakten basieren. Wenn Du eine Information in den Dokumenten nicht findest, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch.")

if "summary" not in st.session_state.keys():
    st.session_state.summary = ""

def summarize_text(file_name):
    with open(f"./bot/data_altenhein/{file_name}", 'r', encoding='utf-8') as file:
        content = file.read()
    prompt = "Deine Aufgabe ist es, den gegebenen Text in ungefäng 100 Wörtern zusammenzufassen. Gebe nur die Zusammenfassung wieder ohne weitere Angaben."
    response = ollama.chat(
        model="llama3",
        stream=True,
        messages=[
            {
                'role': 'user',
                'content': f"{prompt}. Das ist der Text: {content}"
            },
        ],
        )
    full_response = ""
    for item in response:
        full_response += item['message']['content']
        placeholder.write(full_response)
    st.session_state.summary = full_response
    placeholder.write(f"Zusammenfassung: {st.session_state.summary}")
file_name = st.selectbox("Datei auswählen", os.listdir(c.docs_path))
st.button("Zusammenfassen", on_click=summarize_text, args=(file_name,))
placeholder = st.empty()
# if st.session_state.summary is not None: 
#     placeholder.write(st.session_state.summary)
placeholder.write(st.session_state.summary)
# st.write(f"Zusammenfassung: {st.session_state.summary}")
    # st.write(st.session_state.summary)
# with open(f"./bot/data_altenhein/{file_name}", 'r', encoding='utf-8') as file:
#     content = file.read()
# prompt = "Deine Aufgabe ist es, den gegebenen Text in ungefähr 500 Wörtern zusammenzufassen. Gebe nur die Zusammenfassung wieder ohne weitere Angaben."
# response = ollama.chat(
#     model="llama3",
#     stream=True,
#     messages=[
#         {
#             'role': 'user',
#             'content': f"{prompt}. Das ist der Text: {content}"
#         },
#     ],
#     )
# # st.write_stream(response)
# placeholder = st.empty()
# full_response = ""
# for item in response:
#     full_response += item['message']['content']
#     placeholder.write(full_response)
    # st.write(item['message']['content'])
# summary = response['message']['content']
# st.write(summary)