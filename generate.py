import os
import logging
import click
import torch
import utils
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
import subprocess

# from langchain_groq import ChatGroq

# To run it, type: python run_localGPT_g.py --save_qa
#Streamlit code starts here

import streamlit as st

# Set page title
st.set_page_config(page_title='EvidenceBot')


# Define custom CSS styles

# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #FFFFFF;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

custom_css = """
<style>
    body {
      background-color:  #000000;
    }
    .container {
      max-width: 800px;
      margin: 40px auto;
      padding: 20px;
      background-color: #fff;
      border-radius: 5px;
      box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .form-group label {
      font-weight: bold;
    }
    .chat-container {
      margin-top: 20px;
      border: 1px solid #ccc;
      border-radius: 5px;
      padding: 10px;
      background-color: #f8f9fa;
    }
    .chat-messages {
      max-height: 300px;
      overflow-y: auto;
      margin-bottom: 10px;
    }
    .chat-message {
      background-color: #e9ecef;
      padding: 10px;
      margin-bottom: 10px;
      border-radius: 5px;
    }
    .chat-input {
      display: flex;
      align-items: center;
    }
    .chat-input input {
      flex: 1;
      margin-right: 10px;
    }
    .navbar {
      background-color: #8dd5e3;
      padding: 10px;
      border-radius: 5px;
      margin-bottom: 20px;
      display: flex;
       justify-content: space-between;
       align-items: center;

    }
    .navbar-brand {
      font-weight: bold;
      font-size: 36px;
      color: #343a40;
      text-decoration: none;
    }

    .navbar-brand:hover,
    .navbar-brand:focus {
      text-decoration: none; /* Add this line to remove the underline on hover and focus */
      color: #343a40; /* Add this line to maintain the text color on hover and focus */
    }


    .navbar-nav {
      display: flex;
      list-style-type: none; /* Add this line */
    }
    .navbar-nav .nav-item {
      margin-left: 10px;
    }
    .navbar-nav .nav-link {
      color: #343a40;
      padding: 5px 15px;
      border-radius: 5px;
      transition: background-color 0.3s;
      margin-left: auto;
      text-decoration: none;
      font-size: 20px;
    }
    .navbar-nav .nav-link:hover,
    .navbar-nav .nav-link.active {
      background-color: #343a40;
      color: #fff;
    }


  
</style>
"""

# Display custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Navbar
st.markdown("""
<nav class="navbar">
  <a class="navbar-brand" href="#">
    <img src="https://raw.githubusercontent.com/Nafiz43/portfolio/main/img/EvidenceBotLogo.webp" alt="Logo" width="60" height="60" class="d-inline-block align-top">
    EvidenceBot
  </a>
  <ul class="navbar-nav flex-row">
    <li class="nav-item">
      <a class="nav-link active" href="#">Generate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" href="#">Evaluate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" href="#" data-target="about">About</a>
    </li>
  </ul>
</nav>
""", unsafe_allow_html=True)

# Generate Response section
st.markdown('<div id="generateResponse" class="section active">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    m_name = st.selectbox('Model Name', ['llama2:latest', 'LLAMA2-70B', 'LLAMA3-8B', 'MIXTRAL-7B', 'MIXTRAL-8x7B'])

with col2:
    quantized_version = st.selectbox('Embedding Model', ['hkunlp/instructor-large'])

col3, col4 = st.columns(2)

with col3:
    temp = st.selectbox('Temperature', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

with col4:
    prompt_technique = st.selectbox('Prompting Technique', ['CoT', 'ToT', 'base', 'n-shot', 'React'])

pdf_input = st.file_uploader('Upload PDF Document', type='pdf')

col5, col6 = st.columns([1, 1])

with col5:
    chat_history = st.checkbox('Chat History')

with col6:
    save_output = st.checkbox('Save Output')

# chat_container = st.empty()

# with chat_container.container():
#     st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
#     # Chat messages will be dynamically added here
#     st.markdown('</div>', unsafe_allow_html=True)

chat_input = st.text_area('Enter your instruction', key='instruction_input')


st.markdown('</div>', unsafe_allow_html=True)

LLM_ANSWER = "answer"
LLM_SOURCE_DOC = "source"



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)



def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="mistral"):
    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    # llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    print(m_name)

    llm= Ollama(model=m_name, temperature= temp)
    # mixtral-8x7b-32768
    # model_name = 'gemma-7b-it',
    # llama2-70b-4096

    # llm = ChatGroq(
		# groq_api_key = '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',
		# model_name = 'mixtral-8x7b-32768',
    #     temperature = temp
    # )

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
# @click.command()
# @click.option(
#     "--device_type",
#     default="cuda" if torch.cuda.is_available() else "cpu",
#     type=click.Choice(
#         [
#             "cpu",
#             "cuda",
#             "ipu",
#             "xpu",
#             "mkldnn",
#             "opengl",
#             "opencl",
#             "ideep",
#             "hip",
#             "ve",
#             "fpga",
#             "ort",
#             "xla",
#             "lazy",
#             "vulkan",
#             "mps",
#             "meta",
#             "hpu",
#             "mtia",
#         ],
#     ),
#     help="Device to run on. (Default is cuda)",
# )
# @click.option(
#     "--show_sources",
#     "-s",
#     is_flag=True,
#     help="Show sources along with answers (Default is False)",
# )
# @click.option(
#     "--use_history",
#     "-h",
#     is_flag=True,
#     help="Use history (Default is False)",
# )
# @click.option(
#     "--model_type",
#     default="llama",
#     type=click.Choice(
#         ["llama", "mistral", "non_llama"],
#     ),
#     help="model type, llama, mistral or non_llama",
# )
# @click.option(
#     "--save_qa",
#     is_flag=True,
#     help="whether to save Q&A pairs to a CSV file (Default is False)",
# )

# @click.option(
#     "--model_name",
#     default="llama2:latest",
#     type=click.Choice(
#         ["gemma:7b-instruct", "mistral:7b-instruct", "mixtral:8x7b-instruct-v0.1-q4_K_M", 
#          "llama2:latest", "llama2:70b-chat-q4_K_M", "llama2:13b-chat", "llama3:8b-instruct-q4_K_M"],
#     ),
#     help="model type, llama, mistral or non_llama",
# )
# @click.option(
#     "--prompt_tech",
#     default="base",
#     type=click.Choice(
#         ["base", "n_shot", "CoT", 
#          "React", "ToT"],
#     ),
#     help="model type, llama, mistral or non_llama",
# )


def get_llm_response():
    global m_name
    global save_output
    global LLM_ANSWER
    global LLM_SOURCE_DOC


    global prompt_technique  
    # prompt_technique = prompt_tech

    print(m_name, prompt_technique)

    device_type = "cpu"
    show_sources = True
    use_history = True
    model_type = "llama"

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    cnt =0
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    res = qa(chat_input)
    answer, docs = res["result"], res["source_documents"]

    LLM_ANSWER = answer

    LLM_SOURCE_DOC = docs

    # LLM_ANSWER = "hey"

    # LLM_SOURCE_DOC = "there"

    print("\n\n> Question:")
    print(chat_input)
    print("\n> Answer:")
    print(LLM_ANSWER)

    if show_sources:  # this is a flag that you can set to disable showing answers.
        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

            # Log the Q&A to CSV only if save_qa is True
            
    if save_output:
        actual_annotation = ''
        # utils.log_to_csv(query, answer, actual_annotation, m_name, prompt_technique)
    print("end of main()")
    return 1
    

#This is the main func
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )



    if st.button("Send"):
        if(chat_input == ""):
            st.warning("Enter your instruction")
        else:
            # command = 'conda init && conda activate localGPT && python ingest.py'

            # # Run the command
            # subprocess.run(command, shell=True, check=True)


            # subprocess.run(['python', 'ingest.py'])
            if pdf_input is not None:
              with open(os.path.join('SOURCE_DOCUMENTS', pdf_input.name), 'wb') as f:
                f.write(pdf_input.getbuffer())
              st.write(f'PDF file "{pdf_input.name}" uploaded successfully.')

            get_llm_response()
            # st.write('Model Name:', m_name)
            # st.write('Quantized Version:', quantized_version)
            # st.write('Temperature:', temp)
            # st.write('Prompting Technique:', prompt_technique)
            # st.write('PDF Input:', pdf_input)
            # st.write('Chat History:', chat_history)
            # st.write('Save Output:', save_output)
            # st.write('Chat Output:', chat_input)
            
            st.write('LLM Response:', LLM_ANSWER)
            st.write('SOURCE:', LLM_SOURCE_DOC)
            
        
        
    # instruction = st.session_state.instruction_input
    # if instruction.strip() != '':
    #     chat_message = f'<div class="chat-message">{instruction}</div>'
    #     chat_container.markdown(chat_message, unsafe_allow_html=True)
    #     st.session_state.instruction_input = ''

        # Print the selected options
        
        # x = calculate()
        



