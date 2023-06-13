import streamlit as st
import os

import numpy as np
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

# ====================

# os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
# ====================

# Loading VectorStore and setup the QA with Source Chain
persist_directory = 'db'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
model = OpenAI(temperature=0)
retriever = vectordb.as_retriever()
retriever.search_kwargs['k'] = 4
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=model, chain_type="stuff", retriever=retriever, return_source_documents=True)

# App Design
st.title('Ask the ECSS')

st.write("""
Ask any questions over the Active ECSS Standards 
(European Cooperation for Space Standardisation)""")

query = st.text_area("Question:", value="")

# query = st.text_input("Type in your question", "")

if st.button('Submit'):

    with st.spinner('Wait for it...'):
        result = chain(
            {"question": query},
            return_only_outputs=True,)

        st.success('Done!')

        st.write(result['sources'])

        st.write(result['answer'])

        with st.expander("See explanation"):
            for doc in result['source_documents']:
                st.write(doc.page_content)
