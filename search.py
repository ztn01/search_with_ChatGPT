import wikipedia
from langchain.agents import load_tools, AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from langchain.utilities import GoogleSerperAPIWrapper
import wikipedia
import pandas as pd
from langchain.chat_models import ChatOpenAI

import streamlit as st
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Search with LLM")
st.title("Search with LLM")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

serper_api_key = st.sidebar.text_input("serper API Key", type="password")

if input := st.text_input("输入搜索词"):


    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not serper_api_key:
        st.info("Please add your serper API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)





    keywords=llm("Please extract keywords for the following question, using Spaces instead of commas between keywords, question:{text}",text=input)

    keywords=keywords.split(',')
    vectorstore={}
    text={}

    search_google = GoogleSerperAPIWrapper()
    results_google = search_google.results(keywords)
    urls = []

    for result in results_google['organic']:
        urls.append(result['link'])


    wiki_url = f"https://en.wikipedia.org/w/index.php?search={query}"
    response = requests.get(wiki_url)
    soup = BeautifulSoup(response.text, "html.parser")
    results_wiki = soup.find_all("div", class_="mw-search-result-heading")
    for result in results_wiki:
        url = "https://en.wikipedia.org" + result.find("a")["href"]
        urls.append(url)
        
    


    url_data = pd.DataFrame(columns=['url', 'summary', 'score'])


    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    summary_template = """
    Analyze the following text, give a summary of the text according to the question
    text：{context}
    question: {question}
    summary:"""

    SUMMARY_CHAIN_PROMPT = PromptTemplate.from_template(summary_template)

    score_template = """Analyze the following text and give a score based on whether it matches the question, out of 100 points, you only need to give the number of the final score, you do not need to give a reason分析以下文本，并根据是否匹配问题给出一个得分，满分为100分，你只需要给出最终得分的数字，不需要给出理由
    text：{context}
    question: {question}
    score:"""

    SCORE_CHAIN_PROMPT = PromptTemplate.from_template(score_template)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        
    vectorstore={}
        
    for url in urls:
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.content, 'html.parser')
        for ad in soup.find_all(class_='ad'):
            ad.decompose()
        text = soup.get_text()
        
        all_splits = text_splitter.split_text(text)

        vectorstore[url] = Chroma.from_texts(texts=all_splits, embedding=OpenAIEmbeddings())
        
        summary_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore[url].as_retriever(),
        chain_type_kwargs={"prompt": SUMMARY_CHAIN_PROMPT})
        
        summary = summary_chain({"query": input})["result"]
        
        score_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore[url].as_retriever(),
        chain_type_kwargs={"prompt": SCORE_CHAIN_PROMPT})
        
        score = int(score_chain({"query": input})["result"])
        
        
        data = [
        {'url': url, 'summary':summary, 'score': score}
        ]

        url_data = pd.concat([url_data, pd.DataFrame(data)])
    sorted_data=url_data.sort_values('score', ascending=False)
    for index, row in sorted_data.iterrows():
        url=row['url']
        summary=row['summary']
        score=row['score']
        
        st.write("URL:", url)
        st.write("Summary:", summary)


