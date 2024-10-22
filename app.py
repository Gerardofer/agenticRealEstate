import os
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
import chainlit as cl  # importing chainlit for our app
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
# from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
# from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools

load_dotenv()

# Environment variables
open_ai_key = os.getenv("OPENAI_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Document loader
document = PyMuPDFLoader("assets/real_estate_book_practice.pdf").load()

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
  # LLM model
  llm = ChatOpenAI(api_key=open_ai_key, model="gpt-4o-mini")
  # LLM prompt
  RAG_PROMPT = """
  CONTEXT: 
  {context}

  QUERY: 
  {question}

  You are a Real Estate Agent with 20 years of experience that enjoys mentoring your clients. Use the available context to answer the question. If you can't answer the question say you don't know.
  """

  # Rag prompt
  rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

  def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(
        text,
    )
    return len(tokens)

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 300,
      chunk_overlap = 0,
      length_function = tiktoken_len,
  )

  split_chunks = text_splitter.split_documents(document)

  qdrant_vectorstore = Qdrant.from_documents(
  split_chunks,
  embedding_model,
  location=":memory:",
  collection_name="extending_context_window_llama_3",
)
  # Qdrant retriever
  qdrant_retriever = qdrant_vectorstore.as_retriever()

  rag_chain = (
  {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
  | rag_prompt | llm | StrOutputParser()
)

  cl.user_session.set("runnable", rag_chain)

# RAG chain

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
