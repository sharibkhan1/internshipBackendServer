
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from transformers import AutoTokenizer, TextStreamer, pipeline, AutoModelForQuestionAnswering
from auto_gptq import AutoGPTQForCausalLM
import torch
from flask import Flask, jsonify,request
import pinecone
embedding_model = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
def load_text_files(directory):
  pinecone.init(api_key='aa199580-6d1e-4a83-9668-44a3013f39f9')
  index = pinecone.Index('chatbot-university')
    

  loader = PyPDFDirectoryLoader(directory)
  docs_before_split= loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
  docs = text_splitter.split_documents(docs_before_split)

  sub_chunk_sizes = [128, 256, 512]
  sub_chunk_splitter = [
    RecursiveCharacterTextSplitter(chunk_size=c, chunk_overlap=0) for c in sub_chunk_sizes
  ]
  all_chunks = []

  for doc in docs:
    for splitter in sub_chunk_splitter:
          sub_chunks = splitter.split_documents([doc])
          for chunk in sub_chunks:
                all_chunks.append(chunk.page_content)
  
  
  db = FAISS.from_texts(all_chunks, embedding_model)
  index.upsert(vectors = db)
#   db.save_local("faiss")