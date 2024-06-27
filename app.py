import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from transformers import AutoTokenizer, TextStreamer, pipeline, AutoModelForQuestionAnswering
from auto_gptq import AutoGPTQForCausalLM
import torch
from flask import Flask, jsonify,request
app = Flask(__name__)

print("hi")
print("hi")
print("hi")
print("hi")
embedding_model = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
def load_text_files(directory):
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
  db.save_local("faiss")

db = FAISS.load_local('faiss', embedding_model,allow_dangerous_deserialization=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=None,
)

base_retriever=db.as_retriever(search_kwargs={"k": 2})
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,

)
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=base_retriever,
    return_only_outputs=True,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
)

@app.route("/generate",methods=['POST'])
def generateresult(query):
  #result = qa_chain(query)
  #return jsonify(result['result'])
    return  "cmgh"