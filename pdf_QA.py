from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path("env_variables.env"))

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# provide the path of  pdf file/files.
pdfreader = PdfReader('stories.pdf')

# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 4000,
    chunk_overlap  = 1000,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "The ant and the dove story i need"
docs = document_search.similarity_search(query)

answer = chain.run(input_documents=docs, question=query)

print(answer)