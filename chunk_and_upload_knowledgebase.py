from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chromadb")
CHROMA_DB_COLLECTION_NAME = os.getenv("CHROMA_DB_COLLECTION_NAME", "knowledgebase")
KNOWLEDGEBASE_FOLDER = os.getenv("KNOWLEDGEBASE_FOLDER", "./knowledgebase")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11431")

print(f"Use embedding model: {EMBEDDING_MODEL}")
print(f"Chroma DB path: {CHROMA_DB_PATH}")
print(f"Chroma DB collection: {CHROMA_DB_COLLECTION_NAME}")
print(f"Knowledge folder: {KNOWLEDGEBASE_FOLDER}")
print(f"Ollama base URL: {OLLAMA_BASE_URL}")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
if os.path.exists(CHROMA_DB_PATH):
    print(f"Deleting old database: {CHROMA_DB_PATH}")
    shutil.rmtree(CHROMA_DB_PATH)

vector_store = Chroma(
    collection_name=CHROMA_DB_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def extract_title_from_content(content: str, filename: str) -> str:
    lines = content.strip().split('\n')
    for line in lines:
        if line.lower().startswith('project name:'):
            return line.split(':', 1)[1].strip()
        if line.lower().startswith('projektname:'):
            return line.split(':', 1)[1].strip()
    return Path(filename).stem


def detect_language(content: str) -> str:
    german_indicators = ['der', 'die', 'das', 'und', 'ist', 'für', 'mit', 'von']
    content_lower = content.lower()
    german_count = sum(1 for word in german_indicators if f' {word} ' in content_lower)
    return 'de' if german_count >= 3 else 'en'


def process_txt_files(folder_path: str):
    folder = Path(folder_path)
    txt_files = list(folder.glob("*.txt"))
    
    print(f"\nFound {len(txt_files)} text files")
    
    total_chunks = 0
    
    for file_path in txt_files:
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"Skip empty file: {file_path.name}")
                continue

            title = extract_title_from_content(content, file_path.name)
            language = detect_language(content)

            metadata = {
                "source": file_path.name,
                "title": title,
                "language": language
            }

            chunks = text_splitter.create_documents(
                [content],
                metadatas=[metadata]
            )

            uuids = [str(uuid4()) for _ in range(len(chunks))]
            vector_store.add_documents(documents=chunks, ids=uuids)
            total_chunks += len(chunks)
            print(f"Complete {file_path.name}: {len(chunks)} chunks (language: {language})")
            
        except Exception as e:
            print(f"Error in handling {file_path.name}: {e}")
    
    return total_chunks


if __name__ == "__main__":
    total = process_txt_files(KNOWLEDGEBASE_FOLDER)
    print(f"\nTotal docs in Database: {vector_store._collection.count()}")
