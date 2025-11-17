import os
import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# --- خطای شما در این خط بود (اصلاح شد) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. تنظیمات اولیه ---
DOCS_PATH = "docs" # پوشه‌ای که PDF ها در آن هستند
DB_PATH = "vector_db" # پوشه‌ای که دیتابیس وکتوری در آن ساخته می‌شود

def create_vector_db():
    print(f"در حال بارگذاری اسناد از پوشه: {DOCS_PATH}...")
    
    # --- 2. بارگذاری PDF ها ---
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="*.pdf", # فقط فایل‌های PDF
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    
    if not documents:
        print("خطا: هیچ فایل PDF در پوشه 'docs' پیدا نشد.")
        return

    print(f"تعداد {len(documents)} فایل PDF با موفقیت بارگذاری شد.")

    # --- 3. خرد کردن متن (Chunking) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0
    )
    splits = text_splitter.split_documents(documents)
    print(f"اسناد به {len(splits)} قطعه (Chunk) تقسیم شدند.")

    # --- 4. تعریف مدل Embedding ---
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    
    print("مدل Embedding (MiniLM) با موفقیت بارگذاری شد.")

    # --- 5. ساخت و ذخیره دیتابیس وکتوری ---
    print(f"در حال ساخت دیتابیس وکتوری در پوشه: {DB_PATH}...")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("\n--- ✅ موفقیت! ---")
    print(f"دیتابیس وکتوری با موفقیت ساخته شد و در '{DB_PATH}' ذخیره گردید.")
    print("مغز RAG شما آماده است. حالا می‌توانید اسکریپت 'agent.py' را اجرا کنید.")

if __name__ == "__main__":
    create_vector_db()