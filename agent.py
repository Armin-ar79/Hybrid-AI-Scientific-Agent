import os
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- 1. تنظیمات اولیه ---
DB_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

TEMPLATE = """You are an expert scientific researcher. Answer the user's question based ONLY on the following context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}
"""

def join_and_truncate_docs(docs, tokenizer, reserved_for_answer_tokens=128):
    pieces = [getattr(d, "page_content", str(d)).strip() for d in docs if getattr(d, "page_content", None) or str(d).strip()]
    if not pieces:
        return ""
    joined = "\n\n---\n\n".join(pieces)

    model_max = getattr(tokenizer, "model_max_length", None)
    if not model_max or model_max <= 0 or model_max > 100000:
        model_max = 512

    max_context_tokens = max(32, model_max - reserved_for_answer_tokens)

    enc = tokenizer(joined, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    if input_ids.shape[0] > max_context_tokens:
        truncated_ids = input_ids[:max_context_tokens]
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return truncated_text
    else:
        return joined

def create_agent():
    print("در حال بارگذاری مدل Embedding (MiniLM)...")
    device_available = torch.cuda.is_available()
    device_name = 'cuda' if device_available else 'cpu'
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device_name}
    )

    print(f"در حال بارگذاری دیتابیس وکتوری از: {DB_PATH}...")
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    print(f"در حال بارگذاری LLM ({LLM_MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    try:
        import accelerate  # فقط برای بررسی وجود
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
        )
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
        if device_available:
            model.to("cuda")
            pipeline_device = 0
            print("Device set to use cuda")
        else:
            pipeline_device = -1
            print("Device set to use cpu")
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            device=pipeline_device,
        )

    # برگرداندن vectordb برای fallback در صورت نیاز
    return {
        "vectordb": vectordb,
        "retriever": retriever,
        "tokenizer": tokenizer,
        "pipeline": hf_pipeline,
        "template": TEMPLATE
    }

def retrieve_documents_safe(agent, query):
    retriever = agent.get("retriever")
    vectordb = agent.get("vectordb")

    # 1) تلاش برای متد مرسوم روی retriever
    if retriever is not None:
        if hasattr(retriever, "get_relevant_documents"):
            try:
                return retriever.get_relevant_documents(query)
            except Exception as e:
                print("get_relevant_documents on retriever failed:", e)
        if hasattr(retriever, "similarity_search"):
            try:
                return retriever.similarity_search(query, k=3)
            except Exception as e:
                print("similarity_search on retriever failed:", e)

    # 2) تلاش برای متد روی خود vectordb (مثلاً Chroma)
    if vectordb is not None:
        if hasattr(vectordb, "similarity_search"):
            try:
                return vectordb.similarity_search(query, k=3)
            except Exception as e:
                print("similarity_search on vectordb failed:", e)
        if hasattr(vectordb, "get_relevant_documents"):
            try:
                return vectordb.get_relevant_documents(query)
            except Exception as e:
                print("get_relevant_documents on vectordb failed:", e)

    # 3) اگر retriever callable بود، امتحان کن
    if callable(retriever):
        try:
            return retriever(query)
        except Exception as e:
            print("Calling retriever(query) failed:", e)

    # 4) هیچ روش معتبری پیدا نشد — چاپ وضعیت برای دیباگ و بالا بردن خطا
    print("==== Debug info: retriever and vectordb types and available attributes ====")
    print("Retriever type:", type(retriever))
    if retriever is not None:
        print("Retriever dir():", [n for n in dir(retriever) if not n.startswith("_")])
    print("Vectordb type:", type(vectordb))
    if vectordb is not None:
        print("Vectordb dir():", [n for n in dir(vectordb) if not n.startswith("_")])
    raise RuntimeError("No known retrieval method found on retriever or vectordb. See debug output above.")

def answer_query(agent, query, k=3):
    tokenizer = agent["tokenizer"]
    hf_pipeline = agent["pipeline"]
    template = agent["template"]

    docs = retrieve_documents_safe(agent, query)

    if docs is None:
        docs = []
    if not isinstance(docs, list):
        try:
            docs = list(docs)
        except Exception:
            docs = [docs]

    if len(docs) == 0:
        context = ""
    else:
        context = join_and_truncate_docs(docs, tokenizer, reserved_for_answer_tokens=150)

    prompt = template.format(context=context, question=query)
    outputs = hf_pipeline(prompt, max_new_tokens=200)
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        text = first.get("generated_text") or first.get("text") or str(first)
    else:
        text = str(outputs)

    return text

def start_chat(agent):
    print("\n--- ✅ عامل هوشمند علمی آماده است ---")
    print("درباره PDFهایی که آپلود کردید، سؤال بپرسید (برای خروج 'exit' را تایپ کنید)")

    while True:
        query = input("\n❓ سوال شما: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            print("لطفا سوال خود را تایپ کنید.")
            continue

        print("در حال جستجو در اسناد و فکر کردن...")
        try:
            answer = answer_query(agent, query)
            print("\n--- 🧠 پاسخ عامل ---")
            print(answer)
        except Exception as e:
            print(f"بروز خطا در هنگام اجرای عامل: {e}")

if __name__ == "__main__":
    agent = create_agent()
    start_chat(agent)