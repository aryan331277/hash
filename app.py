import os
import json
import hashlib
import re
from flask import Flask, request, jsonify
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "badal-embeddings")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Globals
index = None
parent_lookup = {}
initialized = False

def create_hash_embedding(text, dimension=384):
    """Create semantic-ish embedding using hash + text features"""
    # Clean text
    text = text.lower().strip()
    words = re.findall(r'\w+', text)
    
    # Text features
    word_count = len(words)
    char_count = len(text)
    unique_words = len(set(words))
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    
    # Multiple hash seeds for variety
    hashes = []
    for seed in range(0, dimension, 8):
        hash_input = f"{text}_{seed}_{word_count}_{unique_words}"
        hash_val = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Convert hex to numbers
        for i in range(0, min(32, dimension - len(hashes)), 4):
            if len(hashes) >= dimension:
                break
            hex_chunk = hash_val[i:i+4]
            num = int(hex_chunk, 16) / 65535.0 * 2 - 1  # Normalize to [-1, 1]
            hashes.append(num)
    
    # Add text feature dimensions
    while len(hashes) < dimension:
        feature_idx = len(hashes)
        if feature_idx % 4 == 0:
            val = (word_count % 100) / 50.0 - 1
        elif feature_idx % 4 == 1:
            val = (char_count % 500) / 250.0 - 1
        elif feature_idx % 4 == 2:
            val = (unique_words % 100) / 50.0 - 1
        else:
            val = (int(avg_word_len * 10) % 20) / 10.0 - 1
        hashes.append(val)
    
    return hashes[:dimension]

def initialize_models():
    """Initialize Pinecone only"""
    global index, initialized
    
    if initialized:
        return
        
    try:
        logger.info("Connecting to Pinecone...")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY required")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if needed
        existing_indexes = pc.list_indexes()
        if PINECONE_INDEX_NAME not in [i.name for i in existing_indexes]:
            logger.info(f"Creating index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}}
            )
        
        index = pc.Index(PINECONE_INDEX_NAME)
        load_data_files()
        initialized = True
        logger.info("Initialized successfully")
        
    except Exception as e:
        logger.error(f"Init error: {str(e)}")
        raise

def load_data_files():
    """Load data files"""
    global parent_lookup
    try:
        if os.path.exists("parent.json"):
            with open("parent.json", "r", encoding="utf-8") as f:
                parents = json.load(f)
            parent_lookup = {p["parent_id"]: p for p in parents}
            logger.info(f"Loaded {len(parents)} parents")
    except Exception as e:
        logger.error(f"Data load error: {str(e)}")

def ensure_initialized():
    if not initialized:
        initialize_models()

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def upload_data_to_pinecone():
    """Upload data to Pinecone"""
    try:
        if not os.path.exists("child.json"):
            return
            
        with open("child.json", "r", encoding="utf-8") as f:
            children = json.load(f)
        
        logger.info(f"Uploading {len(children)} entries...")
        
        for batch in chunk_list(children, 50):
            vectors = []
            for child in batch:
                child_id = child["child_id"]
                parent_id = child["parent_id"]
                parent_obj = parent_lookup.get(parent_id, {})
                
                # Create hash embedding
                embedding = create_hash_embedding(child["text"])
                
                vectors.append({
                    "id": child_id,
                    "values": embedding,
                    "metadata": {
                        "parent_id": parent_id,
                        "parent_source": parent_obj.get("source", ""),
                        "parent_title": parent_obj.get("title", ""),
                        "parent_text": parent_obj.get("text", ""),
                        "child_text": child["text"],
                        "original_data": json.dumps(child.get("original_data", {})),
                        "parent_tables": json.dumps(parent_obj.get("tables", []))
                    }
                })
            
            index.upsert(vectors=vectors)
        
        logger.info("Upload complete")
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")

def search_similar_chunks(query, top_k=5):
    """Search chunks"""
    try:
        query_embedding = create_hash_embedding(query)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return results["matches"]
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

def generate_answer_with_gemini(query, context_chunks):
    """Generate answer with Gemini"""
    if not gemini_model:
        return "Gemini API not configured"
    
    try:
        context = ""
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk["metadata"]
            context += f"Source {i}: {metadata.get('parent_title', 'N/A')}\n"
            context += f"Content: {metadata.get('child_text', '')}\n\n"
        
        prompt = f"Answer this question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "RAG API running",
        "initialized": initialized,
        "webhook_url": f"{request.url_root}webhook"
    })

@app.route("/upload", methods=["POST"])
def upload_data():
    ensure_initialized()
    try:
        upload_data_to_pinecone()
        return jsonify({"message": "Data uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    ensure_initialized()
    try:
        data = request.get_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not query:
            return jsonify({"error": "Query required"}), 400
        
        chunks = search_similar_chunks(query, top_k)
        
        results = []
        for chunk in chunks:
            results.append({
                "score": chunk["score"],
                "content": chunk["metadata"]["child_text"],
                "parent_title": chunk["metadata"].get("parent_title", ""),
                "parent_source": chunk["metadata"].get("parent_source", "")
            })
        
        return jsonify({"results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    ensure_initialized()
    try:
        data = request.get_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not query:
            return jsonify({"error": "Query required"}), 400
        
        chunks = search_similar_chunks(query, top_k)
        
        if not chunks:
            return jsonify({
                "answer": "No relevant information found",
                "sources": []
            })
        
        answer = generate_answer_with_gemini(query, chunks)
        
        sources = []
        for chunk in chunks:
            sources.append({
                "title": chunk["metadata"].get("parent_title", ""),
                "source": chunk["metadata"].get("parent_source", ""),
                "content": chunk["metadata"]["child_text"][:200] + "...",
                "relevance_score": chunk["score"]
            })
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "query": query
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    """Your precious webhook"""
    ensure_initialized()
    try:
        data = request.get_json()
        logger.info(f"Webhook: {data}")
        
        event_type = data.get("type", "unknown")
        
        if event_type == "query":
            query = data.get("query", "")
            if query:
                chunks = search_similar_chunks(query, 3)
                answer = generate_answer_with_gemini(query, chunks)
                
                return jsonify({
                    "status": "success",
                    "answer": answer,
                    "event_type": event_type
                })
        
        return jsonify({
            "status": "received",
            "event_type": event_type,
            "message": "Webhook processed"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
