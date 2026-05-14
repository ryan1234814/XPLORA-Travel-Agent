import os
from typing import List, Dict, Any, Optional
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
from config.api_config import api_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a free hugging face embeddings model to avoid OpenAI API costs
# all-MiniLM-L6-v2 is small and fast
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class TravelRAG:
    """RAG System using Pinecone to store and retrieve travel blogs/guides."""
    
    def __init__(self):
        self.api_key = api_config.PINECONE_API_KEY
        self.index_name = api_config.PINECONE_INDEX_NAME or "travel-guides"
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL) if HUGGINGFACE_AVAILABLE else None
        self.vectorstore = None
        
        # In-memory fallback if Pinecone API key is not configured
        self._mock_data = [
            {"page_content": "Paris in spring is lovely. The Eiffel Tower is a must-see, and the Louvre has the best art.", "metadata": {"source": "blog", "destination": "Paris"}},
            {"page_content": "When visiting Tokyo, you must try the street food in Shinjuku and visit the ancient temples in Asakusa. Use the subway, it's very efficient.", "metadata": {"source": "guide", "destination": "Tokyo"}},
            {"page_content": "A hidden gem in Rome is the Trastevere neighborhood. Great local food and less crowded than the Colosseum area.", "metadata": {"source": "blog", "destination": "Rome"}}
        ]
        
        self.is_mock = not bool(self.api_key) or not HUGGINGFACE_AVAILABLE or not PINECONE_AVAILABLE
        
        if not self.is_mock:
            try:
                self.pc = Pinecone(api_key=self.api_key)
                self._initialize_index()
                self.vectorstore = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    pinecone_api_key=self.api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}. Falling back to mock data.")
                self.is_mock = True
        else:
            logger.warning("Pinecone API key not found. Using mock RAG system.")

    def _initialize_index(self):
        """Create the Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384, # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index '{self.index_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error checking/creating index: {e}")
            raise e

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add new travel blogs or guides to the vector store."""
        if self.is_mock:
            for i, doc in enumerate(documents):
                meta = metadatas[i] if metadatas else {}
                self._mock_data.append({"page_content": doc, "metadata": meta})
            logger.info(f"Added {len(documents)} documents to mock RAG.")
            return

        if self.vectorstore:
            try:
                self.vectorstore.add_texts(texts=documents, metadatas=metadatas)
                logger.info(f"Successfully added {len(documents)} documents to Pinecone.")
            except Exception as e:
                logger.error(f"Failed to add documents to Pinecone: {e}")

    def query(self, search_term: str, k: int = 3) -> str:
        """Search the vector database for relevant travel guides."""
        if self.is_mock:
            # Simple keyword matching for mock
            results = [doc for doc in self._mock_data if any(word.lower() in doc["page_content"].lower() or word.lower() in doc["metadata"].get("destination", "").lower() for word in search_term.split())]
            if not results:
                # Fallback to some generic results if nothing matches
                results = self._mock_data[:k]
            
            formatted_results = []
            for doc in results[:k]:
                source = doc["metadata"].get("source", "Unknown")
                formatted_results.append(f"[Source: {source}]\n{doc['page_content']}")
            
            return "\n\n".join(formatted_results) if formatted_results else "No relevant travel blogs found."

        try:
            docs = self.vectorstore.similarity_search(search_term, k=k)
            formatted_results = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "Travel Guide")
                formatted_results.append(f"[{title} - {source}]\n{doc.page_content}")
            
            return "\n\n".join(formatted_results) if formatted_results else "No relevant travel blogs found."
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return "Failed to retrieve travel blogs from Knowledge Base."

# Global instance for use in tools
rag_db = TravelRAG()
