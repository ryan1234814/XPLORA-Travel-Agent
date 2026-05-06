import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.rag import rag_db

sample_blogs = [
    {
        "content": "A definitive guide to Tokyo: Visit the Meiji Shrine early in the morning to avoid crowds. For the best sushi, skip the main Tsukiji outer market and head to the smaller local shops in Ginza. The subway pass is a lifesaver.",
        "metadata": {"source": "Tokyo Travel Blog", "destination": "Tokyo", "author": "TravelNinja"}
    },
    {
        "content": "Backpacking in Rome: You can drink from the public fountains called 'nasoni' safely. To save money, don't sit down at cafes near major monuments; standing at the bar costs a fraction of the price. Best pizza is in Trastevere.",
        "metadata": {"source": "BudgetBackpacker", "destination": "Rome", "author": "BudgetBob"}
    },
    {
        "content": "Luxury Dubai: The Burj Khalifa At The Top experience is best booked at sunset. For a unique desert safari, opt for the vintage Land Rover tours instead of the standard dune bashing. Stay near Downtown Dubai for the best access to high-end dining.",
        "metadata": {"source": "LuxeLife", "destination": "Dubai", "author": "LuxeLinda"}
    },
    {
        "content": "Hidden gems in Paris: Everyone goes to the Louvre, but the Musee d'Orsay has an incredible impressionist collection in a beautiful old train station. For a great view without the Eiffel Tower lines, go to the top of the Montparnasse Tower.",
        "metadata": {"source": "Parisian Secrets", "destination": "Paris", "author": "FrenchFan"}
    }
]

def main():
    print("Ingesting sample travel blogs to Pinecone (or local mock DB)...")
    
    documents = [blog["content"] for blog in sample_blogs]
    metadatas = [blog["metadata"] for blog in sample_blogs]
    
    rag_db.add_documents(documents, metadatas)
    
    print("\nIngestion complete. Test queries:")
    
    queries = [
        "What are some tips for visiting Rome on a budget?",
        "Where can I get a good view in Paris besides the Eiffel Tower?",
        "Tips for Tokyo transportation and food"
    ]
    
    for q in queries:
        print(f"\n--- Query: {q} ---")
        result = rag_db.query(q, k=1)
        print(result)

if __name__ == "__main__":
    main()
