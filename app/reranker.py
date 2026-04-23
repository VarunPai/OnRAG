from flashrank import Ranker, RerankRequest

def rerank_documents(query, documents):
    """Sorts retrieved documents by actual relevance to the query."""
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="opt")
    
    # Format for FlashRank
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(documents)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Return top 3 as standard LangChain docs
    return results[:3]