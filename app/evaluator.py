import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

try:
    from app.engine import get_rag_response_with_sources
except:
    from engine import get_rag_response_with_sources

def run_evaluation():
    """Runs a benchmark on OnRAG current knowledge base."""
    
    # 1. Define a set of 'Ground Truth' questions and expected answers 
    # based on the notes in data/raw/
    test_questions = [
        {
            "question": "What is the main goal of Project OnRAG?",
            "ground_truth": "The goal is to build a privacy-first AI assistant."
        },
        {
            "question": "Where is the engineer currently working?",
            "ground_truth": "The engineer is working in Kanagawa."
        }
    ]

    # 2. Collect OnRAG's responses
    results = []
    for item in test_questions:
        print(f"Testing: {item['question']}")
        # Get the actual RAG output
        output = get_rag_response_with_sources(item['question'])
        
        # To do: Call Fast api
        docs = db.similarity_search(item['question'], k=3)
        context_strings = [doc.page_content for doc in docs]

        results.append({
            "question": item['question'],
            "answer": output["answer"],
            "contexts": context_strings,  # Must be List[str]
            "ground_truth": item['ground_truth']
        })

    # 3. Convert to Dataset
    dataset = Dataset.from_list(results)

    # 4. Run Evaluation
    print("\n--- Running RAGAS Evaluation ---")
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    
    return score

if __name__ == "__main__":
    scores = run_evaluation()
    print("\n" + "="*30)
    print("OnRAG PERFORMANCE METRICS")
    print("="*30)
    print(scores)