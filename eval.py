import streamlit as st
from langchain_ollama import ChatOllama
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval
)
from deepeval.evaluate import CacheConfig
from chatbot import build_graph

# ==========================================
# 1. SETUP STREAMLIT E GRAFO
# ==========================================
print("Compilazione del grafo LangGraph...")
rag_app = build_graph()

# ==========================================
# 2. GIUDICE OLLAMA
# ==========================================
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="mistral-nemo"):
        self.model = ChatOllama(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content
        
    def get_model_name(self):
        return "Ollama - mistral-nemo"

judge = OllamaJudge()

metrics = [
    ContextualPrecisionMetric(threshold=0.7, model=judge),
    ContextualRecallMetric(threshold=0.7, model=judge),
    AnswerRelevancyMetric(threshold=0.7, model=judge),
    FaithfulnessMetric(threshold=0.7, model=judge),
    GEval(
        name="Factual Correctness",
        criteria="Determina se la risposta effettiva (actual output) è fattualmente corretta e concorda in modo accurato con l'output atteso (expected output).",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.7,
        model=judge
    )
]

# ==========================================
# 3. DATASET
# ==========================================
from dataset import dataset

# ==========================================
# 4. GENERAZIONE DINAMICA E VALUTAZIONE
# ==========================================
def main():
    test_cases = []
    
    print(f"\nInizio generazione risposte per {len(dataset)} domande...\n")
    
    for item in dataset:
        print(f"➜ Domanda: {item['question']}")
        
        initial_state = {
            "question": item["question"], 
            "chat_history": [], 
            "retry_count": 0,
            "current_model": "llama3.1" 
        }
        
        final_state = rag_app.invoke(initial_state)
        
        actual_output = final_state.get("generation", "")

        raw_context = final_state.get("context", "")
        retrieval_context = [raw_context] if raw_context else ["Nessun documento trovato."]
        
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=actual_output,
            expected_output=item["expected"],
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)
        print("  ✓ Risposta e contesto acquisiti.\n")

    print("=========================================")
    print("Avvio della valutazione DeepEval")
    print("=========================================")
    
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        cache_config=CacheConfig(use_cache=True)
    )
    
    print("\nValutazione completata. Risultati:")
    for metric_name, score in results.items():
        print(f"  {metric_name}: {score:.4f}")

if __name__ == "__main__":
    main()