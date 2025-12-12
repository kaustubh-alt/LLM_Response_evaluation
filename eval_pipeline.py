"""
RAG Evaluation Pipeline - Refined & Fixed
Evaluates: Relevance (Cross-Encoder), Completeness (Semantic), Hallucination, Accuracy, Latency, Cost

Requirements:
pip install sentence-transformers spacy nltk scikit-learn tiktoken numpy pandas tqdm scipy

python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
"""

import json
import time
import re
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# NLP Imports
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# SpaCy handling with robust fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not available - entity detection and semantic chunking will be limited.")

# Ensure NLTK data exists
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    turn_number: int
    user_query: str
    ai_response: str
    
    # Relevance & Completeness
    relevance_score: float
    completeness_score: float
    answer_type_match: bool
    
    # Hallucination & Factual Accuracy
    hallucination_score: float
    factual_accuracy_score: float
    entity_accuracy: float
    numerical_accuracy: float
    unsupported_claims: List[str]
    
    # Context Info
    top_contexts_used: List[str]
    context_similarity_scores: List[float]
    num_contexts_retrieved: int
    
    # Latency & Cost
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    
    # Flags
    requires_human_review: bool
    failure_reasons: List[str]
    
    def __post_init__(self):
        if self.unsupported_claims is None:
            self.unsupported_claims = []
        if self.failure_reasons is None:
            self.failure_reasons = []


class ModelCache:
    """Singleton for model loading to prevent reloading on every turn"""
    _embedding_model = None
    _cross_encoder = None
    _nlp_model = None
    _tokenizer = None
    
    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            print("📦 Loading Bi-Encoder (all-MiniLM-L6-v2)...")
            cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedding_model
    
    @classmethod
    def get_cross_encoder(cls):
        """Load Cross-Encoder for high-accuracy relevance scoring"""
        if cls._cross_encoder is None:
            print("📦 Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
            cls._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cls._cross_encoder
    
    @classmethod
    def get_nlp_model(cls):
        if cls._nlp_model is None and SPACY_AVAILABLE:
            try:
                if not spacy.util.is_package("en_core_web_sm"):
                    print("⚠️  Downloading en_core_web_sm...")
                    spacy.cli.download("en_core_web_sm")
                cls._nlp_model = spacy.load('en_core_web_sm')
                print("✓ NLP model loaded")
            except Exception as e:
                print(f"⚠️  Could not load spaCy model: {e}")
                cls._nlp_model = None
        return cls._nlp_model
    
    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = tiktoken.get_encoding("cl100k_base")
        return cls._tokenizer


class RelevanceEvaluator:
    """Evaluate response relevance using Cross-Encoders and Semantic Matching"""
    
    def __init__(self, embedding_model, nlp_model):
        self.embedding_model = embedding_model
        self.cross_encoder = ModelCache.get_cross_encoder()
        self.nlp = nlp_model
    
    def evaluate(self, query: str, response: str) -> Tuple[float, bool, float]:
        """Returns (relevance_score, answer_type_match, completeness_score)"""
        if not query or not response:
            return 0.0, False, 0.0

        # 1. Relevance: Cross-Encoder (High Accuracy)
        # Predicts a logit score (unbounded). We use sigmoid to squash to 0-1.
        raw_score = self.cross_encoder.predict([(query, response)])[0]
        relevance = 1 / (1 + np.exp(-raw_score))
        
        # 2. Answer Type Match
        type_match = self._check_answer_type(query, response)
        
        # 3. Completeness: Semantic Coverage
        completeness = self._check_completeness_semantic(query, response)
        
        return float(relevance), type_match, completeness
    
    def _check_answer_type(self, query: str, response: str) -> bool:
        """Heuristic check for expected answer format"""
        q_lower = query.lower()
        r_lower = response.lower()
        
        patterns = {
            'how_to': (r'\bhow (do|to|can|should)\b', 
                      lambda r: bool(re.search(r'\b(step|first|then|process|guide)\b', r))),
            'list': (r'\b(list|name|enumerate|what are the)\b',
                    lambda r: bool(re.search(r'(\d\.|•|-|,)', r))),
            'comparison': (r'\b(compare|difference|vs|versus)\b',
                          lambda r: bool(re.search(r'\b(while|whereas|but|however|both|unlike)\b', r))),
            'yes_no': (r'^(is|are|does|do|can|will|should)\b',
                      lambda r: bool(re.search(r'\b(yes|no|correct|incorrect|true|false)\b', r[:50])))
        }
        
        for pattern, validator in patterns.values():
            if re.search(pattern, q_lower):
                return validator(r_lower)
        return True
    
    def _check_completeness_semantic(self, query: str, response: str) -> float:
        """Check if key concepts in query are semantically present in response"""
        if self.nlp is None:
            # Fallback for systems without spaCy: simple word overlap
            q_words = set(re.findall(r'\w+', query.lower())) - {'what', 'how', 'is', 'the', 'a'}
            if not q_words: return 1.0
            r_words = set(re.findall(r'\w+', response.lower()))
            return len(q_words & r_words) / len(q_words)

        # 1. Extract Concepts (Noun Chunks + Verbs)
        q_doc = self.nlp(query)
        q_concepts = [t.text for t in q_doc if t.pos_ in ['NOUN', 'PROPN', 'VERB'] and not t.is_stop]
        
        if not q_concepts:
            return 1.0
            
        # 2. Embed concepts and response sentences
        concept_embs = self.embedding_model.encode(q_concepts, convert_to_tensor=True)
        
        # Split response into sentences (better granularity)
        r_sentences = [s.text for s in self.nlp(response).sents]
        if not r_sentences: r_sentences = [response]
        r_embs = self.embedding_model.encode(r_sentences, convert_to_tensor=True)
        
        # 3. Calculate Semantic Similarity Matrix
        # Shape: (num_concepts, num_response_sentences)
        cos_scores = util.cos_sim(concept_embs, r_embs)
        
        # 4. Check coverage
        # For each concept, find the sentence in response that best matches it
        max_scores_per_concept = cos_scores.max(dim=1).values
        
        # If best match > 0.45, we consider the concept "covered" (handles synonyms)
        covered_count = (max_scores_per_concept > 0.45).sum().item()
        
        return covered_count / len(q_concepts)


class HallucinationDetector:
    """Multi-layer hallucination detection"""
    
    def __init__(self, embedding_model, nlp_model):
        self.embedding_model = embedding_model
        self.nlp = nlp_model
    
    def evaluate(self, response: str, context: str) -> Tuple[float, float, float, float, List[str]]:
        if not context or not response:
            return 1.0, 0.0, 0.0, 0.0, [response]
        
        # Layer 1: Semantic grounding (Vector similarity of sentences)
        semantic_score = self._semantic_grounding(response, context)
        
        # Layer 2: Entity consistency
        entity_acc = self._entity_consistency(response, context)
        
        # Layer 3: Numerical consistency
        numerical_acc = self._numerical_consistency(response, context)
        
        # Layer 4: Unsupported claims extraction
        unsupported = self._find_unsupported_claims(response, context)
        
        # Weighted average for factual accuracy
        factual_accuracy = (semantic_score * 0.4) + (entity_acc * 0.3) + (numerical_acc * 0.3)
        hallucination_score = 1.0 - factual_accuracy
        
        return float(hallucination_score), float(factual_accuracy), float(entity_acc), float(numerical_acc), unsupported
    
    def _semantic_grounding(self, response: str, context: str) -> float:
        sentences = sent_tokenize(response)
        if not sentences: return 1.0
        
        # Embed context as a single block for grounding check
        context_emb = self.embedding_model.encode(context, convert_to_tensor=True)
        sentence_embs = self.embedding_model.encode(sentences, convert_to_tensor=True)
        
        # Calculate similarity of each sentence to the context
        scores = util.cos_sim(sentence_embs, context_emb)
        return float(scores.mean().item())
    
    def _entity_consistency(self, response: str, context: str) -> float:
        if self.nlp is None:
            return self._simple_entity_check(response, context)
            
        r_doc = self.nlp(response)
        c_doc = self.nlp(context)
        
        r_ents = {e.text.lower() for e in r_doc.ents if e.label_ not in ['DATE', 'TIME', 'CARDINAL']}
        c_ents = {e.text.lower() for e in c_doc.ents}
        
        if not r_ents: return 1.0
        
        supported = len(r_ents & c_ents)
        return supported / len(r_ents)
    
    def _simple_entity_check(self, response: str, context: str) -> float:
        # Fallback: Capitalized words
        r_caps = set(re.findall(r'\b[A-Z][a-z]+\b', response))
        c_caps = set(re.findall(r'\b[A-Z][a-z]+\b', context))
        if not r_caps: return 1.0
        return len(r_caps & c_caps) / len(r_caps)
    
    def _numerical_consistency(self, response: str, context: str) -> float:
        # Extract numbers (integers, floats, percentages)
        pattern = r'\b\d+(?:\.\d+)?%?\b'
        r_nums = [float(re.sub(r'[^\d.]', '', n)) for n in re.findall(pattern, response)]
        c_nums = [float(re.sub(r'[^\d.]', '', n)) for n in re.findall(pattern, context)]
        
        if not r_nums: return 1.0
        if not c_nums: return 0.0
        
        matches = 0
        for rn in r_nums:
            # Check for exact or approximate match (within 5%)
            if any(abs(rn - cn) <= (0.05 * max(abs(cn), 1)) for cn in c_nums):
                matches += 1
                
        return matches / len(r_nums)
    
    def _find_unsupported_claims(self, response: str, context: str) -> List[str]:
        sentences = sent_tokenize(response)
        unsupported = []
        if not sentences: return []
        
        s_embs = self.embedding_model.encode(sentences, convert_to_tensor=True)
        c_emb = self.embedding_model.encode(context, convert_to_tensor=True)
        
        scores = util.cos_sim(s_embs, c_emb)
        
        for i, score in enumerate(scores):
            # If similarity is very low, flag as unsupported
            if score.item() < 0.35:
                unsupported.append(sentences[i])
                
        return unsupported


class CostEstimator:
    """Estimate token usage and cost"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pricing = {
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'gpt-4o': {'input': 0.0025, 'output': 0.01},
        }
    
    def estimate(self, query: str, context: str, response: str, model: str = 'gpt-3.5-turbo') -> Dict:
        try:
            input_text = f"{query}\n{context}"
            input_tokens = len(self.tokenizer.encode(input_text))
            output_tokens = len(self.tokenizer.encode(response))
            
            price = self.pricing.get(model, self.pricing['gpt-3.5-turbo'])
            cost = (input_tokens * price['input'] / 1000 + output_tokens * price['output'] / 1000)
            
            return {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'estimated_cost': cost
            }
        except Exception:
            return {'input_tokens': 0, 'output_tokens': 0, 'estimated_cost': 0.0}


class RAGEvaluationPipeline:
    """Main evaluation pipeline"""
    
    def __init__(self, top_k_contexts: int = 3):
        self.top_k = top_k_contexts
        
        # Load models
        self.embedding_model = ModelCache.get_embedding_model()
        self.nlp_model = ModelCache.get_nlp_model()
        self.tokenizer = ModelCache.get_tokenizer()
        
        # Initialize evaluators
        self.relevance_eval = RelevanceEvaluator(self.embedding_model, self.nlp_model)
        self.hallucination_detector = HallucinationDetector(self.embedding_model, self.nlp_model)
        self.cost_estimator = CostEstimator(self.tokenizer)
        
        # Thresholds
        self.thresholds = {
            'min_relevance': 0.6,
            'max_hallucination': 0.4,
            'min_completeness': 0.6,
            'min_factual': 0.7
        }
    
    def load_chat_data(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        turns = data.get('conversation_turns', [])
        conversations = []
        for i, turn in enumerate(turns):
            if turn.get('role') == 'User':
                # Look ahead for AI response
                for j in range(i + 1, len(turns)):
                    if turns[j].get('role') == 'AI/Chatbot':
                        conversations.append({
                            'turn_number': turn.get('turn', i),
                            'user_query': turn.get('message', ''),
                            'ai_response': turns[j].get('message', '')
                        })
                        break
        return conversations
    
    def load_context_data(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('data', {}).get('vector_data', [])
    
    def retrieve_relevant_contexts(self, query: str, all_contexts: List[Dict]) -> Tuple[List[str], List[float]]:
        if not all_contexts: return [], []
        
        query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
        ctx_texts = [c.get('text', '') for c in all_contexts]
        ctx_embs = self.embedding_model.encode(ctx_texts, convert_to_tensor=True)
        
        # Calculate similarity
        scores = util.cos_sim(query_emb, ctx_embs)[0]
        
        # Top K
        top_k_indices = scores.argsort(descending=True)[:self.top_k]
        top_contexts = [ctx_texts[i] for i in top_k_indices]
        top_scores = [scores[i].item() for i in top_k_indices]
        
        return top_contexts, top_scores
    
    def evaluate_turn(self, query: str, response: str, retrieved_ctx: List[str], ctx_scores: List[float], turn_id: int):
        context_text = "\n\n".join(retrieved_ctx)
        
        # 1. Relevance & Completeness
        relevance, type_match, completeness = self.relevance_eval.evaluate(query, response)
        
        # 2. Hallucination
        hal_score, fact_acc, ent_acc, num_acc, unsupported = \
            self.hallucination_detector.evaluate(response, context_text)
        
        # 3. Cost
        cost = self.cost_estimator.estimate(query, context_text, response)
        
        # 4. Flags
        reasons = []
        if relevance < self.thresholds['min_relevance']: reasons.append(f"Low Relevance ({relevance:.2f})")
        if hal_score > self.thresholds['max_hallucination']: reasons.append(f"High Hallucination ({hal_score:.2f})")
        if completeness < self.thresholds['min_completeness']: reasons.append(f"Incomplete ({completeness:.2f})")
        if not type_match: reasons.append("Wrong Answer Type")
        
        return EvaluationResult(
            turn_number=turn_id,
            user_query=query,
            ai_response=response,
            relevance_score=relevance,
            completeness_score=completeness,
            answer_type_match=type_match,
            hallucination_score=hal_score,
            factual_accuracy_score=fact_acc,
            entity_accuracy=ent_acc,
            numerical_accuracy=num_acc,
            unsupported_claims=unsupported,
            top_contexts_used=retrieved_ctx,
            context_similarity_scores=ctx_scores,
            num_contexts_retrieved=len(retrieved_ctx),
            estimated_input_tokens=cost['input_tokens'],
            estimated_output_tokens=cost['output_tokens'],
            estimated_cost_usd=cost['estimated_cost'],
            requires_human_review=len(reasons) > 0,
            failure_reasons=reasons
        )

    def run(self, chat_file: str, context_file: str, output_file: str):
        print(f"🚀 Starting Evaluation Pipeline...")
        print(f"📂 Chat: {chat_file}")
        print(f"📂 Context: {context_file}")
        
        chats = self.load_chat_data(chat_file)
        contexts = self.load_context_data(context_file)
        results = []
        
        for chat in tqdm(chats, desc="Processing Turns"):
            # Retrieve
            top_ctx, scores = self.retrieve_relevant_contexts(chat['user_query'], contexts)
            
            # Evaluate
            res = self.evaluate_turn(
                chat['user_query'], 
                chat['ai_response'], 
                top_ctx, 
                scores, 
                chat['turn_number']
            )
            results.append(res)
            
        # Export
        self._save_results(results, output_file)

    def _save_results(self, results: List[EvaluationResult], filename: str):
        # JSON Save
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_turns': len(results),
                    'flagged_turns': sum(1 for r in results if r.requires_human_review),
                    'avg_relevance': float(np.mean([r.relevance_score for r in results])),
                    'avg_factual_accuracy': float(np.mean([r.factual_accuracy_score for r in results]))
                },
                'details': [asdict(r) for r in results]
            }, f, indent=2, ensure_ascii=False)
            
        # CSV Export
        df = pd.DataFrame([
            {
                'Turn': r.turn_number,
                'Relevance': r.relevance_score,
                'Completeness': r.completeness_score,
                'Hallucination': r.hallucination_score,
                'Factual_Acc': r.factual_accuracy_score,
                'Flagged': r.requires_human_review,
            } for r in results
        ])
        csv_path = filename.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to {filename} and {csv_path}")


if __name__ == "__main__":
    # CONFIGURATION
    CHAT_FILE = "sample-chat-conversation-01.json"
    CONTEXT_FILE = "sample_context_vectors-01.json"
    OUTPUT_FILE = "eval_results_refined.json"

        
    pipeline = RAGEvaluationPipeline(top_k_contexts=3)

    pipeline.run(CHAT_FILE, CONTEXT_FILE, OUTPUT_FILE)
