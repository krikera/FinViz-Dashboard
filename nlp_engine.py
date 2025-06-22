import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core NLP Libraries with fallback handling
import warnings
warnings.filterwarnings('ignore')

# Initialize availability flags
NLP_AVAILABLE = False
SPACY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
TEXTBLOB_AVAILABLE = False
WORDCLOUD_AVAILABLE = False
GENSIM_AVAILABLE = False
SKLEARN_AVAILABLE = False

# Import libraries individually with error handling
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("SpaCy not available")

# Transformers will be tested at runtime to avoid TensorFlow import issues
TRANSFORMERS_AVAILABLE = None  # Will be determined at runtime

# Sentence Transformers will be tested at runtime to avoid TensorFlow import issues
SENTENCE_TRANSFORMERS_AVAILABLE = None  # Will be determined at runtime

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("TextBlob not available")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("WordCloud not available")

try:
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    print("Gensim not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Scikit-learn not available")

# Set overall availability
NLP_AVAILABLE = any([SPACY_AVAILABLE, TRANSFORMERS_AVAILABLE, TEXTBLOB_AVAILABLE, SKLEARN_AVAILABLE])

class FinancialNLPEngine:
    """Advanced NLP Engine for Financial Transaction Analysis"""
    
    def __init__(self):
        self.nlp_available = NLP_AVAILABLE
        if not self.nlp_available:
            return
            
        # Initialize models
        self._init_models()
        self._load_financial_keywords()
        
    def _init_models(self):
        """Initialize NLP models based on available libraries"""
        self.nlp = None
        self.sentiment_analyzer = None
        self.sentence_model = None
        self.tfidf = None
        
        # Initialize SpaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
            
        # Test and initialize Transformers at runtime
        global TRANSFORMERS_AVAILABLE
        if TRANSFORMERS_AVAILABLE is None:
            try:
                # Test import at runtime to avoid TensorFlow issues
                from transformers import pipeline
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                TRANSFORMERS_AVAILABLE = True
            except Exception as e:
                print(f"Transformers not available: {e}")
                self.sentiment_analyzer = None
                TRANSFORMERS_AVAILABLE = False
        elif TRANSFORMERS_AVAILABLE:
            try:
                from transformers import pipeline
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            except Exception as e:
                print(f"Error loading sentiment analyzer: {e}")
                self.sentiment_analyzer = None
                TRANSFORMERS_AVAILABLE = False
            
        # Test and initialize Sentence Transformers at runtime
        global SENTENCE_TRANSFORMERS_AVAILABLE
        if SENTENCE_TRANSFORMERS_AVAILABLE is None:
            try:
                # Test import at runtime to avoid TensorFlow issues
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                SENTENCE_TRANSFORMERS_AVAILABLE = True
            except Exception as e:
                print(f"Sentence Transformers not available: {e}")
                self.sentence_model = None
                SENTENCE_TRANSFORMERS_AVAILABLE = False
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Error loading sentence transformer: {e}")
                self.sentence_model = None
                SENTENCE_TRANSFORMERS_AVAILABLE = False
        
        # Initialize TF-IDF if scikit-learn is available
        if SKLEARN_AVAILABLE:
            try:
                self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            except Exception as e:
                print(f"Error initializing TF-IDF: {e}")
                self.tfidf = None
        
    def _load_financial_keywords(self):
        """Load financial domain-specific keywords and categories"""
        self.financial_categories = {
            'food_dining': ['restaurant', 'cafe', 'food', 'dining', 'pizza', 'burger', 'coffee', 'lunch', 'dinner', 'starbucks', 'mcdonalds', 'kfc'],
            'shopping': ['store', 'shop', 'mall', 'amazon', 'purchase', 'buy', 'retail', 'walmart', 'target', 'costco'],
            'transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'metro', 'bus', 'parking', 'toll', 'vehicle'],
            'utilities': ['electric', 'water', 'internet', 'phone', 'cable', 'utility', 'power', 'gas_bill'],
            'healthcare': ['hospital', 'doctor', 'medical', 'pharmacy', 'health', 'dental', 'clinic', 'medicine'],
            'entertainment': ['movie', 'cinema', 'theater', 'netflix', 'spotify', 'game', 'entertainment', 'music'],
            'finance': ['bank', 'atm', 'interest', 'fee', 'loan', 'credit', 'investment', 'insurance'],
            'education': ['school', 'university', 'education', 'tuition', 'book', 'course', 'training'],
            'travel': ['hotel', 'flight', 'travel', 'booking', 'airbnb', 'vacation', 'trip', 'airline']
        }
        
    def advanced_transaction_categorization(self, descriptions: List[str]) -> List[Dict]:
        """Advanced categorization using multiple NLP techniques"""
        if not self.nlp_available:
            return [{'category': 'Unknown', 'confidence': 0.0} for _ in descriptions]
            
        results = []
        
        for desc in descriptions:
            # Clean and preprocess text
            clean_desc = self._preprocess_text(desc)
            
            # Method 1: Rule-based with financial keywords
            rule_category, rule_confidence = self._rule_based_categorization(clean_desc)
            
            # Method 2: Embedding-based similarity
            embedding_category, embedding_confidence = self._embedding_based_categorization(clean_desc)
            
            # Method 3: Named Entity Recognition
            entities = self._extract_entities(clean_desc)
            
            # Combine results with weighted scoring
            final_category, final_confidence = self._combine_categorization_results(
                rule_category, rule_confidence,
                embedding_category, embedding_confidence,
                entities
            )
            
            results.append({
                'category': final_category,
                'confidence': final_confidence,
                'entities': entities,
                'methods': {
                    'rule_based': {'category': rule_category, 'confidence': rule_confidence},
                    'embedding': {'category': embedding_category, 'confidence': embedding_confidence}
                }
            })
            
        return results
    
    def sentiment_analysis(self, descriptions: List[str]) -> List[Dict]:
        """Analyze sentiment of transaction descriptions with fallback options"""
        if not self.nlp_available:
            return [{'sentiment': 'NEUTRAL', 'confidence': 0.0} for _ in descriptions]
            
        results = []
        for desc in descriptions:
            try:
                sentiment_result = None
                
                # Try transformer model first
                if self.sentiment_analyzer:
                    try:
                        sentiment_result = self.sentiment_analyzer(desc)[0]
                    except Exception:
                        sentiment_result = None
                
                # Use TextBlob as primary or fallback
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(desc)
                    textblob_sentiment = 'POSITIVE' if blob.sentiment.polarity > 0.1 else 'NEGATIVE' if blob.sentiment.polarity < -0.1 else 'NEUTRAL'
                    
                    if sentiment_result:
                        results.append({
                            'sentiment': sentiment_result['label'],
                            'confidence': sentiment_result['score'],
                            'textblob_sentiment': textblob_sentiment,
                            'polarity': blob.sentiment.polarity,
                            'subjectivity': blob.sentiment.subjectivity
                        })
                    else:
                        # Use TextBlob as primary
                        confidence = abs(blob.sentiment.polarity) if blob.sentiment.polarity != 0 else 0.1
                        results.append({
                            'sentiment': textblob_sentiment,
                            'confidence': confidence,
                            'textblob_sentiment': textblob_sentiment,
                            'polarity': blob.sentiment.polarity,
                            'subjectivity': blob.sentiment.subjectivity
                        })
                else:
                    # No sentiment analysis available
                    results.append({
                        'sentiment': 'NEUTRAL',
                        'confidence': 0.0,
                        'error': 'No sentiment analysis libraries available'
                    })
                    
            except Exception as e:
                results.append({
                    'sentiment': 'NEUTRAL',
                    'confidence': 0.0,
                    'error': str(e)
                })
                
        return results
    
    def extract_financial_insights(self, descriptions: List[str]) -> Dict:
        """Extract comprehensive financial insights from transaction descriptions"""
        if not self.nlp_available:
            return {}
            
        insights = {
            'merchants': [],
            'locations': [],
            'amounts_mentioned': [],
            'time_indicators': [],
            'financial_terms': [],
            'spending_themes': []
        }
        
        for desc in descriptions:
            if self.nlp:
                doc = self.nlp(desc)
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        insights['merchants'].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        insights['locations'].append(ent.text)
                    elif ent.label_ == "MONEY":
                        insights['amounts_mentioned'].append(ent.text)
                    elif ent.label_ in ["DATE", "TIME"]:
                        insights['time_indicators'].append(ent.text)
                
                # Extract financial terms
                financial_terms = self._extract_financial_terms(desc)
                insights['financial_terms'].extend(financial_terms)
        
        # Clean and deduplicate
        for key in insights:
            insights[key] = list(set(insights[key]))
            
        # Topic modeling for spending themes
        insights['spending_themes'] = self._discover_spending_themes(descriptions)
        
        return insights
    
    def natural_language_search(self, descriptions: List[str], query: str, data: pd.DataFrame) -> pd.DataFrame:
        """Natural language search through transactions"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.sentence_model:
            return data
            
        try:
            # Convert query to embedding
            query_embedding = self.sentence_model.encode([query])
            
            # Convert descriptions to embeddings
            desc_embeddings = self.sentence_model.encode(descriptions)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, desc_embeddings)[0]
            
            # Add similarity scores to dataframe
            data = data.copy()
            data['nlp_similarity'] = similarities
            
            # Return top matches (similarity > 0.3)
            return data[data['nlp_similarity'] > 0.3].sort_values('nlp_similarity', ascending=False)
        except Exception:
            return data
    
    def generate_spending_summary(self, data: pd.DataFrame) -> str:
        """Generate natural language summary of spending patterns"""
        if not self.nlp_available or data.empty:
            return "NLP analysis not available."
            
        try:
            # Basic statistics
            total_transactions = len(data)
            total_spent = data['Withdrawls'].sum() if 'Withdrawls' in data.columns else 0
            avg_transaction = total_spent / total_transactions if total_transactions > 0 else 0
            
            # Category analysis
            top_categories = data['Category'].value_counts().head(3) if 'Category' in data.columns else []
            
            # Sentiment analysis of recent transactions
            recent_descriptions = data['Description'].tail(10).tolist() if 'Description' in data.columns else []
            sentiments = self.sentiment_analysis(recent_descriptions)
            positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
            
            # Generate summary
            summary = f"""📊 Financial Summary:
            
• Total Transactions: {total_transactions:,}
• Total Amount: ${total_spent:,.2f}
• Average Transaction: ${avg_transaction:.2f}

🏆 Top Spending Categories:"""
            
            for i, (category, count) in enumerate(top_categories.items(), 1):
                summary += f"\n• {i}. {category}: {count} transactions"
            
            summary += f"\n\n💭 Recent Transaction Sentiment: {positive_count}/{len(sentiments)} transactions show positive sentiment"
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def detect_anomalous_descriptions(self, descriptions: List[str]) -> List[bool]:
        """Detect anomalous transaction descriptions using NLP"""
        if not SKLEARN_AVAILABLE or not self.tfidf or len(descriptions) < 2:
            return [False] * len(descriptions)
            
        try:
            tfidf_matrix = self.tfidf.fit_transform(descriptions)
            
            # Cluster descriptions
            n_clusters = min(10, len(descriptions) // 5) if len(descriptions) > 10 else 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Find outliers (small clusters)
            cluster_counts = pd.Series(clusters).value_counts()
            outlier_clusters = cluster_counts[cluster_counts < len(descriptions) * 0.05].index
            
            anomalies = [cluster in outlier_clusters for cluster in clusters]
            return anomalies
            
        except Exception:
            return [False] * len(descriptions)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _rule_based_categorization(self, text: str) -> Tuple[str, float]:
        """Rule-based categorization using financial keywords"""
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self.financial_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score / len(keywords)
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category]
            return best_category, confidence
        
        return 'other', 0.1
    
    def _embedding_based_categorization(self, text: str) -> Tuple[str, float]:
        """Embedding-based categorization using sentence transformers"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.sentence_model:
            return 'other', 0.1
            
        try:
            # Create category descriptions
            category_descriptions = {
                'food_dining': 'restaurant food dining eating meal',
                'shopping': 'shopping store purchase retail buying',
                'transportation': 'transport gas fuel uber taxi driving',
                'utilities': 'utility bill electric water internet phone',
                'healthcare': 'medical health doctor hospital pharmacy',
                'entertainment': 'entertainment movie music game fun',
                'finance': 'bank atm financial interest fee money',
                'education': 'education school university learning course',
                'travel': 'travel hotel flight vacation trip booking'
            }
            
            # Get embeddings
            text_embedding = self.sentence_model.encode([text])
            category_embeddings = self.sentence_model.encode(list(category_descriptions.values()))
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, category_embeddings)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_category = list(category_descriptions.keys())[best_idx]
            confidence = similarities[best_idx]
            
            return best_category, confidence
            
        except Exception:
            return 'other', 0.1
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        entities = {'organizations': [], 'locations': [], 'money': [], 'dates': []}
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        entities['organizations'].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities['locations'].append(ent.text)
                    elif ent.label_ == "MONEY":
                        entities['money'].append(ent.text)
                    elif ent.label_ in ["DATE", "TIME"]:
                        entities['dates'].append(ent.text)
            except Exception:
                pass
        
        return entities
    
    def _combine_categorization_results(self, rule_cat, rule_conf, emb_cat, emb_conf, entities) -> Tuple[str, float]:
        """Combine different categorization methods"""
        # Weighted combination
        rule_weight = 0.4
        embedding_weight = 0.6
        
        if rule_conf > 0.5 and emb_conf > 0.5:
            # Both methods agree or have high confidence
            if rule_cat == emb_cat:
                return rule_cat, (rule_conf * rule_weight + emb_conf * embedding_weight)
            else:
                # Choose the one with higher confidence
                if rule_conf > emb_conf:
                    return rule_cat, rule_conf
                else:
                    return emb_cat, emb_conf
        elif rule_conf > emb_conf:
            return rule_cat, rule_conf
        else:
            return emb_cat, emb_conf
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial terms from text"""
        financial_terms = ['payment', 'transfer', 'deposit', 'withdrawal', 'fee', 'charge', 'refund', 'purchase', 'sale']
        found_terms = []
        text_lower = text.lower()
        
        for term in financial_terms:
            if term in text_lower:
                found_terms.append(term)
                
        return found_terms
    
    def _discover_spending_themes(self, descriptions: List[str], num_topics: int = 5) -> List[str]:
        """Discover spending themes using topic modeling"""
        if not GENSIM_AVAILABLE:
            return ['Topic modeling not available - Gensim required']
            
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(desc).split() for desc in descriptions]
            processed_texts = [text for text in processed_texts if len(text) > 1]
            
            if len(processed_texts) < 2:
                return ['Insufficient data for theme discovery']
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            # Train LDA model
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
            
            # Extract topics
            topics = []
            for idx, topic in lda_model.print_topics(-1):
                # Extract main words from topic
                words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
                theme = ' + '.join(words[:3])  # Top 3 words
                topics.append(f"Theme {idx+1}: {theme}")
            
            return topics
            
        except Exception as e:
            return [f'Theme discovery error: {str(e)}']
    
    def create_wordcloud(self, descriptions: List[str]):
        """Create word cloud from transaction descriptions"""
        if not WORDCLOUD_AVAILABLE:
            return None
            
        try:
            # Combine all descriptions
            text = ' '.join([self._preprocess_text(desc) for desc in descriptions])
            
            if not text.strip():
                return None
            
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100,
                                colormap='viridis').generate(text)
            
            return wordcloud
            
        except Exception:
            return None 