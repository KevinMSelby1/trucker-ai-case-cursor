"""
Feature Analysis Script for TruckerPath App Reviews
with progress tracking for long-running operations
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import string
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text cleaning and preprocessing operations."""
    
    def __init__(self):
        """Initialize the text preprocessor and download required NLTK data."""
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        
    @staticmethod
    def _ensure_nltk_resources():
        """Download required NLTK resources with progress tracking."""
        required_resources = ['stopwords', 'punkt', 'vader_lexicon']
        try:
            for resource in tqdm(required_resources, 
                               desc="Downloading NLTK resources",
                               unit="resource"):
                nltk.download(resource, quiet=True)
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download NLTK resources: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return str(text)
            
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        text_tokens = word_tokenize(text)
        return ' '.join([word for word in text_tokens if word not in self.stop_words])

    def batch_clean_texts(self, texts: pd.Series) -> pd.Series:
        """
        Clean multiple texts with progress tracking.
        
        Args:
            texts (pd.Series): Series of texts to clean
            
        Returns:
            pd.Series: Cleaned texts
        """
        cleaned_texts = []
        for text in tqdm(texts, desc="Cleaning texts", unit="review"):
            cleaned_texts.append(self.clean_text(text))
        return pd.Series(cleaned_texts, index=texts.index)

class FeatureAnalyzer:
    """Handles feature extraction and analysis of review data."""
    
    def __init__(self, input_path: str, output_dir: str):
        """Initialize the feature analyzer."""
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.df: Optional[pd.DataFrame] = None
        self.preprocessor = TextPreprocessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.lda_model: Optional[LatentDirichletAllocation] = None
        
    def load_and_clean_data(self) -> None:
        """Load and clean the review data with progress tracking."""
        try:
            logger.info("Loading data...")
            self.df = pd.read_csv(self.input_path)
            
            with tqdm(total=4, desc="Cleaning data") as pbar:
                # Clean missing values
                self.df = self.df.dropna(subset=['content'])
                pbar.update(1)
                
                # Convert dates
                self.df['at'] = pd.to_datetime(self.df['at'])
                self.df['repliedAt'] = pd.to_datetime(self.df['repliedAt'], errors='coerce')
                pbar.update(1)
                
                # Handle optional columns
                optional_columns = {
                    'reviewCreatedVersion': 'Unknown',
                    'appVersion': 'Unknown',
                    'replyContent': 'No Reply'
                }
                
                for col, default in optional_columns.items():
                    if col in self.df.columns:
                        self.df[col] = self.df[col].fillna(default)
                pbar.update(1)
                
                # Clean text content
                self.df['clean_content'] = self.preprocessor.batch_clean_texts(self.df['content'])
                pbar.update(1)
                
            logger.info(f"Loaded and cleaned {len(self.df)} reviews")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def extract_features(self, max_features: int = 50) -> None:
        """Extract TF-IDF features with progress tracking."""
        try:
            logger.info("Extracting features...")
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=max_features,
                ngram_range=(2, 3)
            )
            
            with tqdm(total=2, desc="Feature extraction") as pbar:
                # Fit vectorizer
                self.vectorizer.fit(self.df['clean_content'])
                pbar.update(1)
                
                # Transform text
                self.features = self.vectorizer.transform(self.df['clean_content'])
                pbar.update(1)
                
            logger.info(f"Extracted {max_features} TF-IDF features")
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def analyze_topics(self, n_topics: int = 5, n_top_words: int = 5) -> Dict[int, List[str]]:
        """Perform topic modeling with progress tracking."""
        try:
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42
            )
            
            with tqdm(total=2, desc="Topic modeling") as pbar:
                # Fit LDA model
                self.lda_model.fit(self.features)
                pbar.update(1)
                
                # Extract topics
                words = self.vectorizer.get_feature_names_out()
                topics = {}
                
                for idx, topic in enumerate(self.lda_model.components_):
                    top_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
                    topics[idx] = top_words
                pbar.update(1)
                
            return topics
        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
            raise

    def export_results(self) -> None:
        """Export analysis results with progress tracking."""
        try:
            with tqdm(total=3, desc="Exporting results") as pbar:
                # Calculate topic distribution
                topic_distribution = self.lda_model.transform(self.features)
                self.df['predicted_topic'] = topic_distribution.argmax(axis=1)
                pbar.update(1)
                
                # Prepare feature scores
                feature_scores = pd.DataFrame({
                    'Feature': self.vectorizer.get_feature_names_out(),
                    'Score': self.features.sum(axis=0).A1
                }).sort_values('Score', ascending=False)
                pbar.update(1)
                
                # Export files
                output_files = {
                    'reviews_with_topics.csv': self.df[['content', 'predicted_topic']],
                    'feature_scores.csv': feature_scores
                }
                
                for filename, data in output_files.items():
                    output_path = self.output_dir / filename
                    data.to_csv(output_path, index=False)
                pbar.update(1)
                
                logger.info("Results exported successfully")
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise

    def visualize_features(self, n_features: int = 10) -> None:
        """Create visualizations for top features."""
        try:
            output_dir = Path('/app/data')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = self.features.sum(axis=0).A1
            
            top_features_df = pd.DataFrame({
                'Feature': feature_names,
                'Score': tfidf_scores
            }).nlargest(n_features, 'Score')
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_features_df, x='Score', y='Feature')
            plt.title(f'Top {n_features} Most Important Features')
            plt.tight_layout()
            
            output_path = output_dir / 'top_features.png'
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Feature visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise

def main():
    """Main execution function with overall progress tracking."""
    try:
        input_path = '/app/data/truckerpath_1star_3620.csv'
        output_dir = '/app/data'
        
        analyzer = FeatureAnalyzer(input_path, output_dir)
        
        with tqdm(total=4, desc="Overall progress") as pbar:
            # Load and clean data
            analyzer.load_and_clean_data()
            pbar.update(1)
            
            # Extract features
            analyzer.extract_features()
            pbar.update(1)
            
            # Analyze topics
            topics = analyzer.analyze_topics()
            for topic_id, words in topics.items():
                logger.info(f"Topic {topic_id}: {', '.join(words)}")
            pbar.update(1)
            
            # Export results
            analyzer.export_results()
            pbar.update(1)
            
            # Visualize features
            analyzer.visualize_features()
            pbar.update(1)
            
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
