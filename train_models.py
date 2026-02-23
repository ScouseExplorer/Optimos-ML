# train_models.py
"""
Train LinkedIn engagement prediction models using real post data
This script loads the CSV with 1,608 LinkedIn posts and trains ML models
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ml.regression_models import LinkedInNLPRegressionModels
from nlp.feature_extractor import NLPFeatureExtractor
import pickle

def load_linkedin_posts(csv_path: str = 'data/linkedin_posts.csv'):
    """Load LinkedIn posts from CSV"""
    print(f"Loading LinkedIn posts from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} posts")
    print(f"✓ Columns: {list(df.columns)}")
    return df

def generate_synthetic_engagement_scores(texts, seed=42):
    """
    Generate realistic engagement scores based on text features
    Since your CSV doesn't have actual engagement metrics, we simulate them
    based on text characteristics (until you collect real engagement data)
    """
    np.random.seed(seed)
    extractor = NLPFeatureExtractor()
    
    engagement_scores = []
    
    print("\nGenerating engagement scores based on text features...")
    for i, text in enumerate(texts):
        if i % 200 == 0:
            print(f"  Processing post {i}/{len(texts)}...")
        
        # Extract quick features
        features = extractor.extract_all_features(text)
        
        # Calculate engagement based on known patterns
        base_score = 10  # Base engagement
        
        # Add score based on features
        score = base_score
        score += features.get('word_count', 0) * 0.05  # Longer posts
        score += features.get('hashtag_count', 0) * 2.5  # More hashtags
        score += features.get('has_question', 0) * 8  # Questions
        score += features.get('has_emoji', 0) * 6  # Emojis
        score += features.get('sentiment_polarity', 0) * 10  # Positive sentiment
        score += features.get('engagement_trigger_count', 0) * 3  # CTAs
        score += features.get('power_word_ratio', 0) * 100  # Power words
        
        # Add some randomness (real engagement varies)
        score += np.random.normal(0, 15)
        
        # Ensure positive
        score = max(5, score)
        
        engagement_scores.append(score)
    
    print(f"✓ Generated {len(engagement_scores)} engagement scores")
    print(f"  Mean: {np.mean(engagement_scores):.1f}, Median: {np.median(engagement_scores):.1f}")
    print(f"  Min: {np.min(engagement_scores):.1f}, Max: {np.max(engagement_scores):.1f}")
    
    return engagement_scores

def train_and_save_models(output_dir: str = 'models'):
    """Complete training pipeline"""
    
    # Create models directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    df = load_linkedin_posts()
    texts = df['Post Content'].tolist()
    
    # 2. Generate engagement scores (or load from CSV if you have them)
    # TODO: Replace this with real engagement data when available
    engagement_scores = generate_synthetic_engagement_scores(texts)
    
    # 3. Initialize model trainer
    print("\nInitializing LinkedIn NLP Regression Models...")
    trainer = LinkedInNLPRegressionModels()
    
    # 4. Prepare training data with NLP feature extraction
    print("\nPreparing training data (this may take a few minutes)...")
    training_df = trainer.prepare_training_data_with_nlp(texts, engagement_scores)
    
    # 5. Train models
    print("\nTraining models...")
    trainer.train_models(training_df)
    
    # 6. Evaluate models
    print("\nEvaluating models...")
    trainer.evaluate_models(training_df)
    
    # 7. Save models to disk
    print(f"\nSaving models to {output_dir}/...")
    
    # Save trained model object
    model_path = os.path.join(output_dir, 'linkedin_models.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'regression_models': trainer.regression_models,
            'classification_models': trainer.classification_models,
            'scaler': trainer.scaler,
            'feature_names': trainer.feature_names
        }, f)
    print(f"✓ Saved models to {model_path}")
    
    # Save feature names for reference
    feature_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_path, 'w') as f:
        f.write('\n'.join(trainer.feature_names))
    print(f"✓ Saved {len(trainer.feature_names)} feature names to {feature_path}")
    
    # Save training info
    info_path = os.path.join(output_dir, 'model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"LinkedIn Growth AI - Model Training Info\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Number of Posts: {len(texts)}\n")
        f.write(f"Number of Features: {len(trainer.feature_names)}\n")
        f.write(f"Models Trained:\n")
        f.write(f"  - Linear Regression (engagement)\n")
        f.write(f"  - Ridge Regression (engagement)\n")
        f.write(f"  - Logistic Regression (viral prediction)\n")
        f.write(f"  - Logistic Regression (comment prediction)\n")
    print(f"✓ Saved model info to {info_path}")
    
    print("\n" + "=" * 60)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nYour models are ready in the '{output_dir}/' directory")
    print(f"Next step: Update app.py to load these models")
    
    return trainer

if __name__ == "__main__":
    print("=" * 60)
    print("LinkedIn Growth AI - Model Training Pipeline")
    print("=" * 60)
    print()
    
    try:
        trainer = train_and_save_models()
        print("\n✓ Success! Your models are trained and ready to use.")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
