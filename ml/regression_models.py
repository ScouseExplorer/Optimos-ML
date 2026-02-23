# ml/regression_models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pickle
import json
from typing import Dict, List
import sys
import os

# Add nlp module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.feature_extractor import NLPFeatureExtractor

class LinkedInNLPRegressionModels:
    """Regression models using NLP features as input"""
    
    def __init__(self):
        self.nlp_extractor = NLPFeatureExtractor()
        self.regression_models = {}
        self.classification_models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_training_data_with_nlp(self, texts: List[str], engagement_scores: List[float]):
        """Create training data from text content"""
        print("Extracting NLP features from texts...")
        
        # Extract NLP features for each text
        nlp_features = []
        for text in texts:
            features = self.nlp_extractor.extract_all_features(text)
            nlp_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(nlp_features)
        self.feature_names = df.columns.tolist()
        
        # Add engagement scores
        df['engagement_score'] = engagement_scores
        
        # Create binary classification targets
        engagement_threshold = np.percentile(engagement_scores, 75)
        df['is_viral'] = (df['engagement_score'] > engagement_threshold).astype(int)
        
        # Create comment prediction target (simulated)
        df['gets_comments'] = ((df['engagement_score'] > np.median(engagement_scores)) & 
                              (df['has_question'] > 0)).astype(int)
        
        print(f"Created dataset with {len(df)} samples and {len(self.feature_names)} NLP features")
        print(f"Feature examples: {self.feature_names[:10]}...")
        
        return df
    
    def train_models(self, df: pd.DataFrame):
        """Train regression and classification models"""
        # Prepare features and targets
        X = df[self.feature_names].values
        y_regression = df['engagement_score'].values
        y_viral = df['is_viral'].values
        y_comments = df['gets_comments'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_regression, test_size=0.2, random_state=42
        )
        
        _, _, y_train_viral, y_test_viral = train_test_split(
            X_scaled, y_viral, test_size=0.2, random_state=42
        )
        
        _, _, y_train_comments, y_test_comments = train_test_split(
            X_scaled, y_comments, test_size=0.2, random_state=42
        )
        
        # 1. Linear Regression for engagement prediction
        print("\nTraining Linear Regression for engagement...")
        lr = LinearRegression()
        lr.fit(X_train, y_train_reg)
        self.regression_models['linear'] = lr
        
        # 2. Ridge Regression (regularized)
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train_reg)
        self.regression_models['ridge'] = ridge
        
        # 3. Logistic Regression for viral prediction
        print("Training Logistic Regression (viral)...")
        logreg_viral = LogisticRegression(max_iter=1000, class_weight='balanced')
        logreg_viral.fit(X_train, y_train_viral)
        self.classification_models['viral'] = logreg_viral
        
        # 4. Logistic Regression for comment prediction
        print("Training Logistic Regression (comments)...")
        logreg_comments = LogisticRegression(max_iter=1000)
        logreg_comments.fit(X_train, y_train_comments)
        self.classification_models['comments'] = logreg_comments
        
        # Evaluate models
        self._evaluate_models(X_test, y_test_reg, y_test_viral, y_test_comments)
        
        # Analyze feature importance
        self._analyze_feature_importance()
        
        return True
    
    def evaluate_models(self, df: pd.DataFrame):
        """Public method to evaluate models on a dataset"""
        # Prepare features and targets
        X = df[self.feature_names].values
        y_regression = df['engagement_score'].values
        y_viral = df['is_viral'].values
        y_comments = df['gets_comments'].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Train-test split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_regression, test_size=0.2, random_state=42
        )
        
        _, _, y_train_viral, y_test_viral = train_test_split(
            X_scaled, y_viral, test_size=0.2, random_state=42
        )
        
        _, _, y_train_comments, y_test_comments = train_test_split(
            X_scaled, y_comments, test_size=0.2, random_state=42
        )
        
        # Evaluate
        self._evaluate_models(X_test, y_test_reg, y_test_viral, y_test_comments)
    
    def _evaluate_models(self, X_test, y_test_reg, y_test_viral, y_test_comments):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")
        
        # Regression models
        for name, model in self.regression_models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test_reg, y_pred)
            r2 = r2_score(y_test_reg, y_pred)
            print(f"{name.upper()} Regression - MSE: {mse:.2f}, R²: {r2:.3f}")
        
        # Classification models
        for name, model in self.classification_models.items():
            y_test = y_test_viral if name == 'viral' else y_test_comments
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name.upper()} Classification - Accuracy: {accuracy:.3f}")
    
    def _analyze_feature_importance(self):
        """Analyze which NLP features are most important"""
        if 'linear' not in self.regression_models:
            return
        
        lr_model = self.regression_models['linear']
        coefficients = lr_model.coef_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n=== Top 10 Most Important NLP Features ===")
        for idx, row in importance_df.head(10).iterrows():
            effect = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"{row['feature']}: {effect} engagement by {abs(row['coefficient']):.3f} points")
        
        # Save feature importance
        os.makedirs('ml/models', exist_ok=True)
        importance_df.to_csv('ml/models/feature_importance.csv', index=False)
        
        return importance_df
    
    def predict_from_text(self, text: str) -> Dict:
        """Make predictions from raw text"""
        # Extract NLP features
        features_dict = self.nlp_extractor.extract_all_features(text)
        
        # Convert to array in correct order
        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = {}
        
        # Regression predictions
        for name, model in self.regression_models.items():
            predictions[f'{name}_engagement'] = float(model.predict(X_scaled)[0])
        
        # Classification predictions
        for name, model in self.classification_models.items():
            proba = model.predict_proba(X_scaled)[0]
            predictions[f'{name}_probability'] = float(proba[1])  # Positive class probability
            predictions[f'is_{name}'] = bool(model.predict(X_scaled)[0])
        
        # Get optimization suggestions based on features
        suggestions = self._generate_optimization_suggestions(features_dict)
        
        return {
            'predictions': predictions,
            'nlp_features': {k: float(v) for k, v in features_dict.items()},
            'optimization_suggestions': suggestions
        }
    
    def _generate_optimization_suggestions(self, features_dict: Dict) -> List[str]:
        """Generate content optimization suggestions based on NLP features"""
        suggestions = []
        
        # Engagement keyword suggestions
        if features_dict.get('engagement_keyword_score', 0) < 2:
            suggestions.append("Add more engagement keywords: 'insight', 'learn', 'share', 'excited'")
        
        # CTA suggestions
        if features_dict.get('cta_score', 0) < 1:
            suggestions.append("Add a call-to-action: 'What are your thoughts?' or 'Share your experience'")
        
        # Question suggestions
        if features_dict.get('has_question', 0) == 0:
            suggestions.append("Add a question to encourage comments and engagement")
        
        # Length suggestions
        word_count = features_dict.get('word_count', 0)
        if word_count < 100:
            suggestions.append("Post is too short. Aim for 150-300 words for better engagement")
        elif word_count > 400:
            suggestions.append("Post is very long. Consider breaking into multiple posts or using bullet points")
        
        # Hashtag suggestions
        hashtag_count = features_dict.get('hashtag_count', 0)
        if hashtag_count < 3:
            suggestions.append("Add 2-3 more relevant hashtags")
        elif hashtag_count > 8:
            suggestions.append("Too many hashtags. Limit to 5-7 for optimal performance")
        
        # Readability suggestions
        flesch_score = features_dict.get('flesch_reading_ease', 0)
        if flesch_score < 60:
            suggestions.append("Content may be too complex. Simplify sentence structure")
        
        # Sentiment suggestions
        sentiment = features_dict.get('sentiment_polarity', 0)
        if sentiment < -0.3:
            suggestions.append("Content is negative. Consider more positive phrasing")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def save_models(self, path: str = 'ml/models/'):
        """Save all models and metadata"""
        os.makedirs(path, exist_ok=True)
        
        # Save regression models
        for name, model in self.regression_models.items():
            with open(f'{path}/{name}_regression.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save classification models
        for name, model in self.classification_models.items():
            with open(f'{path}/{name}_classification.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(f'{path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_count': len(self.regression_models) + len(self.classification_models),
            'feature_count': len(self.feature_names)
        }
        
        with open(f'{path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {path}")
        
    def load_models(self, path: str = 'ml/models/'):
        """Load trained models"""
        # Load metadata
        with open(f'{path}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        
        # Load models
        model_files = os.listdir(path)
        
        for file in model_files:
            if file.endswith('_regression.pkl'):
                name = file.replace('_regression.pkl', '')
                with open(f'{path}/{file}', 'rb') as f:
                    self.regression_models[name] = pickle.load(f)
            elif file.endswith('_classification.pkl'):
                name = file.replace('_classification.pkl', '')
                with open(f'{path}/{file}', 'rb') as f:
                    self.classification_models[name] = pickle.load(f)
            elif file == 'scaler.pkl':
                with open(f'{path}/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
        
        print(f"Loaded {len(self.regression_models)} regression and {len(self.classification_models)} classification models")

# Example usage
def create_sample_dataset():
    """Create sample dataset with LinkedIn posts"""
    sample_posts = [
        "Excited to announce our new AI-powered platform! 🚀 After months of hard work, we're launching something special. #AI #Tech #Innovation",
        "Just read an amazing article about machine learning in healthcare. The potential is incredible! What are your thoughts on AI in medicine?",
        "Another day, another meeting. Sometimes work feels overwhelming. #MondayBlues",
        "Proud to share that our team won the innovation award! This wouldn't be possible without our amazing collaborators. #TeamWork #Success",
        "Looking for recommendations: best tools for remote team collaboration? Our company is transitioning to hybrid work. #RemoteWork #Tools",
    ]
    
    # Simulated engagement scores (likes + comments + shares)
    engagement_scores = [245, 189, 32, 312, 156]
    
    return sample_posts, engagement_scores

def main():
    """Main training pipeline"""
    print("Training NLP-enhanced regression models...")
    
    # Create sample data
    posts, engagement = create_sample_dataset()
    
    # Initialize model
    model = LinkedInNLPRegressionModels()
    
    # Prepare training data
    df = model.prepare_training_data_with_nlp(posts, engagement)
    
    # Train models
    model.train_models(df)
    
    # Save models
    model.save_models()
    
    # Test prediction
    test_text = "Launching our new product next week! Can't wait to show you what we've been working on. #Launch #Product #Excited"
    prediction = model.predict_from_text(test_text)
    
    print("\n=== Test Prediction ===")
    print(f"Text: {test_text[:50]}...")
    print(f"Predicted Engagement: {prediction['predictions']['linear_engagement']:.1f}")
    print(f"Viral Probability: {prediction['predictions']['viral_probability']:.2%}")
    print("\nOptimization Suggestions:")
    for suggestion in prediction['optimization_suggestions']:
        print(f"• {suggestion}")
    
    return model

if __name__ == "__main__":
    main()