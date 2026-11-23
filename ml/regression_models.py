# ml/regression_models.py
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import pickle
import json
from typing import Dict, List, Optional
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
        self.training_sample_count = 0
        self.validation_sample_count = 0
        self.evaluation_sample_count = 0
        self.selected_regression_model = None
        self.validation_results = {}
        self.final_test_results = {}
        self.viral_threshold = None
        
    def prepare_training_data_with_nlp(
        self,
        texts: List[str],
        engagement_scores: List[float],
        comment_counts: List[int],
        duplicate_group_ids: Optional[List[str]] = None,
        author_ids: Optional[List[Optional[str]]] = None,
        author_names: Optional[List[Optional[str]]] = None,
        published_at: Optional[List[str]] = None,
    ):
        """Create training data using observed engagement labels."""
        if not (len(texts) == len(engagement_scores) == len(comment_counts)):
            raise ValueError("Texts, engagement scores, and comment counts must have equal lengths")
        if duplicate_group_ids is not None and len(duplicate_group_ids) != len(texts):
            raise ValueError("Duplicate group IDs must have the same length as texts")
        for values, name in (
            (author_ids, "author IDs"),
            (author_names, "author names"),
            (published_at, "publication dates"),
        ):
            if values is not None and len(values) != len(texts):
                raise ValueError(f"{name} must have the same length as texts")

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
        if duplicate_group_ids is not None:
            df['duplicate_group_id'] = duplicate_group_ids
        if author_ids is not None:
            df['author_id'] = author_ids
        if author_names is not None:
            df['author_name'] = author_names
        if published_at is not None:
            df['published_at'] = published_at
        
        # Predict whether a post received at least one observed comment.
        df['gets_comments'] = (np.asarray(comment_counts) > 0).astype(int)
        
        print(f"Created dataset with {len(df)} samples and {len(self.feature_names)} NLP features")
        print(f"Feature examples: {self.feature_names[:10]}...")
        
        return df
    
    def train_models(self, df: pd.DataFrame):
        """Tune on validation data and evaluate once on the final future holdout."""
        X = df[self.feature_names].values
        y_regression = df['engagement_score'].values
        y_comments = df['gets_comments'].values

        train_indices, validation_indices, test_indices = self._chronological_author_split_indices(df)
        self.training_sample_count = len(train_indices)
        self.validation_sample_count = len(validation_indices)
        self.evaluation_sample_count = len(test_indices)
        y_train = y_regression[train_indices]
        self.viral_threshold = float(np.percentile(y_train, 75))
        y_viral = (y_regression > self.viral_threshold).astype(int)

        X_train = X[train_indices]
        X_validation = X[validation_indices]
        validation_scaler = StandardScaler().fit(X_train)
        X_train_scaled = validation_scaler.transform(X_train)
        X_validation_scaled = validation_scaler.transform(X_validation)

        candidates = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
        }
        baseline_validation = float(np.mean(y_train))
        self.validation_results['baseline_mean'] = self._regression_metrics(
            y_regression[validation_indices],
            np.full(len(validation_indices), baseline_validation),
        )
        for name, model in candidates.items():
            model.fit(X_train_scaled, y_train)
            self.validation_results[name] = self._regression_metrics(
                y_regression[validation_indices],
                model.predict(X_validation_scaled),
            )

        self.selected_regression_model = min(
            self.validation_results,
            key=lambda name: self.validation_results[name]['mae'],
        )
        print("\n=== Validation Model Selection ===")
        for name, metrics in self.validation_results.items():
            print(f"{name}: MAE {metrics['mae']:.2f}, RMSE {metrics['rmse']:.2f}")
        print(f"Selected by validation MAE: {self.selected_regression_model}")

        # Refit using train + validation; the final future test remains untouched.
        fit_indices = np.concatenate((train_indices, validation_indices))
        X_fit = X[fit_indices]
        y_fit = y_regression[fit_indices]
        self.scaler = StandardScaler().fit(X_fit)
        X_fit_scaled = self.scaler.transform(X_fit)
        X_test_scaled = self.scaler.transform(X[test_indices])
        baseline_model = DummyRegressor(strategy='mean')
        baseline_model.fit(X_fit_scaled, y_fit)
        self.regression_models = {'baseline': baseline_model}
        for name, model in candidates.items():
            final_model = LinearRegression() if name == 'linear' else Ridge(alpha=1.0)
            final_model.fit(X_fit_scaled, y_fit)
            self.regression_models[name] = final_model

        y_viral_fit = y_viral[fit_indices]
        y_comments_fit = y_comments[fit_indices]
        print("\nTraining Logistic Regression (viral)...")
        logreg_viral = LogisticRegression(max_iter=1000, class_weight='balanced')
        logreg_viral.fit(X_fit_scaled, y_viral_fit)
        self.classification_models['viral'] = logreg_viral

        print("Training Logistic Regression (comments)...")
        logreg_comments = LogisticRegression(max_iter=1000)
        logreg_comments.fit(X_fit_scaled, y_comments_fit)
        self.classification_models['comments'] = logreg_comments

        test_targets = y_regression[test_indices]
        baseline_test = float(np.mean(y_fit))
        self.final_test_results = {
            'baseline_mean': self._regression_metrics(
                test_targets,
                np.full(len(test_indices), baseline_test),
            ),
            'linear': self._regression_metrics(
                test_targets,
                self.regression_models['linear'].predict(X_test_scaled),
            ),
            'ridge': self._regression_metrics(
                test_targets,
                self.regression_models['ridge'].predict(X_test_scaled),
            ),
            'comment_accuracy': float(accuracy_score(
                y_comments[test_indices],
                self.classification_models['comments'].predict(X_test_scaled),
            )),
            'viral_accuracy': float(accuracy_score(
                y_viral[test_indices],
                self.classification_models['viral'].predict(X_test_scaled),
            )),
        }
        print("\n=== Final Future Test (one-time evaluation) ===")
        reported_models = ['baseline_mean']
        reported_models.extend(('linear', 'ridge'))
        for name in reported_models:
            metrics = self.final_test_results[name]
            print(f"{name}: MAE {metrics['mae']:.2f}, RMSE {metrics['rmse']:.2f}")
            print(f"  Spearman rank correlation: {metrics['spearman']}")
        selected_test_name = 'baseline_mean' if self.selected_regression_model == 'baseline_mean' else self.selected_regression_model
        print(f"Validation-selected model: {selected_test_name}")
        print(f"Comment classification accuracy: {self.final_test_results['comment_accuracy']:.3f}")
        print(f"Viral classification accuracy: {self.final_test_results['viral_accuracy']:.3f}")

        self._analyze_feature_importance()
        return True
    
    def evaluate_models(self, df: pd.DataFrame):
        """Report stored final-test metrics; does not tune or resplit models."""
        if not self.final_test_results:
            raise RuntimeError("Train the models before requesting final-test metrics")
        print(self.final_test_results)

    @staticmethod
    def _regression_metrics(actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        actual_ranks = pd.Series(actual).rank(method='average').to_numpy()
        predicted_ranks = pd.Series(predicted).rank(method='average').to_numpy()
        if len(actual) < 2 or np.std(actual_ranks) == 0 or np.std(predicted_ranks) == 0:
            spearman = None
        else:
            spearman = float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1])
        return {
            'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'spearman': spearman,
        }

    @staticmethod
    def _chronological_author_split_indices(
        df: pd.DataFrame,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
    ):
        required = {'published_at', 'duplicate_group_id'}
        if not required.issubset(df.columns) or not ({'author_id', 'author_name'} & set(df.columns)):
            raise ValueError("Chronological split requires published_at, author identity, and duplicate_group_id")
        if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
            raise ValueError("Validation and test fractions must be positive and sum to less than one")
        if len(df) < 10:
            raise ValueError("At least 10 rows are required for chronological train/validation/test splits")

        dates = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
        if dates.isna().any():
            raise ValueError("Every row must have a valid publication date for chronological splitting")
        sorted_dates = dates.sort_values().reset_index(drop=True)
        validation_start = sorted_dates.iloc[int(len(df) * (1 - validation_fraction - test_fraction))]
        test_start = sorted_dates.iloc[int(len(df) * (1 - test_fraction))]

        author_ids = df['author_id'] if 'author_id' in df else pd.Series([None] * len(df), index=df.index)
        author_names = df['author_name'] if 'author_name' in df else pd.Series([None] * len(df), index=df.index)
        authors = []
        for index, (author_id, author_name) in enumerate(zip(author_ids, author_names)):
            if pd.notna(author_id) and str(author_id).strip():
                authors.append(f"id:{str(author_id).strip()}")
            elif pd.notna(author_name) and str(author_name).strip():
                authors.append(f"name:{str(author_name).strip().casefold()}")
            else:
                authors.append(f"unknown:{index}")
        authors = np.asarray(authors, dtype=object)
        duplicate_groups = df['duplicate_group_id'].astype(str).to_numpy()
        if pd.isna(df['duplicate_group_id']).any():
            raise ValueError("Every row must have a duplicate_group_id")

        date_values = np.asarray([date.value for date in dates], dtype=np.int64)
        validation_start_ns = validation_start.value
        test_start_ns = test_start.value
        test_indices = np.flatnonzero(date_values >= test_start_ns)
        test_authors = set(authors[test_indices])
        test_groups = set(duplicate_groups[test_indices])

        validation_candidates = np.flatnonzero(
            (date_values >= validation_start_ns) & (date_values < test_start_ns)
        )
        validation_indices = np.asarray([
            index for index in validation_candidates
            if authors[index] not in test_authors and duplicate_groups[index] not in test_groups
        ], dtype=int)
        validation_authors = set(authors[validation_indices])
        validation_groups = set(duplicate_groups[validation_indices])

        train_candidates = np.flatnonzero(date_values < validation_start_ns)
        excluded_authors = test_authors | validation_authors
        excluded_groups = test_groups | validation_groups
        train_indices = np.asarray([
            index for index in train_candidates
            if authors[index] not in excluded_authors and duplicate_groups[index] not in excluded_groups
        ], dtype=int)

        partitions = (train_indices, validation_indices, test_indices)
        if any(len(partition) == 0 for partition in partitions):
            raise ValueError("Chronological author/duplicate constraints produced an empty partition")
        for left in range(len(partitions)):
            for right in range(left + 1, len(partitions)):
                if set(authors[partitions[left]]) & set(authors[partitions[right]]):
                    raise ValueError("Author identity overlaps across chronological partitions")
                if set(duplicate_groups[partitions[left]]) & set(duplicate_groups[partitions[right]]):
                    raise ValueError("Duplicate groups overlap across chronological partitions")
        return train_indices, validation_indices, test_indices
    
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
    
    comment_counts = [24, 17, 0, 31, 12]
    return sample_posts, engagement_scores, comment_counts

def main():
    """Run a small synthetic demonstration of the model API."""
    sample_posts, sample_engagement, sample_comments = create_sample_dataset()
    sample_count = 50
    posts = [f"{sample_posts[index % len(sample_posts)]}\n\nSample variation {index}" for index in range(sample_count)]
    engagement = [sample_engagement[index % len(sample_engagement)] for index in range(sample_count)]
    comments = [sample_comments[index % len(sample_comments)] for index in range(sample_count)]
    duplicate_groups = [f"sample-content-{index}" for index in range(sample_count)]
    author_ids = [f"sample-author-{index}" for index in range(sample_count)]
    dates = pd.date_range("2025-01-01", periods=sample_count, freq="D", tz="UTC").astype(str).tolist()

    model = LinkedInNLPRegressionModels()
    df = model.prepare_training_data_with_nlp(
        posts, engagement, comments, duplicate_groups, author_ids, author_ids, dates
    )
    model.train_models(df)
    model.save_models()

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