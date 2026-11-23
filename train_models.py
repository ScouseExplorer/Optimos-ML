# train_models.py
"""
Train LinkedIn engagement prediction models using real post data
This script loads the JSON with 3000 LinkedIn posts and trains ML models
"""

import pandas as pd
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ml.regression_models import LinkedInNLPRegressionModels
from data.prepare_training_data import prepare_dataset
import pickle

def load_linkedin_posts(json_path: str = 'data/training_posts_canonical.json'):
    """Load canonical posts and keep only rows with complete count targets."""
    print(f"Loading LinkedIn posts from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        canonical_data = json.load(f)

    if not isinstance(canonical_data, dict) or canonical_data.get('schema_version') != '1.0':
        raise ValueError("Expected a canonical dataset with schema_version '1.0'; run data.prepare_training_data first")
    posts = canonical_data.get('records')
    if not isinstance(posts, list):
        raise ValueError("Canonical dataset must contain a records list")

    df = pd.json_normalize(posts)
    required_columns = [
        'post_id', 'text', 'published_at', 'metrics_captured_at',
        'engagement.likes', 'engagement.comments', 'engagement.shares',
        'author_id', 'author_name', 'author_type', 'search_query', 'duplicate_group_id',
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Canonical dataset is missing required fields: {', '.join(missing_columns)}")

    df = df[required_columns].copy()

    metric_columns = ['engagement.likes', 'engagement.comments', 'engagement.shares']
    for column in metric_columns:
        raw_values = df[column]
        values = pd.to_numeric(raw_values, errors='coerce')
        if (raw_values.notna() & values.isna()).any() or (values.dropna() < 0).any():
            raise ValueError(f"Canonical field {column} must contain nonnegative numeric counts or null")
        df[column] = values

    complete_targets = df[metric_columns].notna().all(axis=1)
    excluded_rows = int((~complete_targets).sum())
    df = df.loc[complete_targets].copy()
    if df.empty:
        raise ValueError("No posts have complete likes, comments, and shares targets")

    df['engagement_score'] = df[metric_columns].sum(axis=1)
    df.attrs['excluded_incomplete_target_rows'] = excluded_rows
    print(f"✓ Loaded {len(df)} posts with complete engagement targets")
    if excluded_rows:
        print(f"⚠ Excluded {excluded_rows} posts with missing target metrics; canonical values remain unchanged")
    print("✓ Search queries and author metadata are retained as provenance, not model features")
    return df

def train_and_save_models(output_dir: str = 'models', allow_legacy_experimental: bool = False):
    """Complete experimental training pipeline; it cannot enable forecasting."""
    if not allow_legacy_experimental:
        raise RuntimeError(
            "Forecasting retraining is disabled: the master dataset lacks impressions, "
            "a capture-time observation window and an approved forecasting target. "
            "Use --allow-legacy-experimental only for explicitly labeled historical model experiments."
        )
    
    # Create models directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Canonicalize source data and retain an auditable rejection report.
    source_path = project_root / 'data' / 'linkedin_master_dataset_final.json'
    canonical_path = project_root / 'data' / 'training_posts_canonical.json'
    rejected_path = project_root / 'data' / 'training_posts_rejected.json'
    preparation = prepare_dataset(str(source_path), str(canonical_path), rejected_path=str(rejected_path))
    print(f"Canonical data: {preparation['summary']}")

    # 2. Load valid records with complete model targets.
    df = load_linkedin_posts(str(canonical_path))
    texts = df['text'].tolist()

    # 3. Use observed interactions as training targets
    engagement_scores = df['engagement_score'].tolist()
    comment_counts = df['engagement.comments'].tolist()
    
    # 4. Initialize model trainer
    print("\nInitializing LinkedIn NLP Regression Models...")
    trainer = LinkedInNLPRegressionModels()
    
    # 5. Prepare training data with NLP feature extraction
    print("\nPreparing training data (this may take a few minutes)...")
    training_df = trainer.prepare_training_data_with_nlp(
        texts,
        engagement_scores,
        comment_counts,
        df['duplicate_group_id'].tolist(),
        df['author_id'].tolist(),
        df['author_name'].tolist(),
        df['published_at'].tolist(),
    )
    
    # 6. Train models
    print("\nTraining models...")
    trainer.train_models(training_df)
    
    # 7. Evaluate models
    print("\nEvaluating models...")
    print("Evaluation is performed on the held-out split during training.")
    
    # 8. Save models to disk
    print(f"\nSaving models to {output_dir}/...")
    
    # Save trained model object
    model_path = os.path.join(output_dir, 'linkedin_models.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'regression_models': trainer.regression_models,
            'classification_models': trainer.classification_models,
            'scaler': trainer.scaler,
            'feature_names': trainer.feature_names,
            'training_metadata': {
                'dataset_samples': len(training_df),
                'training_samples': trainer.training_sample_count,
                'validation_samples': trainer.validation_sample_count,
                'evaluation_samples': trainer.evaluation_sample_count,
                'engagement_target': 'observed likes + comments + shares',
                'comment_target': 'observed comments > 0',
                'excluded_incomplete_target_rows': df.attrs['excluded_incomplete_target_rows'],
                'canonical_schema_version': preparation['schema_version'],
                'duplicate_split_grouping': 'duplicate_group_id',
                'near_duplicate_similarity_threshold': preparation['near_duplicate_similarity_threshold'],
                'metrics_capture_time_missing': preparation['summary']['records_missing_metrics_capture_time'],
                'selected_regression_model': trainer.selected_regression_model,
                'validation_metrics': trainer.validation_results,
                'final_test_metrics': trainer.final_test_results,
                'prediction_model_available': False,
                'model_schema_compatible': False,
                'post_level_split_verified': True,
                'locked_final_test': True,
                'validation_status': 'experimental_only',
            },
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
        f.write(f"Training Samples: {trainer.training_sample_count}\n")
        f.write(f"Validation Samples: {trainer.validation_sample_count}\n")
        f.write(f"Evaluation Samples: {trainer.evaluation_sample_count}\n")
        f.write(f"Number of Features: {len(trainer.feature_names)}\n")
        f.write("Engagement Target: Observed likes + comments + shares\n")
        f.write("Comment Target: Observed comments > 0\n")
        f.write(f"Exact Duplicate Groups: {preparation['summary']['exact_duplicate_groups']}\n")
        f.write(f"Near Duplicate Groups: {preparation['summary']['near_duplicate_groups']}\n")
        f.write(f"Metrics Capture Time Missing: {preparation['summary']['records_missing_metrics_capture_time']}\n")
        f.write(f"Selected Regression Model: {trainer.selected_regression_model}\n")
        for name, metrics in trainer.validation_results.items():
            f.write(f"Validation {name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}\n")
        for name, metrics in trainer.final_test_results.items():
            if isinstance(metrics, dict):
                f.write(
                    f"Final Test {name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                    f"Spearman={metrics['spearman']}\n"
                )
            else:
                f.write(f"Final Test {name}: {metrics:.4f}\n")
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
    print("Next step: Restart the API to load the newly trained models.")
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-legacy-experimental",
        action="store_true",
        help="Train an explicitly labeled legacy raw-count experiment; not for forecasting.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LinkedIn Growth AI - Model Training Pipeline")
    print("=" * 60)
    print()
    
    try:
        trainer = train_and_save_models(allow_legacy_experimental=args.allow_legacy_experimental)
        print("\n✓ Success! Your models are trained and ready to use.")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
