# update_training_data.py
"""
Update training data with timestamps and engagement metrics
Run this after you've collected additional data from LinkedIn
"""

import pandas as pd
import numpy as np
from pathlib import Path

def update_with_timestamps():
    """
    Add timestamp-based features to your training data
    """
    
    print("\n" + "="*60)
    print("LinkedIn Growth AI - Data Update Script")
    print("="*60)
    
    # Check if new data file exists
    new_data_path = 'data/linkedin_posts_with_metrics.csv'
    
    if not Path(new_data_path).exists():
        print(f"\n⚠️  File not found: {new_data_path}")
        print("\nSteps to add your data:")
        print("1. Export your posts from LinkedIn with timestamps")
        print("2. Save as: data/linkedin_posts_with_metrics.csv")
        print("3. Run this script again")
        print("\nRequired columns:")
        print("  - Post Content (text)")
        print("  - Posted At (timestamp: YYYY-MM-DD HH:MM:SS)")
        print("\nOptional columns (for better accuracy):")
        print("  - Impressions")
        print("  - Likes")
        print("  - Comments") 
        print("  - Shares")
        return False
    
    # Load the new data
    print(f"\n✓ Loading data from: {new_data_path}")
    df = pd.read_csv(new_data_path)
    
    print(f"✓ Loaded {len(df)} posts")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check required columns
    if 'Post Content' not in df.columns:
        print("\n❌ Missing required column: 'Post Content'")
        return False
    
    # Process timestamps if available
    if 'Posted At' in df.columns:
        print("\n📅 Processing timestamps...")
        df['Posted At'] = pd.to_datetime(df['Posted At'], errors='coerce')
        
        # Extract time features
        df['hour_of_day'] = df['Posted At'].dt.hour
        df['day_of_week'] = df['Posted At'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_name'] = df['Posted At'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['hour_of_day'].isin([7, 8, 9, 12, 17, 18]).astype(int)
        
        # Show distribution
        print("\n📊 Posting Time Distribution:")
        print("\nTop posting hours:")
        print(df['hour_of_day'].value_counts().head(5))
        print("\nPosts by day:")
        print(df['day_name'].value_counts())
        
        valid_timestamps = df['Posted At'].notna().sum()
        print(f"\n✓ Processed {valid_timestamps}/{len(df)} timestamps")
    else:
        print("\n⚠️  No 'Posted At' column found - skipping time features")
    
    # Process engagement metrics if available
    engagement_cols = ['Impressions', 'Likes', 'Comments', 'Shares', 'Clicks']
    available_metrics = [col for col in engagement_cols if col in df.columns]
    
    if available_metrics:
        print(f"\n📈 Found engagement metrics: {', '.join(available_metrics)}")
        
        # Calculate total engagement
        if all(col in df.columns for col in ['Likes', 'Comments', 'Shares']):
            # Weighted engagement (comments and shares are more valuable)
            df['engagement_score'] = (
                df['Likes'] + 
                df['Comments'] * 2 + 
                df['Shares'] * 3
            )
            
            if 'Impressions' in df.columns:
                # Engagement rate (percentage)
                df['engagement_rate'] = (df['engagement_score'] / df['Impressions'] * 100).fillna(0)
                print(f"\n✓ Calculated engagement rate")
                print(f"  Mean: {df['engagement_rate'].mean():.2f}%")
                print(f"  Median: {df['engagement_rate'].median():.2f}%")
            
            print(f"\n✓ Calculated engagement score")
            print(f"  Mean: {df['engagement_score'].mean():.1f}")
            print(f"  Median: {df['engagement_score'].median():.1f}")
            print(f"  Max: {df['engagement_score'].max():.1f}")
        else:
            print("\n⚠️  Incomplete engagement metrics - need Likes, Comments, Shares")
    else:
        print("\n⚠️  No engagement metrics found - will use synthetic scores")
    
    # Save processed data
    output_path = 'data/training_data_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved processed data to: {output_path}")
    
    # Show summary
    print("\n" + "="*60)
    print("📊 Data Summary")
    print("="*60)
    print(f"Total posts: {len(df)}")
    if 'Posted At' in df.columns:
        print(f"Posts with timestamps: {df['Posted At'].notna().sum()}")
    if available_metrics:
        print(f"Posts with engagement data: {df[available_metrics[0]].notna().sum()}")
    
    print("\n" + "="*60)
    print("✅ Data update complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review: data/training_data_processed.csv")
    print("2. Retrain: python train_models.py")
    print("3. Test: Your predictions will be much more accurate!")
    
    return True

if __name__ == "__main__":
    update_with_timestamps()
