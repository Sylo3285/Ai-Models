"""
Dataset Downloader for DailyDialog
Downloads the bayes-group-diffusion/daily_dialog dataset from Hugging Face
and converts it to CSV format with input/output columns for easy training.
"""

from datasets import load_dataset
import pandas as pd
import os
from config import Config


def download_dailydialog():
    """Download and process DailyDialog dataset from Hugging Face."""
    
    print('=' * 80)
    print('DAILYDIALOG DATASET DOWNLOADER')
    print('=' * 80)
    print()
    
    # Create datasets directory if it doesn't exist
    Config.ensure_dirs()
    
    print('Downloading dataset from Hugging Face...')
    print('Dataset: bayes-group-diffusion/daily_dialog')
    print()
    
    try:
        # Load dataset
        dataset = load_dataset('bayes-group-diffusion/daily_dialog')
        
        print(f'Dataset loaded successfully!')
        print(f'Splits available: {list(dataset.keys())}')
        print()
        
        # Process all splits (train, validation, test)
        all_data = []
        
        for split_name in dataset.keys():
            print(f'Processing {split_name} split...')
            split_data = dataset[split_name]
            
            for example in split_data:
                # Each example has 'source' and 'target' fields
                all_data.append({
                    'input': example['source'].strip(),
                    'output': example['target'].strip()
                })
            
            print(f'  - Extracted {len([d for d in all_data])} pairs from {split_name}')
        
        print()
        print(f'Total conversational pairs: {len(all_data)}')
        print()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        output_path = Config.CSV_PATH
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print('=' * 80)
        print('DOWNLOAD COMPLETE!')
        print('=' * 80)
        print(f'Saved to: {output_path}')
        print(f'Total rows: {len(df)}')
        print()
        
        # Show sample data
        print('Sample data (first 5 rows):')
        print('-' * 80)
        for idx, row in df.head(5).iterrows():
            print(f"\nInput:  {row['input']}")
            print(f"Output: {row['output']}")
        print()
        
        # Show statistics
        print('=' * 80)
        print('DATASET STATISTICS')
        print('=' * 80)
        print(f'Total conversational pairs: {len(df)}')
        print(f'Average input length: {df["input"].str.len().mean():.1f} characters')
        print(f'Average output length: {df["output"].str.len().mean():.1f} characters')
        print(f'Unique inputs: {df["input"].nunique()}')
        print(f'Unique outputs: {df["output"].nunique()}')
        print()
        
        print('✅ Dataset ready for training!')
        print(f'   Use this file in train_csv.py by setting: CSV_PATH = "{output_path}"')
        print()
        
    except Exception as e:
        print(f'❌ Error downloading dataset: {e}')
        print()
        print('Make sure you have the datasets library installed:')
        print('  pip install datasets')
        return False
    
    return True


if __name__ == '__main__':
    download_dailydialog()
