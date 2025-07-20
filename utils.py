import pandas as pd

from datasets import load_dataset


def load_as_pd(dataset_name):
    dataset = load_dataset(dataset_name)
    data = dataset['train']  # 'train' split is used by default
    # Examine structure of a single sample
    print("Model type of data :")
    print(data.features)
    df = pd.DataFrame(data)
    return df


def calculate_stat_distr_len(df, column_name):
    len_stats = df[column_name].str.split().apply(len)
    # Statistical analysis
    print(f"\n{column_name} length stats:")
    print(len_stats.describe())


def display_examples(df, n=10, random_state=42):
    print("Sample Examples:")
    for i, row in df.sample(n=n, random_state=random_state).iterrows():
        print(f"\nExample {i+1}:")
        print(f"Instruction: {row['instruction']}")
        print(f"Input: {row['input']}")
        print(f"Output: {row['output']}")




