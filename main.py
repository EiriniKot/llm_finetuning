from utils import load_as_pd, calculate_stat_distr_len, display_examples
from model import LLMModel, model_info


# df_set = load_as_pd("yahma/alpaca-cleaned")
# calculate_stat_distr_len(df_set, column_name='instruction')
# calculate_stat_distr_len(df_set, column_name='output')
# display_examples(df_set, n=10, random_state=42)
#
# # Create smaller subset of 100 examples
# small_subset = df_set.sample(100, random_state=123).reset_index(drop=True)

# Optionally, save it to disk for later use
# small_subset.to_json("alpaca_subset_100.json", orient="records", lines=True)


model_obj = LLMModel("Qwen/Qwen3-0.6B")
output = model_obj.generate("Explain the interview process of an ML Engineer")
print(output[0]['generated_text'])

model_info()
# ✅ Running on GPU: NVIDIA GeForce RTX 3050 Laptop GPU
# GPU memory allocated: 2.39 GB