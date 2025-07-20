import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMModel:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt, max_new_tokens=200):
        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
        output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True)
        return output


def model_info():
    if torch.cuda.is_available():
        print(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("✅ Running on CPU (no CUDA available)")
