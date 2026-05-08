from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./Qwen3.6-27B-pruned')
ids = tok.encode('def fibonacci(n: int) -> int:')
print(tok.decode(ids))