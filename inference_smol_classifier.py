import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "./sft_output/checkpoint-517"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
# model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

prompt = "What is the primary function of mitochondria within a cell?"

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
pipe(prompt, max_new_tokens=250)
print(pipe.predict(prompt))
