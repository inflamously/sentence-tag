# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-360M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

from transformers import pipeline
# Let's test the base model before training
# prompt = """
# Meta Prompt:
# "You are an AI designed to classify a given sentence into the categories medium, type, format, and subject based on the provided guidelines.
#
# Goal:
# Accurately categorize sentence 'x' into the given categories.
#
# Answer Format:
# Provide the classification as a structured output:
#
# Medium: [Specify the medium (e.g., text, audio, video, etc.)]
# Type: [Describe the type (e.g., informative, narrative, instructional, etc.)]
# Format: [Identify the format (e.g., essay, podcast, report, etc.)]
# Subject: [Determine the subject matter (e.g., science, history, technology, etc.)]
# Do Not Do's:
# Do not infer information beyond what is explicitly given.
# Do not assume additional context that is not present in the sentence.
# Do not provide multiple conflicting classifications; select the most accurate one.
# Do not explain the reasoning unless explicitly requested.
# Context:
# Your task is to analyze sentence 'x' and classify it solely based on its structure, wording, and implied meaning. If any category is ambiguous, provide the most probable classification."
#
# Here is your sentence x: "2008 Ford Mustang"
#
# """
prompt = "What is the primary function of mitochondria within a cell?"

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
pipe(prompt, max_new_tokens=200)
print(pipe.predict(prompt))
# [{'generated_text': 'What is the primary function of mitochondria within a cell?\n\The function of the mitochondria is to produce energy for the cell through a process called cellular respiration.'}]

from datasets import load_dataset

ds = load_dataset("argilla/synthetic-concise-reasoning-sft-filtered")
def tokenize_function(examples):
    examples["text"] = tokenizer.apply_chat_template([{"role": "user", "content": examples["prompt"].strip()}, {"role": "assistant", "content": examples["completion"].strip()}], tokenize=False)
    return examples
ds = ds.map(tokenize_function)

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=1,
    gradient_checkpointing=True,
    per_device_train_batch_size=8,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=100,  # Frequency of logging training metrics
    use_mps_device= True if device == "mps" else False,
    hub_model_id="argilla/SmolLM2-360M-synthetic-concise-reasoning",  # Set a unique name for your model
    push_to_hub=False, # TODO
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    tokenizer=tokenizer,
)
trainer.train()
# {'loss': 1.4498, 'grad_norm': 2.3919131755828857, 'learning_rate': 4e-05, 'epoch': 0.1}
# {'loss': 1.362, 'grad_norm': 1.6650595664978027, 'learning_rate': 3e-05, 'epoch': 0.19}
# {'loss': 1.3778, 'grad_norm': 1.4778285026550293, 'learning_rate': 2e-05, 'epoch': 0.29}
# {'loss': 1.3735, 'grad_norm': 2.1424977779388428, 'learning_rate': 1e-05, 'epoch': 0.39}
# {'loss': 1.3512, 'grad_norm': 2.3498542308807373, 'learning_rate': 0.0, 'epoch': 0.48}
# {'train_runtime': 1911.514, 'train_samples_per_second': 1.046, 'train_steps_per_second': 0.262, 'train_loss': 1.3828572998046875, 'epoch': 0.48}
