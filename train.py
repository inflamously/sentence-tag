# --- Fine-tuning GLiNER for NER Example
import ast
import re

from gliner import GLiNER
from gliner.data_processing import DataCollator
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoTokenizer
from datasets import load_dataset, tqdm

if __name__ == "__main__":
    # 1. Load the pretrained GLiNER model.
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")

    # 2. Prepare a dataset.
    ds = load_dataset("numind/NuNER")


    # For simplicity, split the dataset into training and evaluation halves.
    # split_dataset = ds.train_test_split(test_size=0.3)
    # train_dataset = split_dataset["train"]
    # eval_dataset = split_dataset["test"]

    # 3. Define a tokenization function that also aligns the labels to the sub-tokens.
    def tokenize_text(text):
        """Tokenizes the input text into a list of tokens."""
        if not text: return ['']
        words = re.findall(r'\w+(?:[-_]\w+)*|\S', text)
        return list(map(lambda w: w.replace("'", '"'), words))


    def tokenize_and_align_labels(samples):
        """
            Original code: https://github.com/urchade/GLiNER/blob/main/data/process_nuner.py
            Modified to match train.py
            Processes entities in the dataset to extract tokenized text and named entity spans.
        """
        tokenized_text = [tokenize_text(sample) for sample in samples["input"]]
        parsed_output = [ast.literal_eval(sample) for sample in samples["output"]]

        mapped_results = {
            "input": [],
            "output": [],
        }

        for tokenized_text, entity_output in zip(tokenized_text, parsed_output):
            if any(map(lambda ent: ent.count("<>") != 1, entity_output)):
                mapped_results["input"].append(tokenized_text)
                mapped_results["output"].append([{"start": -1, "end": -1, "entity": "N/A"}])
                continue

            entity_output_clean = list(filter(lambda x: len(x) > 0, entity_output))
            entity_texts, entity_types = zip(*[output.split("<>") for output in entity_output_clean])
            entity_spans = []
            for j, entity_text in enumerate(entity_texts):
                entity_tokens = tokenize_text(entity_text)
                matches = []
                for i in range(len(tokenized_text) - len(entity_tokens) + 1):
                    if " ".join(tokenized_text[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                        matches.append({
                            "start": i,
                            "end": i + len(entity_tokens) - 1,
                            "entity": entity_types[j]
                        })
                if matches:
                    entity_spans.extend(matches)

            mapped_results["input"].append(tokenized_text)
            mapped_results["output"].append(entity_spans)

        eval_count = int(len(mapped_results) * 0.15)
        return {
            "train": {"tokenized_text": mapped_results["input"][eval_count:],
                      "ner": mapped_results["output"][eval_count:]},
            "eval": {"tokenized_text": mapped_results["input"][:eval_count],
                     "ner": mapped_results["output"][:eval_count]},
        }


    # Tokenize both training and evaluation datasets.
    processed_dataset = ds.map(tokenize_and_align_labels, batched=True)

    # 4. Create a data collator
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # 5. Set up training arguments.
    num_steps = 500
    batch_size = 8
    data_size = len(processed_dataset["train"])
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=5e-6,
        weight_decay=0.01,
        # others_lr=1e-5,
        # others_weight_decay=0.01,
        lr_scheduler_type="linear",  # cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        save_steps=100,
        save_total_limit=10,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["eval"],
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
