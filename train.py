import ast
import os
import re

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from gliner import GLiNER
from gliner.data_processing import DataCollator, GLiNERDataset, WordsSplitter
from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == "__main__":
    # 1. Load the pretrained GLiNER model.
    model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    tokenizer = AutoTokenizer.from_pretrained(model.config.model_name)
    words_splitter = WordsSplitter(model.config.words_splitter_type)


    # 2. Prepare a dataset.

    def process_token(token) -> list[str]:
        """Tokenizes the input text into a list of tokens."""
        if not token: return ['']
        words = re.findall(r'\w+(?:[-_]\w+)*|\S', token)
        return list(map(lambda w: w.replace("'", '"').lower(), words))


    def process_dataset(example):
        results = []
        for i in range(len(example["input"])):
            if example["input"][i] is None or example["output"][i] is None:
                results.append({"tokenized_text": ['N/A'], "ner": [(-1, -1)], "entities": ['N/A']})
                continue

            tokens = example["input"][i].split()
            tokens = [token for token_list in [process_token(token) for token in tokens] for token in token_list]

            entities = example["output"][i]
            entities_map_list = list(
                filter(lambda e: len(e) > 1, [entity.split("<>") for entity in ast.literal_eval(entities)]))
            if len(entities_map_list) <= 0:
                results.append({"tokenized_text": tokens, "ner": [(-1, -1)], "entities": ['N/A']})
                continue

            ner_span = []
            entities = []
            for entity_map in entities_map_list:
                s_index = None
                e_index = None
                target_sequence, entity_key = entity_map
                entity_key = entity_key.strip()
                target_sequence_tokens = process_token(target_sequence)

                for subset_index in range(len(tokens) - len(target_sequence_tokens) + 1):
                    sliced_tokens = tokens[subset_index:subset_index + len(target_sequence_tokens)]
                    if sliced_tokens != target_sequence_tokens:
                        continue
                    else:
                        s_index = subset_index
                        e_index = subset_index + (len(target_sequence_tokens) - 1)
                        break

                if s_index is None or e_index is None:
                    # print(tokens, target_sequence_tokens)
                    s_index = -1
                    e_index = -1
                    entity_key = 'N/A'

                entities.append(entity_key)
                ner_span.append((s_index, e_index))
            if len(entities) <= 0:
                ner_span.append((-1, -1))
                entities = ['N/A']
            results.append({"tokenized_text": tokens, "ner": ner_span, "entities": entities})
        return {"processed_items": results}


    ds = load_dataset("numind/NuNER")["entity"]
    # inputs, outputs = ds["input"], ds["output"]
    processed_ds = ds.map(process_dataset, batched=True).remove_columns(["input", "output"])


    def remap_output(item):
        return {"tokenized_text": item['tokenized_text'],
                "ner": [[*item['ner'][ner_idx], item['entities'][ner_idx]] for ner_idx in range(len(item['ner']))]}


    train_dataset = list(map(remap_output, processed_ds["processed_items"]))
    eval_count = int(len(train_dataset) * 0.30)
    eval_dataset = train_dataset[:eval_count]
    train_dataset = train_dataset[eval_count:]

    # 4. Create a data collator
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # 5. Set up training arguments.
    num_steps = 500
    batch_size = 8
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir="logs",
        learning_rate=1e-5,
        weight_decay=0.1,
        # others_lr=3e-5,
        # others_weight_decay=0.01,
        # focal_loss_gamma=config.loss_gamma,
        # focal_loss_alpha=config.loss_alpha,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        max_grad_norm=10.0,
        max_steps=100000,
        evaluation_strategy="epoch",
        save_steps=5000,
        save_total_limit=3,
        dataloader_num_workers=8,
        use_cpu=False,
        report_to="none",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
