import ast
import re
from random import random
from typing import Optional

from gliner import GLiNER, GLiNERConfig
from gliner.data_processing import DataCollator, GLiNERDataset, WordsSplitter, TokenProcessor, SpanProcessor
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoTokenizer
from datasets import load_dataset, tqdm, Dataset


class CustomGLiNERDataset(Dataset):
    def __init__(self, examples,
                 config: Optional[GLiNERConfig],
                 tokenizer: Optional[AutoTokenizer] = None,
                 words_splitter: Optional[WordsSplitter] = None,
                 data_processor=None,
                 entities=None,
                 get_negatives: bool = True):
        self._data = examples
        self.config = config
        if data_processor is not None:
            self.data_processor = data_processor
        else:
            if config.span_mode == "token_level":
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter, preprocess_text=True)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter, preprocess_text=True)

        self.max_neg_type_ratio = int(self.config.max_neg_type_ratio)
        self.get_negatives = get_negatives
        if not entities:
            self.all_entities = self._collect_all_entities()
        else:
            self.all_entities = entities
        self.max_negatives = min(50, len(self.all_entities))

    def _get_entities_from_example(self, example):
        entities = {ner["key"] for ner in example['ner']}
        return entities

    def _collect_all_entities(self):
        print("Collecting all entities...")
        all_entities = set()
        for example in tqdm(self._data):
            curr_entities = self._get_entities_from_example(example)
            all_entities.update(curr_entities)
        print('Total number of entity classes: ', len(all_entities))
        return list(all_entities)

    def _get_negatives(self):
        negatives = random.sample(self.all_entities, k=self.max_negatives)
        random.shuffle(negatives)
        return negatives

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        try:
            example = self._data[idx]
            if self.get_negatives:
                curr_negatives = self._get_negatives()
            else:
                curr_negatives = None

            raw_batch = self.data_processor.collate_raw_batch([example], negatives=curr_negatives)

            model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=True)
            if 'span_idx' in raw_batch:
                model_input['span_idx'] = raw_batch['span_idx']
            if 'span_mask' in raw_batch:
                model_input['span_mask'] = raw_batch['span_mask']
            if 'seq_length' in raw_batch:
                model_input['text_lengths'] = raw_batch['seq_length']
            return model_input
        except Exception as e:
            print(f"Skipping getting item due to error: {e}")
            return None


if __name__ == "__main__":
    # 1. Load the pretrained GLiNER model.
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
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
                results.append({"tokenized_text": 'N/A', "ner": [{"s": -1, "e": -1, "key": 'N/A'}]})
                continue

            tokens = example["input"][i].split()
            entities = example["output"][i]
            entities_map_list = list(
                filter(lambda e: len(e) > 1, [entity.split("<>") for entity in ast.literal_eval(entities)]))
            if len(entities_map_list) <= 0:
                results.append({"tokenized_text": example["input"][i], "ner": [{"s": -1, "e": -1, "key": 'N/A'}]})
                continue

            tokens = [token for token_list in [process_token(token) for token in tokens] for token in token_list]
            ner_span = []
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

                ner_span.append({"s": s_index, "e": e_index, "key": entity_key})
            results.append({"tokenized_text": example["input"][i], "ner": ner_span})
        return {"processed_items": results}


    ds = load_dataset("numind/NuNER")["entity"]
    # inputs, outputs = ds["input"], ds["output"]
    processed_ds = ds.map(process_dataset, batched=True).remove_columns(["input", "output"])

    train_dataset = CustomGLiNERDataset(processed_ds['processed_items'], config=model.config, tokenizer=tokenizer,
                                        words_splitter=words_splitter)

    # 4. Create a data collator
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # 5. Set up training arguments.
    num_steps = 500
    batch_size = 8
    data_size = len(ds["full"])
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        output_dir="logs",
        learning_rate=1e-5,
        weight_decay=0.1,
        others_lr=3e-5,
        others_weight_decay=0.01,
        # focal_loss_gamma=config.loss_gamma,
        # focal_loss_alpha=config.loss_alpha,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
