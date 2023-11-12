# Instalação das bibliotecas necessárias


# Importação das bibliotecas
from transformers import AutoModel
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import numpy as np

model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

from datasets import load_dataset, load_metric, Audio
import re

# Carregando os dados do conjunto de dados Common Voice em português
common_voice_train = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="train+validation[:50%]")
common_voice_test = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="test[:50%]")

import random

# Removendo colunas desnecessárias dos conjuntos de dados
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# Removendo caracteres especiais do texto
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

# Função para extrair todos os caracteres do conjunto de dados
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict

# Removendo caracteres especiais do vocabulário
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Configuração do tokenizador
from transformers import Wav2Vec2CTCTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# Configuração do extrator de características
from transformers import Wav2Vec2FeatureExtractor

# Configuração do processador
from transformers import Wav2Vec2Processor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Conversão dos áudios para o formato correto
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))


# Função para preparar o conjunto de dados
def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=4)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, num_proc=4)

# Configuração do coletor de dados
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Configuração das métricas
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Carregamento do modelo Wav2Vec2ForCTC
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()
model.gradient_checkpointing_enable()

# Configuração dos argumentos de treinamento
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="/home/rafaelrosendo/IC_dimap/wav2vec2-large-xlsr-pt-demo",
    group_by_length=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    save_steps=100,
    eval_steps=5,
    logging_steps=2,
    learning_rate=3e-4,
    warmup_steps=50,
    save_total_limit=100,
)

# Configuração do treinador
from transformers import Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

# Treinamento do modelo

trainer.train()


from huggingface_hub import HfApi, HfFolder

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
api = HfApi()
folder = HfFolder()

# Set your Hugging Face Hub token
token = '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'
folder.save_token(token)

trainer.push_to_hub("Rafaelrosendo1/wave2vec-rafael-pt")