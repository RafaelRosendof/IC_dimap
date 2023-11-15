'''
@brief Transcription deep learning model
@date: 2023-11-15
@author: Rafael Rosendo

'''
#import all the modules 

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import torch.nn
from datasets import Audio
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict
import copy


def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

'''
In summary, this data collator prepares the input and label batches for training a sequence-to-sequence model on speech data,
ensuring that the input audio features and tokenized labels are properly padded and formatted for training.'''
#Data collator class
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):    #Definition of the WER metric
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


import random

#load the datasets from the common voice
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="train+validation[:70%]")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="test[:50%]")

print (len(common_voice["train"])),

#Remove the unnecessary variables present in the columns of the dataset
columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]

# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")

# - Load Tokenizer: WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="pt", task="transcribe")


# STEP 3. Combine elements with WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="pt", task="transcribe")




# -> (1): Downsample from 48kHZ to 16kHZ

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Prepare and use function to prepare our data ready for the Whisper AI model
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=2 # num_proc > 1 will enable multiprocessing
    )


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# STEP 5.1. Define evaluation metric

metric = evaluate.load("wer")

# STEP 5.3. Load a pre-trained Checkpoint

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# STEP 5.4. Define the training configuration




training_args = Seq2SeqTrainingArguments(
    output_dir="/home/rafaelrosendo/IC_dimap/my_models",  # Output directory for the model
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=500,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=200,                ###############################Define the logging parameters
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # testing
)


# Initialize a trainer.


trainer_1 = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: lt, split: test",
    "language": "lt",
    "model_name": "Whisper Large LT - Rafael_Rosendo",  #my name
    "finetuned_from": "openai/whisper-large",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

from huggingface_hub import HfApi, HfFolder

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
api = HfApi()
folder = HfFolder()

# Set your Hugging Face Hub token
token = 'the secret token you can put here '
folder.save_token(token)

# Push your model to the Hugging Face Hub
#create_repo("Rafaelrosendo1/whisper-rafael-pt",private=False)
#pt_model.push_to_hub(model_id="Rafaelrosendo1/whisper-rafael-pt", path="/home/rafaelrosendo/IC_dimap/my_models")


trainer_1.push_to_hub("Rafaelrosendo1/whisper-rafael-pt")
#print('Trained model uploaded to the Hugging Face Hub')

