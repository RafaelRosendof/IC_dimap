import transformers
import torch.distributed as dist
import accelerate
import jiwer
import tensorboard
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from huggingface_hub import HfApi, HfFolder
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch



def main():
    print("COMEÇANDO O FINE TUNE DO WHISPER")

    print("GPU é ")
    print(torch.cuda.is_available())
    api = HfApi()
    folder = HfFolder()

    token = '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'
    folder.save_token(token)

    common_voice = DatasetDict()
    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="test", use_auth_token=True)

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Portuguese", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Portuguese", task="transcribe")

    def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

    # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch


    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice["train"].column_names, num_proc=2)



    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.generation_config.language = "portuguese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

    # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    
    


    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-pt",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=200,
        max_steps=2000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        remove_unused_columns=False,
        generation_max_length=500,
        save_steps=700,
        eval_steps=700,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)
    trainer.train()

    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_11_0",
        "dataset": "Common Voice 11.0",
        "dataset_args": "config: pt, split: test",
        "language": "pt",
        "model_name": "Whisper Small PT - Fine-tuned",  
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
    }

    trainer.push_to_hub("Rafaelrosendo1/whisper_small_teste")
if __name__ == "__main__":
    main()

