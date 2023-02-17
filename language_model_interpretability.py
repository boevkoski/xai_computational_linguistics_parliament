import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import random
import shap
import transformers
import torch
import scipy as sp


training_args = TrainingArguments(output_dir="trainer", evaluation_strategy='epoch', per_device_train_batch_size=2,
                                  num_train_epochs=30, seed=seed,
                                  logging_steps=128, save_steps=2000, learning_rate=6e-6) # define hyper-parameters for the language model
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta") # language model tokenizer from huggingface
model = AutoModelForSequenceClassification.from_pretrained("EMBEDDIA/sloberta", num_labels=2) # language model from huggingface
metric = evaluate.load("accuracy") # metric


def compute_metrics(eval_pred): # function for taking the logits and transforming in binary predictions
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) # tokenize text for language model


df = pd.read_csv('../data/speeches.csv') # read speeches into Pandas

df_hug = pd.DataFrame()
df_hug['label'] = df['Political orientation']
df_hug['text'] = df['Speech text']
dataset = Dataset.from_pandas(df_hug) # create Huggingface Dataset for easier training
orientation = ClassLabel(num_classes=2, names=["left", "right"]) # define prediction classes
dataset = dataset.cast_column("label", orientation)

tokenized_datasets = dataset.map(tokenize_function, batched=True) # tokenize dataset (batched)

tokenized_datasets = tokenized_datasets.train_test_split(shuffle=True, seed=seed+1,
                                                         test_size=0.1, stratify_by_column='label') # train/test split
train_tokenized_dataset = tokenized_datasets['train']
eval_tokenized_dataset = tokenized_datasets['test']

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=eval_tokenized_dataset,
    compute_metrics=compute_metrics,
) # initialize model

trainer.train() # train model
model.save_pretrained('./sloberta_model_slow/') # save model


### INTERPRETATION - SHAPLEY STYLE (SHAP LIBRARY)

model = AutoModelForSequenceClassification.from_pretrained('./sloberta_model_30epochs/2').cuda()
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")

def f(x): # calculate shapley values
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

explainer = shap.Explainer(f, tokenizer) # calculate shapleys - this takes a while...

shap.plots.bar(shap_values.min(0), max_display=20) # plot 20 tokens with highest influence for predicting class=-1 (leftist) - per example

shap.plots.bar(shap_values.max(0), max_display=20) # plot 20 tokens with highest influence for predicting class=1 (rightist) - per example

shap.plots.bar(shap_values.abs.sum(0), max_display=30) # plot 30 tokens with highest total influence in the model (for both classes, for all examples)



