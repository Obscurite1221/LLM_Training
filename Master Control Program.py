import pandas
import os
import torch
import sklearn
import transformers
from pandas import DataFrame
import datasets
from transformers import TrainingArguments
import peft
# Sentencepiece, protobuf
max_seq_length = 8192
dtype = None
load_in_4bit = True

Dataset = datasets.Dataset

dataset = pandas.read_csv("./Biomarker_dataset_2025_3_12.csv")
#dataset.drop(axis="index")
dataset.drop("RowId", axis="columns")
dataset.drop("NCTId", axis="columns")
dataset.drop("Current", axis="columns")
dataset.drop("Link", axis="columns")
dataset.drop("Field", axis="columns")

dataset = dataset.dropna(subset=['Annotation'])
dataset_Y = dataset.iloc[:,5]
dataset_Y = dataset_Y.reset_index(drop=True)
dataset_X = dataset.iloc[:,-2:]

temp_list = []
for x in range(dataset_X.shape[0]):
    temp_list.append(f"<|begin_of_text|>[INST] You are a helpful assistant specializing in medical text annotation. Your task is to analyze outcome measures and descriptions from clinical trials and predict the most medically significant biomarkers. Please list biomarkers in a comma-separated format and do not include any additional information except for the biomarkers.\n" +
                     "The biomarkers should be labelled in a format as follows - SUPERGROUP_name. For example: CSF_p-tau, BLOOD_Aβ40, BLOOD_p-tau, UD_HbA1c, UD_IGF1, UD_T3, UD_IL-6, UD_TNF-α, URINE_F2-isoP, FDG-PET, fMRI, BLOOD_proteome, CSF_proteome. This list is not exhaustive, nor do all possibilities follow the exact convention listed. [/INST]\n" +
                     f"Outcome Measure:\n{dataset_X['Outcome Measure']}\nOutcome Description:\n{dataset_X['Outcome Description']}\n")

#print(dataset_Y.index)

temp_list_2 = []
for x in range(dataset_Y.shape[0]):
    temp_list_2.append(temp_list[x] + dataset_Y[x] + '<|end_of_text|>')

dataset_X = pandas.DataFrame(temp_list)
dataset_Y = pandas.DataFrame(temp_list_2)
df_concat = pandas.concat([dataset_X, dataset_Y], axis=1)

#print(dataset_X)
tokenizer = transformers.AutoTokenizer.from_pretrained('./Model')
tokenizer.pad_token = tokenizer.eos_token

#def tokenize_function_input(text):
#    prompt = f"### Instruction: Please annotate the following clinical trial data using this example format:  CSF_p-tau, CSF_t-tau, BLOOD_Aβ40, BLOOD_Aβ42, BLOOD_p-tau\n### Input: {dataset_X['Outcome Measure']}\n{dataset_X['Outcome Description']}"
#    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

#def tokenize_function_output(text):
#    prompt = dataset_Y['Annotation']

#tokenized_dataset = df_concat.apply(tokenize_function, axis=1)
#print(dataset_X)

tokenized_dataset_X = dataset_X.map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=512))
tokenized_dataset_Y = dataset_Y.map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=512))

TVal1 = []
TVal2 = []
TVal3 = []
for x in range(tokenized_dataset_X[0].shape[0]):
    TVal1.append(tokenized_dataset_X[0][x]['input_ids'])
    TVal2.append(tokenized_dataset_X[0][x]['attention_mask'])
    TVal3.append(tokenized_dataset_Y[0][x]['input_ids'])

#TVal1 = tokenized_dataset_X[0][0].ids
#TVal2 = tokenized_dataset_X[0][0]['attention_mask']
#TVal3 = tokenized_dataset_Y['Annotation'][:]['input_ids']

#print(tokenized_dataset_Y)
#print(dataset_Y)

#tokenized_dataset_X_unpacked = [
#    {"input_ids": entry["input_ids"], "attention_mask": entry["attention_mask"]}
#    for entry in tokenized_dataset_X
#]

#tokenized_dataset_Y_unpacked = [
#    {"input_ids": entry["input_ids"], "attention_mask": entry["attention_mask"]}
#    for entry in tokenized_dataset_Y
#]

#tokenized_dataset_X_unpacked = datasets.Dataset.from_pandas(DataFrame(tokenized_dataset_X_unpacked))
#tokenized_dataset_Y_unpacked = datasets.Dataset.from_pandas(DataFrame(tokenized_dataset_Y_unpacked))


#labels = [
#    {"labels": entry_Y["input_ids"] + [-100] * (512 - len(entry_Y["input_ids"]))}
#    for  entry_Y in tokenized_dataset_Y_unpacked
#]

#tokenized_dataset_merged = {
#    "input_ids": [entry_X["input_ids"] for entry_X in tokenized_dataset_X_unpacked],
#    "attention_mask": [entry_X["attention_mask"] for entry_X in tokenized_dataset_X_unpacked],
#    "labels": [label["labels"] for label in labels]
#}

TVal3 = [
    [token if token != 128001 else -100 for token in seq]
    for seq in TVal3
]

merged_dataset = {
    'input_ids': TVal1,
    'attention_mask': TVal2,
    'labels': TVal3
}

#print(TVal3)
#print(tokenizer.pad_token_id)
#print(merged_dataset)
#print(merged_dataset)
#for x in range(len(tokenized_dataset_merged)):
#    print(tokenized_dataset_merged["labels"])

#tokenized_dataset_adjusted = []
#for x in range(len(tokenized_dataset_merged["input_ids"])):
#    tokenized_dataset_adjusted.append({
#        "input_ids": torch.tensor(tokenized_dataset_merged["input_ids"][x], dtype=torch.long),
#        "attention_mask": torch.tensor(tokenized_dataset_merged["attention_mask"][x], dtype=torch.long),
#        "labels": torch.tensor(tokenized_dataset_merged["labels"][x], dtype=torch.long),
#    })



#for x in range(len(tokenized_dataset_adjusted)):
#    print(len(tokenized_dataset_adjusted[x]["input_ids"]))
#print(len(tokenized_dataset_adjusted[0]["input_ids"]))  # Should match len(labels)
#print(len(tokenized_dataset_adjusted[0]["labels"]))

#print(tokenized_dataset_merged)
#print(len(tokenized_dataset_adjusted[0]))
#print(len(tokenized_dataset_merged["input_ids"]), type(tokenized_dataset_merged["input_ids"]), len(tokenized_dataset_merged["attention_mask"]), type(tokenized_dataset_merged["attention_mask"]), len(tokenized_dataset_merged["labels"]), type(tokenized_dataset_merged["labels"]))
#print(len(tokenized_dataset_merged["input_ids"][0]))
tokenized_dataset_adjusted = Dataset.from_dict(merged_dataset)

split_dataset = tokenized_dataset_adjusted.train_test_split(test_size=0.2, seed=3000)
#train_loader = torch.utils.data.DataLoader(split_dataset['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
#test_loader = torch.utils.data.DataLoader(split_dataset['test'], batch_size=16, collate_fn=data_collator)

X_train = split_dataset['train']
X_test = split_dataset['test']
#X_train, X_test = sklearn.model_selection.train_test_split(tokenized_dataset_merged, train_size=0.80, test_size=0.20, random_state = 3000)

#X_train = datasets.Dataset.from_pandas(DataFrame(X_train))
#Y_train = datasets.Dataset.from_pandas(DataFrame(Y_train))
#input_length_train = len(X_train["input_ids"])
#labels_train = [-100] * input_length_train + Y_train["input_ids"]

#X_test = datasets.Dataset.from_pandas(DataFrame(X_test))
#Y_test = datasets.Dataset.from_pandas(DataFrame(Y_test))
#input_length_test = len(X_test["input_ids"])
#labels_test = [-100] * input_length_test + Y_test["input_ids"]



#print(X_train)
#torch.distributed.init_process_group(backend="nccl", rank=int(os.environ["SLURM_PROCID"]), world_size=7, init_method="env://")

#model = torch.nn.DataParallel(model)
#model = model.cuda()

#print(model.forward)

#model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["SLURM_PROCID"])])
#model, _, _, _ = deepspeed.initialize(model=model, config="./ds_config.json")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["lm_head"],
)

model = transformers.AutoModelForCausalLM.from_pretrained('./Model',local_files_only=True)

#for name, param in model.named_parameters():
#    print(name, param.size())
#for name, module in model.named_modules():
#    print(name)

model = peft.get_peft_model(model, config)

model.print_trainable_parameters()

data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./Model_Output",
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    weight_decay=0.0001,
    learning_rate = 0.0003,
    num_train_epochs = 10,
    eval_strategy="steps",
    eval_steps=15,
    save_strategy="steps",
    save_steps=60,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',
    deepspeed="./ds_config.json",
    #remove_unused_columns=False,
    do_train=True,
    do_eval=True,
    lr_scheduler_type="cosine",
    warmup_steps=15
)

# lr_scheduler_type="cosine_with_restarts"

trainer = transformers.trainer.Trainer(model, args=training_args, train_dataset=X_train, eval_dataset=X_test, data_collator=data_collator)
trainer.train()
trainer.save_model("./Model_Output/Final")
tokenizer.save_pretrained("./Model_Output/Final")
torch.distributed.destroy_process_group()
