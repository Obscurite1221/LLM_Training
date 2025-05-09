import pandas
import torch
import transformers
import datasets
from transformers import TrainingArguments
import peft


### Load dataset and drop all the irrelevant data columns.
### Uses Pandas dataframe and relevant library functions to do so.
Dataset = datasets.Dataset

dataset = pandas.read_csv("./Biomarker_dataset_2025_3_12.csv")
dataset.drop("RowId", axis="columns")
dataset.drop("NCTId", axis="columns")
dataset.drop("Current", axis="columns")
dataset.drop("Link", axis="columns")
dataset.drop("Field", axis="columns")

# Drop all the empty or unfilled rows in the data
# and split the data into the features and labels.
dataset = dataset.dropna(subset=['Annotation'])
dataset_Y = dataset.iloc[:,5]
dataset_Y = dataset_Y.reset_index(drop=True)
dataset_X = dataset.iloc[:,-2:]


### Format the prompts according to what I determined was the most optimal training format for Llama 3.1 Instruct.
### This can still likely be improved, and I highly doubt it's even fully correct as-is. The highly adaptive nature
### of LLMs means that it's incredibly difficult to determine the true optimal format.
temp_list = []
for x in range(dataset_X.shape[0]):
    # Note that some of these symbols are specified as special tokens in the associated tokenizer, however,
    # some of the symbols are literally interpreted by the model in plaintext and are still part of the formatting.
    # An example of this is <|begin_of_text|> vs [INST]. [INST] is a plaintext format called 'Alpaca', and it has no
    # special entry in the token table. However, <|begin_of_text|> is listed as one of the initial tokens, and is a special token.
    temp_list.append(f"<|begin_of_text|>[INST] <<SYS>>\n" +
                     "You are a helpful assistant specializing in medical text annotation. Your task is to analyze the " +
                     "provided outcome measure and outcome description from clinical trials and predict the most medically " +
                     "significant biomarkers.\n" +

                     "1) Output a comma-separated list of predicted biomarkers, using the format SUPERGROUP_name " +
                     "(e.g., CSF_p-tau, BLOOD_Aβ40, FDG-PET). Do not include any additional commentary or explanation.\n" +

                     "2) The biomarker list should follow the general format: SUPERGROUP_name. Example supergroups include " +
                     "CSF, BLOOD, UD (undetermined), URINE, FDG-PET, fMRI, etc. This list is not exhaustive, and not all " +
                     "terms must match the examples exactly.\n" +

                     "3) If no specific biomarkers are clearly indicated, respond with a single word that best represents " +
                     "the relevant biological domain (e.g., 'Metabolomics', 'Inflammation', 'Imaging').\n" +

                     "4) If the outcome measure and description do not relate to any known biological processes or " +
                     "biomarkers, respond with 'Unknown'.\n<</SYS>>" +

                     f"\nOutcome Measure:\n{dataset_X['Outcome Measure']}\nOutcome Description:\n{dataset_X['Outcome Description']}\n[/INST]\n")


### Format the label data. This is what the model is being trained to see as the 'correct' response.
temp_list_2 = []
for x in range(dataset_Y.shape[0]):
    ### This is a choice of whether to tell the model to reprint the original prompt AND the answer, or just the answer.
    ### There are models trained using both methods.
    #temp_list_2.append(temp_list[x] + dataset_Y[x] + '<|end_of_text|>')
    temp_list_2.append(dataset_Y[x] + '<|end_of_text|>')

### Convert the two lists into full dataframes using an inbuilt Pandas conversion function.
### X is the prompts, Y is the labels.
dataset_X = pandas.DataFrame(temp_list)
dataset_Y = pandas.DataFrame(temp_list_2)

### Load the tokenizer and double confirm that the correct padding token is loaded into the tokenizer config.
tokenizer = transformers.AutoTokenizer.from_pretrained('./Model')
tokenizer.pad_token = tokenizer.eos_token

### Run the tokenizer on each prompt and save the tokenized datasets into new variables.
### These now are formatted as ['input_id', 'attention_mask'] tensors.
### The extra parameters assist in back-end prompt formatting, essentially zero-padding or truncating the prompt to match the max length.
tokenized_dataset_X = dataset_X.map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=512))
tokenized_dataset_Y = dataset_Y.map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=512))


### This is a hack to format the data in a more friendly format.
### It groups the prompt, attention mask, and label data into an easier format.
TVal1 = []
TVal2 = []
TVal3 = []
for x in range(tokenized_dataset_X[0].shape[0]):
    TVal1.append(tokenized_dataset_X[0][x]['input_ids'])
    TVal2.append(tokenized_dataset_X[0][x]['attention_mask'])
    TVal3.append(tokenized_dataset_Y[0][x]['input_ids'])

### This iterates over the label values and sets the value for the <|end_of_text|> token to -100, which means
### it will be ignored for cross-entropy loss evaluation.
TVal3 = [
    [token if token != 128001 else -100 for token in seq]
    for seq in TVal3
]

### Merge all three lists together for easy access, once again.
merged_dataset = {
    'input_ids': TVal1,
    'attention_mask': TVal2,
    'labels': TVal3
}

### Run a conversion function using Huggingfaces Dataset library to convert from dict to Dataset type.
tokenized_dataset_adjusted = Dataset.from_dict(merged_dataset)

### Split the dataset into the train/test data.
split_dataset = tokenized_dataset_adjusted.train_test_split(test_size=0.2, seed=3000)

### Label the variables appropriately and split it into a train and validation set.
X_train = split_dataset['train']
X_test = split_dataset['test']


### Define the Peft config settings for finetuning.
config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM, # Model Type
    inference_mode=False, # Whether it will be tested or just trained
    r=64, # The number of ranks to train
    lora_alpha=32, # The weight of the Peft layer
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # The parameters to train on each node
    lora_dropout=0.1, # Same as usual ML dropout, prevents overfitting
    bias="none", # Used for more advanced tuning and types of training.
    modules_to_save=["lm_head"], # The part that actually holds the changes.
)

### Load the actual model from file. Since this is using a Llama 3.1 Instruct, it is considered AutoModelForCausalLM,
### NOT LlamaForCasualLM! It may be possible to use LlamaForCasualLM with Llama 3.1, but I have not experimented with it.
### As such, this code may not be compatible if this is changed.
model = transformers.AutoModelForCausalLM.from_pretrained('./Model',local_files_only=True)

### Wrap the transformer model with the Peft adapter
model = peft.get_peft_model(model, config)

### Show that Peft is loaded and display what percentage of the model is being fine-tuned.
model.print_trainable_parameters()

### Data logging and analytics software. Also apparently helps with formatting issues, but I'm not 100% certain on that.
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

### All the important parameters for training the model.
training_args = TrainingArguments(
    output_dir="./Model_Output", # Save directory
    per_device_train_batch_size = 2, # Per GPU batch, must match with Deepspeed config
    per_device_eval_batch_size = 2, # Per GPU batch, must match with Deepspeed config
    #gradient_checkpointing=True, # Useful for reducing the GPU load, but it's not fully configured right now.
    gradient_accumulation_steps=2, # See above line
    weight_decay=0.00001,  # The normalization to prevent exploding gradients. Keep it close to or smaller than the learning rate or things won't work well.
    learning_rate = 0.00005, # The learning rate. Higher is faster, lower is better.
    num_train_epochs = 50, # Number of passes over the dataset.
    eval_strategy="steps", # Format for logging the training progress. It can be per-n-steps, or per-epoch.
    eval_steps=100, # How often to evaluate performance.
    save_strategy="steps", # Format for saving the model.
    save_steps=100, # How often to save the model.
    save_total_limit=3, # How many versions back you want to save.
    load_best_model_at_end=True, # Save the best model recorded at the end. Can be buggy sometimes, if it tries to load a model that was already deleted.
    logging_strategy="steps", # additional data logging every n steps.
    logging_steps=100, # Interval of data logging. Not very performance degrading.
    logging_dir='./logs', # Self explanatory.
    deepspeed="./ds_config.json", # The location of the deepspeed config file. The params in the deepspeed MUST MATCH these params, or it WILL NOT RUN.
    #remove_unused_columns=False,
    do_train=True, # Tell it that it's going to train the model, it's not just loading it.
    do_eval=True, # Tell it that it's going to test the model as well using a validation dataset.
    lr_scheduler_type="cosine", # Very important to set correctly. Cosine approaches zero over time. There are multiple other types of schedulers.
    warmup_steps=130, # Starts the fine-tuning off slow to prevent instability, ramps up over 130 steps, then begins lowering it as training progresses.
    fp16=True # Use mixed precision to save on memory.
)

# lr_scheduler_type="cosine_with_restarts"

### Note, as this is a proof of concept, it has no actual train/test/validate split. It only uses a train/test split, which may not be indicative of true performance.

### Start training, save the model, then clear up all the parallel threats.
trainer = transformers.trainer.Trainer(model, args=training_args, train_dataset=X_train, eval_dataset=X_test, data_collator=data_collator)
#trainer.train(resume_from_checkpoint="./Model_Output/checkpoint-2580")
trainer.train()
trainer.save_model("./Model_Output/Final")
tokenizer.save_pretrained("./Model_Output/Final")
torch.distributed.destroy_process_group()
