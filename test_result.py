import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import transformers
import peft

model = transformers.AutoModelForCausalLM.from_pretrained("./Model")
lora_config = peft.LoraConfig.from_pretrained("./Model_Output/Final/")
tokenizer = transformers.AutoTokenizer.from_pretrained("./Model_Output/Final/")
model = peft.get_peft_model(model, lora_config)
model = model.merge_and_unload()
#model = model.to(torch.float16)
print(hasattr(model, "peft_config"))
model.save_pretrained("Llamed_3.1", safe_serialization=True)
#tokenizer.save_pretrained("Llamed_3.1")

num_gpus = torch.cuda.device_count()
device = torch.device("cuda")
model = model.to(device)

model.eval()

Format = "### Instruction: Rephrase the following clinical trial data using this example format:  CSF_p-tau, CSF_t-tau, BLOOD_Aβ40, BLOOD_Aβ42, BLOOD_p-tau\n"

# Test 1
#	URINE_8-OHdG/8-OHG	Urine oxidative stress biomarkers	Changes from baseline in: 8-OHdG/8-OHG in urine
OM1 = "Urine oxidative stress biomarkers\n"
OM2 = "Changes from baseline in: 8-OHdG/8-OHG in urine"
OM1_2 = "Brain metabolism"
OM2_2 = "FDG-PET is a highly sensitive means of determining brain metabolism and has been accepted as a good proxy measure of synaptic function. Importantly, FDG-PET based measures of brain metabolism correlate well with cognitive decline in AD, better than amyloid plaques. Data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) study suggest that FDG PET has good power to detect a 25% treatment effect over 12 months"
OM1_3 = "Exploratory Metabolomic Analysis"
OM2_3 = "Change in metabolomics based on a global metabolomics exploratory panel from Baseline to Week 26."

def format(val1, val2):
    #prompt1 = ( f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" +
    #            "You are a helpful assistant specializing in medical text annotation. Your task is to analyze the " +
    #            "provided outcome measure and outcome description from clinical trials and predict the most medically " +
    #            "significant biomarkers.\n" +

    #            "1) Output a comma-separated list of predicted biomarkers, using the format SUPERGROUP_name " +
    #            "(e.g., CSF_p-tau, BLOOD_Aβ40, FDG-PET). Do not include any additional commentary or explanation.\n" +

    #            "2) The biomarker list should follow the general format: SUPERGROUP_name. Example supergroups include " +
    #            "CSF, BLOOD, UD (undetermined), URINE, FDG-PET, fMRI, etc. This list is not exhaustive, and not all " +
    #            "terms must match the examples exactly.\n" +

    #            "3) If no specific biomarkers are clearly indicated, respond with a single word that best represents " +
    #            "the relevant biological domain (e.g., 'Metabolomics', 'Inflammation', 'Imaging').\n" +

    #            "4) If the outcome measure and description do not relate to any known biological processes or " +
    #            "biomarkers, respond with 'Unknown'.\n" +

    #            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\nOutcome Measure:\n{val1}\nOutcome Description:\n{val2}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
    prompt1 = (f"<|begin_of_text|>[INST] <<SYS>>\n" +
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

                     f"\nOutcome Measure:\n{val1}\nOutcome Description:\n{val2}\n[/INST]\nBiomarkers: ")

    return prompt1

prompt_A = format(OM1, OM2)
prompt_B = format(OM1_2, OM2_2)
prompt_C = format(OM1_3, OM2_3)

#print(Format + OM1 + OM2)
#print(Format + OM1_2 + OM2_2)
#print(Format + OM1_3 + OM2_3)

prompts = [prompt_B, prompt_A, prompt_C]
tokenizer.padding_side = "left"
inputs = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        eos_token_id=tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        pad_token_id=tokenizer.pad_token_id,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1,
        repetition_penalty=1.2
    )

generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, response in enumerate(generated_text):
    print(f"Prompt {i + 1}:\n{prompts[i]}")
    print(f"Generated Response {i + 1}:\n{response}\n\n")

#	FDG-PET	Brain metabolism	FDG-PET is a highly sensitive means of determining brain metabolism and has been accepted as a good proxy measure of synaptic function. Importantly, FDG-PET based measures of brain metabolism correlate well with cognitive decline in AD, better than amyloid plaques. Data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) study suggest that FDG PET has good power to detect a 25% treatment effect over 12 months
#   UD_Metabolome	Exploratory Metabolomic Analysis	• Change in metabolomics based on a global metabolomics exploratory panel from Baseline to Week 26.