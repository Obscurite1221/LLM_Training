import transformers
import peft

model = transformers.AutoModelForCausalLM.from_pretrained("./Model")
lora_config = peft.LoraConfig.from_pretrained("./Model_Output/Final/")
tokenizer = transformers.AutoTokenizer.from_pretrained("./Model_Output/Final/")
model = peft.get_peft_model(model, lora_config)
model = model.merge_and_unload()

model.push_to_hub("Obscurite1222/Llamed3.1")
tokenizer.push_to_hub("Obscurite1222/Llamed3.1")