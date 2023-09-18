import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import os


def merge_model_lora_from_config(config_file: str, merged_model_folder: str = 'merged'):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    base_model = config["base_model"]
    lora_model = config["output_dir"]
    return merge_model_lora(base_model, lora_model, merged_model_folder)


def merge_model_lora(base_model_path: str, lora_model_path: str, merged_model_folder: str):

    merged_model_path = f"{lora_model_path}/{merged_model_folder}"

    if os.path.exists(merged_model_path):
        print(f"Model {merged_model_path} already exists, skipping")
        return merged_model_path

    print("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        return_dict=True,
        torch_dtype=torch.float16,
    )

    print("Loading PEFT adapter")
    model = PeftModel.from_pretrained(model, lora_model_path)
    print(f"Merging and unloading...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Model saved to {merged_model_path}")

    return merged_model_path
