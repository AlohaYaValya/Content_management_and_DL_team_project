import sys
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

model = LlamaForCausalLM.from_pretrained(
    "shibing624/ziya-llama-13b-medical-merged", load_in_4bit=True, device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained(
    "shibing624/ziya-llama-13b-medical-merged", load_in_4bit=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. 
    Write a response that appropriately completes the request.
    \n\n### Instruction:{instruction}\n\n### Response: """


def format_prompt(instruction, f=0):
    s = ['Disease', 'Symptom']
    return f"""Below is a disease or a symptom related to a disease. 
    Suggest you are the best doctor.
    Please provide detailed information about the disease
      as a response in the following format: 
    'Disease Name: [Disease Name] \nCauses: [Causes] \nSymptoms: [Symptoms] \nTreatment: [Treatment].\n' 
    Please provide the relevant information.
    \n\n### {s[f]}:{instruction}\n\n### Response: """


def response(input_sentence):
    q = generate_prompt(input_sentence)
    inputs = tokenizer(q, return_tensors="pt").to(device=device)
    generate_ids = model.generate(**inputs, max_new_tokens=500)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    print(output)
    print()
    return output


def format_response(input_sentence, f=0):
    q = format_prompt(input_sentence, f)
    inputs = tokenizer(q, return_tensors="pt").to(device=device)
    generate_ids = model.generate(**inputs, max_new_tokens=500)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    print(output)
    print()
    return output


if __name__ == "__main__":
    while (1):
        s = input()
        if s == 'q':
            break
        format_response(s)
