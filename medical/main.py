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


def response(input_sentence):
    q = generate_prompt(input_sentence)
    inputs = tokenizer(q, return_tensors="pt").to(device=device)
    generate_ids = model.generate(**inputs, max_new_tokens=500)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    print(output)
    print()


if __name__ == "__main__":
    sents = ['一岁宝宝发烧能吃啥药', "who are you?"]
    for s in sents:
        q = generate_prompt(s)
        inputs = tokenizer(q, return_tensors="pt").to(device=device)
        generate_ids = model.generate(**inputs, max_new_tokens=200,
                                      # do_sample=True,
                                      # top_p=0.85,
                                      # temperature=1.0,
                                      # repetition_penalty=1.0,
                                      # eos_token_id=tokenizer.eos_token_id,
                                      # bos_token_id=tokenizer.bos_token_id,
                                      # pad_token_id=tokenizer.pad_token_id,
                                      )
        output = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True)[0]
        print(output)
        print()
