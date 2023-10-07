import sys
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

patient_data = [
    {
        "description": "Runny nose, sneezing, and mild fever.",
        "disease_name": "Common cold"
    },
    {
        "description": "Sore throat, difficulty swallowing, and swollen tonsils.",
        "disease_name": "Tonsillitis"
    },
    {
        "description": "High body temperature, body aches, and fatigue.",
        "disease_name": "Influenza (Flu)"
    },
    {
        "description": "Persistent cough, shortness of breath, and chest pain.",
        "disease_name": "Pneumonia"
    },
    {
        "description": "Red, itchy rash and hives all over the body.",
        "disease_name": "Allergic reaction"
    },
    {
        "description": "Cough with greenish mucus, chest congestion, and fatigue.",
        "disease_name": "Bronchitis"
    },
    {
        "description": "Abdominal pain, diarrhea, and nausea.",
        "disease_name": "Gastroenteritis"
    },
    {
        "description": "Severe headache, sensitivity to light, and nausea.",
        "disease_name": "Migraine"
    },
    {
        "description": "Joint pain, swelling, and morning stiffness.",
        "disease_name": "Rheumatoid Arthritis"
    },
    {
        "description": "Frequent urination, burning sensation, and cloudy urine.",
        "disease_name": "Urinary Tract Infection (UTI)"
    },
    {
        "description": "Sudden chest pain, shortness of breath, and radiating pain in the left arm.",
        "disease_name": "Myocardial Infarction (Heart Attack)"
    },
    {
        "description": "Fever, chills, and a productive cough with yellow or green phlegm.",
        "disease_name": "Pneumonia"
    },
    {
        "description": "Severe abdominal pain, bloating, and vomiting.",
        "disease_name": "Appendicitis"
    },
    {
        "description": "Swollen and painful joints, morning stiffness, and fatigue.",
        "disease_name": "Lupus"
    },
    {
        "description": "Gradual memory loss, confusion, and difficulty with daily tasks.",
        "disease_name": "Alzheimer's Disease"
    }
]


model = LlamaForCausalLM.from_pretrained(
    "shibing624/ziya-llama-13b-medical-merged", load_in_8bit=True, device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained(
    "shibing624/ziya-llama-13b-medical-merged", load_in_8bit=True)
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


def task_prompt(instruction):
    return f"""## role
    Suggest you are a professional doctor, you need to determine what disease the patient has.
    ## task
    Below is a description of the patient's condition.
    Please response only one specific disease name, even if there are somes possible diseases. 
    Do not generate other words or more than one disease.
    
    ## Description: {instruction}
    ## Response: """


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


def task_response(input_sentence):
    q = task_prompt(input_sentence)
    inputs = tokenizer(q, return_tensors="pt").to(device=device)
    generate_ids = model.generate(**inputs, max_new_tokens=150)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    print(output.split("Response: ")[1].strip())
    return output.split("Response: ")[1].strip()


if __name__ == "__main__":
    while (1):
        s = input()
        if s == 'q':
            break
        format_response(s)
for patient in patient_data:
    a = task_response(patient["description"])
    print(patient["disease_name"])
