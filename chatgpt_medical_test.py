
!pip install openai

# Подключение библиотек
import pandas as pd

# Чтение датафрейма из файла
data = pd.read_csv('/content/medical_transcripts_analysis.csv')

# Вывод названий всех столбцов
print(data.columns)

from google.colab import userdata
import os
import openai

client = openai.OpenAI(
    api_key=userdata.get('OPENAI_API_KEY')
)

from tqdm.notebook import tqdm

# Системный запрос
system_prompt = "I want you to act as a classifier of medical term and give an answer according to the description in one word. Answer example: Surgery"

# Создание нового столбца для ответов GPT-3.5, если он еще не существует
if 'gpt3_5_answer' not in data.columns:
    data['gpt3_5_answer'] = pd.Series(dtype='str')

# Обработка строк, начиная с 398-й, с использованием tqdm для отображения прогресса
for index in tqdm(data.iloc[398:].index, desc="Processing rows"):
    try:
        user_content = data.at[index, 'info'][:5000]  # Обрезка до 5000 символов
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        # Запись ответа в датафрейм
        data.at[index, 'gpt3_5_answer'] = response.choices[0].message.content.strip()
    except Exception as e:
        # В случае ошибки, пометить ответ как -1
        data.at[index, 'gpt3_5_answer'] = -1

    # Сохранение датафрейма после каждой обработанной строки
    data.to_csv('/content/medical_transcripts_analysis_updated.csv', index=False)

# Вывод сообщения об окончании обработки
print("Processing completed.")

# Системный запрос
system_prompt = "I want you to act as a classifier of medical term and give an answer according to the description in one word. Answer example: Surgery"

# Создание нового столбца для ответов GPT-4, если он еще не существует
if 'gpt4_answer' not in data.columns:
    data['gpt4_answer'] = pd.Series(dtype='str')

# Обработка всех строк, используя tqdm для отображения прогресса
for index in tqdm(data.index, desc="Processing rows"):
    try:
        user_content = data.at[index, 'info'][:1000]  # Обрезка до 5000 символов
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        # Запись ответа в датафрейм
        answ = response.choices[0].message.content.strip()
        print(answ)
        data.at[index, 'gpt4_answer'] = answ
    except Exception as e:
        # В случае ошибки, пометить ответ как -1
        data.at[index, 'gpt4_answer'] = -1

    # Сохранение датафрейма после каждой обработанной строки
    data.to_excel('/content/medical_transcripts_analysis_updated.xlsx', index=False)

# Вывод сообщения об окончании обработки
print("Processing completed.")

!pip install openpyxl

# Сохранение датафрейма в файл Excel
data.to_excel('/content/medical_transcripts_analysis_updated.xlsx', index=False)