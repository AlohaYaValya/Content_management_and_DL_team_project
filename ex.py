import pandas as pd
from medical_llama import *
ni = 0


def my_function(input_string):
    global ni
    print(ni)
    ni += 1
    if not isinstance(input_string, str):
        print("input_string is not a string")
        return None
    return transcripts_response(input_string)


# read Excel
df = pd.read_excel("medical_transcripts_analysis.xlsx", sheet_name='Sheet1')
df_selected = df.loc[0:999]
print(df_selected)
# read 'info' col as input and apply function
df_selected['special_llm_answer'] = df_selected['info'].apply(my_function)
print(df_selected)
# write output file
df_selected.to_excel("output.xlsx", index=False)
