# 导入pandas库
import pandas as pd
from medical_llama import *
ni = 0


def my_function(input_string):
    global ni
    print(ni)
    ni += 1
    # 这里是一个简单的示例，将输入字符串反转
    if not isinstance(input_string, str):
        print("input_string is not a string")
        return None
    return transcripts_response(input_string)


# 读取Excel文件，假设文件名为data.xlsx
df = pd.read_excel("medical_transcripts_analysis.xlsx", sheet_name='Sheet1')
df_selected = df.loc[0:999]
print(df_selected)
# 读取每行的'info'列作为输入，调用函数得到结果
df_selected['special_llm_answer'] = df_selected['info'].apply(my_function)
print(df_selected)
# 将结果写入Excel文件，假设文件名为output.xlsx
df_selected.to_excel("output.xlsx", index=False)
