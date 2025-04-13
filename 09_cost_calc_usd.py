from _stat_gen import *
import pandas as pd

# AWS BEDROCK COST: https://aws.amazon.com/bedrock/pricing/

questions = pd.array(pd.read_csv('prompt-files/PCL_Questions_V5.csv')['Questions'])
df = pd.read_csv('data/Labeled_Reports_2025_02_14_V02.csv')

token_count_questions = 0
for i in range(0, len(questions)):
    token_count_questions += count_tokens(questions[i])


df['token_count_IP'] = df['Report Text'].apply(lambda x: (count_tokens(x)*39)+token_count_questions)
report_input_ip = round(df['token_count_IP'].mean(),2)
procedure_input_ip = round(df['token_count_IP'].mean()/39, 2)
print("Per procedure input token count for IP prompting", procedure_input_ip)
print("Per Report input token count for IP prompting", report_input_ip)

questions = pd.array(pd.read_csv('prompt-files/PCL_Questions_V5_CoT.csv')['Questions'])
token_count_questions = 0
for i in range(0, len(questions)):
    token_count_questions += count_tokens(questions[i])
df['token_count_CoT'] = df['Report Text'].apply(lambda x: (count_tokens(x)*39)+token_count_questions)
report_input_cot = round(df['token_count_CoT'].mean(),2)
procedure_input_cot = round(df['token_count_CoT'].mean()/39, 2)
print("Per procedure input token count for CoT prompting", procedure_input_cot)
print("Per Report input token count for CoT prompting", report_input_cot)

print()


df = pd.read_csv('local_chat_history/IPO0anthropic.claude-3-5-haiku_FINAL.csv')
df['token_count']= df['reason'].apply(lambda x: count_tokens(x) if pd.notnull(x) else 0)
gen_ip_procedure = round(df['token_count'].mean(), 2)
gen_ip_report = round(df['token_count'].mean()*39, 2)

print("Per procedure Generated Token count for IP prompting", gen_ip_procedure)
print("Per procedure Report Token count for IP prompting", gen_ip_report)

df = pd.read_csv('local_chat_history/CoT0anthropic.claude_FINAL.csv')
df['token_count']= df['reason'].apply(lambda x: count_tokens(x) if pd.notnull(x) else 0)
gen_cot_procedure = round(df['token_count'].mean(), 2)
gen_cot_report = round(df['token_count'].mean()*39, 2)
print("Per procedure Generated Token count for CoT prompting", gen_cot_procedure)
print("Per procedure Report Token count for CoT prompting", gen_cot_report)


print()

ip_prompting_price_procedure = (0.0008/1000)*procedure_input_ip + (0.004/1000)*gen_ip_procedure 
print("Per procedure Price for IP prompting",  round(ip_prompting_price_procedure, 4))
print("Per Report Price for IP prompting", round(ip_prompting_price_procedure*39, 4))

cot_prompting_price = (0.0008/1000)*procedure_input_cot + (0.004/1000)*gen_cot_procedure 

print("Per procedure Price for CoT prompting", round(cot_prompting_price, 4))
print("Per Report Price for CoT prompting", round(cot_prompting_price*39, 4))



data = [
    ['IP', 'Per Procedure (Mean)', procedure_input_ip, gen_ip_procedure, round(ip_prompting_price_procedure, 4)],
    ['IP', 'Per Report (Mean)', report_input_ip, gen_ip_report, round(ip_prompting_price_procedure * 39, 4)],
    ['CoT', 'Per Procedure (Mean)', procedure_input_cot, gen_cot_procedure, round(cot_prompting_price, 4)],
    ['CoT', 'Per Report (Mean)', report_input_cot, gen_cot_report, round(cot_prompting_price * 39, 4)]
]

df_summary = pd.DataFrame(data, columns=[
    'Prompting Method', 'Category', 'Input Token', 'Generated Token', 'Cost (USD)'
])

print(df_summary)

latex_code = df_summary.to_latex(
    index=False,
    escape=False,
    float_format="%.4f" 

)
print(latex_code)