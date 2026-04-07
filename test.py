import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "推荐一下千问家的产品。我现在想做一个漫画翻译的agent，我想通过提取人物图像的特征和名字进行身份匹配，保证每一面翻译时人物的一致性，同时当出现新角色时能快速判断哪些人是之前出现过的"},
    ],
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)