import os
from openai import OpenAI

client = OpenAI(
    api_key = os.environ['DEEPSEEK_API_KEY'],
    base_url = 'https://api.deepseek.com'
)

def main():
    messages = [
        {'role': 'system', 'content': '你是一个大模型应用开发架构师，回答需要专业，简洁'},
        {'role': 'user', 'content': 'RESTful api 是什么？'}
    ]
    print('正在调用DEEPSEEK大模型')
    response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages = messages,
        temperature = 0.8
    )
    print(response.choices[0].message.content)

if __name__ == '__main__':
    main()

