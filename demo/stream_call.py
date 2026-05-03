from openai import OpenAI
import os

client = OpenAI(
    api_key = os.environ.get('DEEPSEEK_API_KEY'),
    base_url = 'https://api.deepseek.com'
)

def main():
    messages = [{'role': 'user', 'content': '给我写一篇带反转的，细思极恐的故事，不少于200字'}]
    stream_response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages = messages,
        temperature = 1,
        stream = True
    )
    for chunk in stream_response:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            print(delta_content, end = '', flush = True)
    print('\n输出完毕')
if __name__ == '__main__':
    main()
