import openai

openai.api_key = "your_api_key"

try:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=10
    )
    print("API key works! Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print("API key test failed:", e)
