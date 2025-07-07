import requests

chatbot_url = "http://127.0.0.1:8000/baseball_info"
res = requests.post(chatbot_url, json={"query": "How many bases are there?"})
output = res.json()['response']
print(output)