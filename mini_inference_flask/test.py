import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "美联储宣布降息50bp, 投资者纷纷买入股票，股市大涨！"}
)

print(response.json())
