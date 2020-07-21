import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Hours':9.25})

print(r.json())