import requests

res = requests.get('https://api.kalmbach.dev/ping-me')
print("API called, response code:", res.status_code)