import requests
tokenline = 'jbwYqshJvjD3LnCrwLSBs3sBo828310l8JYh0LVdifg'
headers = {'Authorization':'Bearer ' + tokenline}
payload = {"message": 'hi'}
files = {'imageFile': open('Fine.jpg', 'rb')}
response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=payload, files=files)

