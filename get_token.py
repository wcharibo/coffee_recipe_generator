import requests

def get_access_token(clien_id, client_secret):
    resp = requests.post(
    'https://openapi.vito.ai/v1/authenticate',
    data={'client_id': clien_id,
          'client_secret': client_secret}
    )
    resp.raise_for_status()
    return resp.json()['access_token']
    
your_id = 'YOUR_CLIENT_ID'
your_secret_code = 'YOUR_SECRET_CODE'
access_token = get_access_token(your_id, your_secret_code)