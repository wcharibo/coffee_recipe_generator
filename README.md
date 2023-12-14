# coffee_recipe_generator

Run pip install

    pip install -r requirements.txt

Add this code to project directory and Write file name like 'get_token.py'

    import requests

    def get_access_token():
        your_id = 'YOUR_CLIENT_ID'
        your_secret_code = 'YOUR_SECRET_CODE'
    
        resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={'client_id': your_id,
              'client_secret': your_secret_code}
        )
        resp.raise_for_status()
        return resp.json()['access_token']

this project used RTZR(returnzero) stt api to extract text from youtube video.   
so you need to get an api key and write at above 'YOUR_CLIENT_ID', 'YOUR_SECRET_CODE'.


    
