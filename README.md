# coffee_recipe_generator


## Training Environment

* Training Data
  
|Data   | Textline |
|-------|----------|
|Youtube| 1K       |

## Requirements

*see [requirements.txt](https://github.com/wcharibo/coffee_recipe_generator/blob/main/requirements.txt)

## How to install

* Clone this repository

```

git clone https://github.com/wcharibo/coffee_recipe_generator.git
cd coffee_recipe_generator
pip install -r requirements.txt

```

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


## How to use

### Using with python3.10

    python3 main.py <Youtube URL>

you have to train model with development.ipynb before run this code

## License
this project is published under Apache-2.0 license.

    
