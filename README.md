<div align="center">    

# PyTorch Model Deployment Using Flask on Heroku

</div>
 
## Description
An example showcasing the deployment of a PyTorch model using Flask on Heroku. I'm using a very simple cats vs dogs classifer with 99% accuracy.

## How to run   
First, install dependencies (a new python virtual environment is recommended).   
```bash
# clone project   
git clone https://github.com/visualCalculus/model-deployment-example

# install requirements   
cd model-deployment-example
pip install -r requirements.txt
 ```   
### Local Deployment
Next, run `app.py`, to deploy locally.  
 ```bash
# module folder
cd model-deployment-example

# run module
python app.py
```

### Heroku Deployment

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/visualCalculus/model-deployment-example)

Note: The following buildpacks are required to deploy on Heroku (see ```app.json```):

1. Python: https://github.com/heroku/heroku-buildpack-python (for gunicorn to work)
2. Apt: https://github.com/heroku/heroku-buildpack-apt (for openCV to work)

A link to my deployed model: https://cats-vs-dogs-pytorch-app.herokuapp.com/

## License

MIT License. Check ```LICENSE``` for more details.   


