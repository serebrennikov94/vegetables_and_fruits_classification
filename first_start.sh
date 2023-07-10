#!/bin/bash

python3 -m venv ./flask_app 

source ./flask_app/bin/activate 

pip install -r requirements.txt 

python app.py
