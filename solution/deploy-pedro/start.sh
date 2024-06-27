#!/bin/bash

# Start Flask app in the background
# python3 /app/app.py &

# Start TensorFlow Serving
# tensorflow_model_server --rest_api_port=8501 --model_name=animals_model --model_base_path=/models/animals_model
# Make sure to give it execute permissions:

#!/bin/sh

export FLASK_APP=app.py
flask run --host=0.0.0.0
