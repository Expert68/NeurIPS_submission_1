
# require bitsandbytes pytorch and ...
FROM anibali/pytorch:1.13.1-cuda11.7

WORKDIR /home/NeurIPS_submission

# copy files
COPY . /home/NeurIPS_submission

#download weights
RUN python scripts/download_weights.py

# Setup server requriements
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# install requirements of transformers
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]