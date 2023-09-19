
# require bitsandbytes pytorch and ...
FROM maggiebasta/llm-testing:latest

WORKDIR /home/NeurIPS_submission

# copy files
COPY . /home/NeurIPS_submission

#download weights
RUN python scripts/download_weights.py

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# install requirements of transformers
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir --upgrade transformers==4.32.0 bitsandbytes==0.41.1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]