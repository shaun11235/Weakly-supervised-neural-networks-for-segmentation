# We used generative AI in an assistive role for this assessment. 
# Specifically, we used ChatGPT (GPT-4), developed by OpenAI (https://chat.openai.com/), to check the comments in our Python files, 
# as well as to identify grammar and spelling issues in the instruction file. Additionally, 
# we used it to review potential errors in our Python code to reduce bugs and improve overall robustness.

import os
import urllib.request
import tarfile

def download_and_extract(url, extract_path):
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_path, filename)

    if not os.path.exists(filepath):
        print(f" Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f" Downloaded {filename}")
    else:
        print(f" {filename} already exists.")

    print(f" Extracting {filename}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f" Extracted to {extract_path}")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "oxford-iiit-pet")
os.makedirs(data_dir, exist_ok=True)

annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

download_and_extract(annotations_url, data_dir)
download_and_extract(images_url, data_dir)
