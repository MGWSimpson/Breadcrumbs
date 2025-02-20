from openai import OpenAI
import os

var_name = os.getenv("OPENAI_API_KEY")

client = OpenAI()
OpenAI.api_key = var_name

models = client.models.list()

for model in models:
    print(model.id)
