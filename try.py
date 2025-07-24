import google.generativeai as genai

genai.configure(api_key="api key here")
models = genai.list_models()

for model in models:
    print(model.name)
