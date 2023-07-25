import os

import torch
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel, AutoAdapterModel

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# An input sentence
sentence = "It's also, clearly, great fun."

# Tokenize the input sentence and create a PyTorch input tensor
input_data = tokenizer(sentence, return_tensors="pt")

# Load pre-trained BERT model from HuggingFace Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
# It can be used with different prediction heads
model = BertAdapterModel.from_pretrained('bert-base-uncased')







# Load pre-trained task adapter from Adapter Hub
# This method call will also load a pre-trained classification head for the adapter task
adapter_name = model.load_adapter("sentiment/sst-2@ukp", config='pfeiffer')

# Activate the adapter we just loaded, so that it is used in every forward pass
model.set_active_adapters(adapter_name)

# Predict output tensor
outputs = model(**input_data)

# Retrieve the predicted class label
predicted = torch.argmax(outputs[0]).item()
assert predicted == 1




# For the sake of this demonstration an example path for loading and storing is given below
example_path = os.path.join(os.getcwd(), "adapter-quickstart")

# Save model
model.save_pretrained(example_path)
# Save adapter
model.save_adapter(example_path, adapter_name)

# Load model, similar to HuggingFace's AutoModel class,
# you can also use AutoAdapterModel instead of BertAdapterModel
model = AutoAdapterModel.from_pretrained(example_path)
model.load_adapter(example_path)














