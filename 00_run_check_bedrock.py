import boto3
import json

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

# model_id = "meta.llama3-1-405b-instruct-v1:0"
	
# model_id = "mistral.mistral-large-2407-v1:0"

# model_id = "anthropic.claude-3-opus-20240229-v1:0"
# model_id = "anthropic.claude-v2"
model_id = "meta.llama3-1-70b-instruct-v1:0"



payload = {
    "prompt": "Human: Write a short poem about AI.\n\nAssistant:",
    # "max_tokens_to_sample": 2000, # Not supported by LLAMA MODELS
    "temperature": 0.7  # Controls randomness (0 = deterministic, 1 = more random)

}

# Convert payload to JSON string
payload_bytes = json.dumps(payload).encode("utf-8")

# Call Bedrock's InvokeModel API
response = bedrock_client.invoke_model(
    modelId=model_id,
    body=payload_bytes
)

# Parse the response
response_body = json.loads(response["body"].read())
print(response_body['generation'])