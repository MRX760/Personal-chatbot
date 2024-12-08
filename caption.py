from huggingface_hub import InferenceClient
from PIL import Image

client = InferenceClient(api_key="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")

image_url = "generated_img.jpg"

for message in client.chat_completion(
	model="meta-llama/Llama-3.2-11B-Vision-Instruct",
	messages=[
		{
			"role": "user",
			"content": [
                {"type": "image", "image": {"data": Image.open(image_url).tobytes(), "mime_type": "image/jpeg"}},  # Assuming JPEG; adjust if different
				{"type": "text", "text": "build a caption for this digital marketing content go-green product"},
			],
		}
	],
	max_tokens=500,
	stream=True,
):
	print(message.choices[0].delta.content, end="")