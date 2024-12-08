# Personal-chatbot
Personal chatbot using LLMs currently supporting huggingface API and Nvidia API with LightRAG implementation for RAG, and stable diffusion for image generation with Lora for image enhancement 

The chatbot code are using OOP method which makes this implementation much flexible for any kind of LLMs API and local running model (future work for demo)

For implementation example, please refer to chabot.py 

LightRAG documentation: [click here](https://github.com/HKUDS/LightRAG)

Supported Nvidia Model: [click here](https://build.nvidia.com/nim)


# limitations:
- large file analysis would make the input token much larger and thus raising an error.
- long chat history also makes input token much larger
- because all of the task performed by LLM, the performance of analysis and RAG are heavily depends on LLM model you're using and text-embedding model for retrieval on RAG.
- for default, this code use llama 70b with limited strong-detailed analysis compared to human.

# How to use the demo:
- install dependencies (pip install -r requirements.txt)
- modify the bot.set_api_key() to be your API key
- streamlit run streamlit.py
