# Personal-chatbot
Personal chatbot using LLMs currently supporting huggingface API and Nvidia API with LightRAG implementation for RAG, and stable diffusion for image generation with Lora for image enhancement 

The chatbot code are using OOP method which makes this implementation much flexible for any kind of LLMs API and local running model (future work for demo)

For implementation example, please refer to chabot.py 

LightRAG documentation: [click here](https://github.com/HKUDS/LightRAG)

Supported Nvidia Model: [click here](https://build.nvidia.com/nim)

Lora and SD checkpoint: [click here](https://drive.google.com/drive/folders/1_AOVmKPLZCHogUpo9m6IPJMpWMvON7O0?usp=sharing)

# limitations:
- large file analysis would make the input token much larger and thus raising an error.
- long chat history also makes input token much larger
- because all of the task performed by LLM, the performance of analysis and RAG are heavily depends on LLM model you're using and text-embedding model for retrieval on RAG.
- for default, this code use llama 70b with limited strong-detailed analysis compared to human.
- RAG method are using [LightRAG](https://github.com/HKUDS/LightRAG)

# How to use the demo:
- install dependencies (pip install -r requirements.txt)
- make sure you've cloned the latest [LightRAG](https://github.com/HKUDS/LightRAG) sub-module used in this repo
- Put [Lora and SD checkpoint](https://drive.google.com/drive/folders/1_AOVmKPLZCHogUpo9m6IPJMpWMvON7O0?usp=sharing) inside stable-diffusion/models corresponding folder. However, it's customizable on your needs but you'll have to modify the streamlit_app.py code.
- modify the bot.set_api_key() to be your API key
- streamlit run streamlit.py

# Starting guide on demo
- send `/relearn` command to chatbot to begin re-learning on every file inside [data folder](https://github.com/MRX760/Personal-chatbot/tree/main/data)
- use command `/rag what-to-do` to use RAG in chatbot
- to perform analysis, simply upload file into chatbot and send this syntax: `/analyze filename.extension what-to-do` or `/analyze filename.extension` for general information analysis

![Button to upload file into chatbot](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/1.png)

- to save an uploaded file into /data files for RAG, simply use `/save` command and chatbot will automatically re-learn without using `/relearn` command. 
- to perform image generation, use english word like `generate me an image of..` or `make me an image..`. Currently, the algorithm works by detecting the patterns of user prompt in english.

# Screenshots of Nvidia API NIM Demo

## Document Analysis Example

### Finding Page
![Finding page](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/2.png | width=600)
<!-- *This screenshot shows the initial page finding process.* -->

### Analyzing CSV File
![Analyzing csv file](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/3.png | width=600)
> **Note:** Performance depends on the type of LLM. The cut-off part of the chat is influenced by the limitations of the LLM used in the screenshot.

### Analyzing PDF File
![Analyzing pdf file](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/4.png | width=600) 
<!-- *This screenshot illustrates the process of analyzing a PDF file.* -->

## Image Generation Examples

### Mouse Image
![Mouse](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/1.jpg | width=600)
<!-- *Generated image of a mouse.* -->

### House Image
![House](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/2.jpg | width=600)
<!-- *Generated image of a house.* -->
