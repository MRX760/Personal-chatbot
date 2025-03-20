# Personal-chatbot
Personal chatbot using LLMs currently supporting huggingface API and Nvidia API with LightRAG implementation for RAG, and stable diffusion for image generation with Lora for image enhancement 

The chatbot code are using OOP method which makes this implementation much flexible for any kind of LLMs API and local running model (future work for demo)

For implementation example, please refer to chabot.py 

LightRAG documentation: [click here](https://github.com/HKUDS/LightRAG)

Supported Nvidia Model: [click here](https://build.nvidia.com/nim)

Lora and SD checkpoint: [click here](https://drive.google.com/drive/folders/1_AOVmKPLZCHogUpo9m6IPJMpWMvON7O0?usp=sharing)

### Streamlit app code flow
![Analyzing pdf file](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/code-flow.png)
<!-- *This screenshot illustrates streamlit app code flow.* -->
> **Note:** Future work: Update on huggingface API for LightRAG implementation example.

# Limitations:
- Large file analysis would make the input token much larger and thus raising an error.
- Long chat history also makes input token much larger
- Because all of the task performed by LLM, the performance of analysis and RAG are heavily depends on LLM model you're using and text-embedding model for retrieval on RAG.
- In default, this code use llama 70b with limited strong-detailed analysis compared to human.
- RAG method are using [LightRAG](https://github.com/HKUDS/LightRAG)

# How to use the demo:
<!-- - Install LightRAG `pip install lightrag-hku` -->
- Install dependencies `pip install -r requirements.txt`
<!-- - Make sure you've cloned the latest [LightRAG](https://github.com/HKUDS/LightRAG) sub-module used in this repo. Using command `git clone https://github.com/HKUDS/LightRAG.git` -->
- Put [Lora and SD checkpoint](https://drive.google.com/drive/folders/1_AOVmKPLZCHogUpo9m6IPJMpWMvON7O0?usp=sharing) inside stable-diffusion/models corresponding folder. However, it's customizable on your needs but you'll have to modify the streamlit_app.py code.
- Insert your API key inside the `bot.set_api_key("nvapi-xxxx")` to use nvidia NIM or `bot.set_api_key("hf_xxxx")` to use hf model.
- To run streamlit GUI, run this on your environment command line or prompt: `streamlit run streamlit_app.py`

# Starting guide on demo
- Send `/relearn` command to chatbot to begin re-learning on every file inside [data folder](https://github.com/MRX760/Personal-chatbot/tree/main/data)
- Use command `/rag what-to-do` to use RAG in chatbot
- To perform analysis, simply upload file into chatbot and send this syntax: `/analyze filename.extension what-to-do` or `/analyze filename.extension` for general information analysis

![Button to upload file into chatbot](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/1.png)

- To save an uploaded file into /data files for RAG, simply use `/save` command and chatbot will automatically re-learn without using `/relearn` command. 
- To perform image generation, use english word like `generate me an image of..` or `make me an image..`. Currently, the algorithm works by detecting the patterns of user prompt in english.

# Screenshots of Nvidia API NIM Demo

## Document Analysis Example

### Finding Page
![Finding page](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/2.png)
<!-- *This screenshot shows the initial page finding process.* -->

### LightRAG on csv file
![LightRAG on csv file reference](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/6.png)
> **Note:** LightRAG erformance depends on the type of LLM.

### Analyzing PDF File
![Analyzing pdf file](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/4.png)
<!-- *This screenshot illustrates the process of analyzing a PDF file.* -->

## Image Generation Examples

### Mouse Image
![Mouse](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/1.jpg)
<!-- *Generated image of a mouse.* -->

### House Image
![House](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/2.jpg)
<!-- *Generated image of a house.* -->

### Swamp Image
![House](https://github.com/MRX760/Personal-chatbot/blob/main/documentation/5.png)
<!-- *Generated image of a swamp.* -->