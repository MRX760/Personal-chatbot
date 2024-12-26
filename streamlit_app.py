#library for UI
import streamlit as st 
from io import StringIO
import pandas as pd

#file handling
import os 
import io
import PyPDF2
from docx import Document
import json
import shutil

#functionality
from chatbot import nvidia_llm_api, SD
from utils import ask_img, create_session_dir, save_session_file, save_to_database, create_session_log, update_session_log, base64_to_image
import asyncio

@st.cache_resource
def initialization():
    #session directory
    temp_file_dir = create_session_dir()
    temp_log_dir = create_session_dir()
    temp_log = create_session_log(temp_log_dir.name)

    #client declaration
    
    #nvidia
    bot = nvidia_llm_api()
    bot.set_api_key("nvapi-xxx")
    # bot.connect(work_dir='./work_dir', model="meta/llama3-70b-instruct")
    bot.connect(temp_dir = temp_file_dir.name, work_dir='./work_dir', model="nvidia/llama-3.1-nemotron-70b-instruct")
    bot.session_memory = temp_log

    #hf
    # bot = huggingface_llm_api()
    # bot.set_api_key("hf_xxxx")
    # bot.connect()

    #SD model declaration
    bot_img = SD()
    # bot_img.load_model(os.path.join("..\stable-diffusion-webui\models\Stable-diffusion", "realisticVisionV60B1_v51HyperVAE.safetensors"))
    # bot_img.load_lora("..\stable-diffusion-webui\models\Lora", ['add_detail.safetensors', 'neon_palette_offset.safetensors', 'more_details.safetensors'], 
    #                 [1.0, 1.0, 0.5])
    # bot_img.activate_lora(["add_detail", "neon_palette_offset", "more_details"])
    bot_img.load_model(os.path.join(".\stable-diffusion\models\SD", "realisticVisionV60B1_v51HyperVAE.safetensors"))
    bot_img.load_lora(".\stable-diffusion\models\Lora", ['add_detail.safetensors', 'neon_palette_offset.safetensors', 'more_details.safetensors'], 
                    [1.0, 1.0, 0.5])
    bot_img.activate_lora(["add_detail", "neon_palette_offset", "more_details"])

    return bot, bot_img, temp_file_dir, temp_log_dir, temp_log

available_command = ['/save', '/rag', '/analyze', '/relearn']
db_changes = False
bot, bot_img, temp_file_dir, temp_log_dir, temp_log = initialization()

#opposite bubbless
st.markdown(
            """
        <style>
            .st-emotion-cache-janbn0 {
                flex-direction: row-reverse;
                text-align: right;
            }
            
            .header-container{
            position: -webkit-sticky; /* For Safari */
            position: sticky;
            top: 0;
            # background-color: white;
            padding: 10px;
            z-index: 1000; /* Make sure it appears above other content */
            border-bottom: 2px solid #ddd; /* Optional: Add a border */
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

#header
with st.container(key="header-container"):
    st.title("💬 Chatbot")
    st.write("Streamlit version: " + st.__version__)

#declaration of chat beginning
if 'messages' not in st.session_state:
    if temp_log:
        try:
            with open(temp_log, 'r') as log:
                content = json.loads(log.read())
                if content:
                    st.session_state["messages"] = content["messages"]
                else:
                    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there, I'm a chatbot. Not as good as monetized GPT but i will try my best 🤖😊"}]    
        except Exception as e:
            print("caught error when trying to read log.", e)
            st.session_state["messages"] = [{"role": "assistant", "content": "Hi there, I'm a chatbot. Not as good as monetized GPT but i will try my best 🤖😊"}]
        finally:
            update_session_log(dict(st.session_state), temp_log)


    else:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there, I'm a chatbot. Not as good as monetized GPT but i will try my best 🤖😊"}]
        update_session_log(dict(st.session_state), temp_log)


#loop for writing all messages into the UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    # Check if the message includes an image
    if "image" in msg:
        with st.chat_message(msg["role"]):
            try:
                img = base64_to_image(msg["image"]["data"])
                st.image(img, caption="Generated Image")
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
            except:
                st.image(msg["image"], caption="Generated Image")
                img_buffer = io.BytesIO()
                msg["image"].save(img_buffer, format="PNG")
                img_buffer.seek(0)

#sidebar
with st.sidebar:
    #doc upload container
    upload_container = st.container(border=True)
    upload_container.subheader("Upload File")
    uploaded_file = upload_container.file_uploader("Choose a file", 
                                                   type=["txt", "pdf", "docx", "csv"], 
                                                   accept_multiple_files=True,
                                                   label_visibility="collapsed")
    
    # print(uploaded_file)
    df = pd.DataFrame({"Filename": os.listdir('./data')})
    temp_df = pd.DataFrame({"Filename":os.listdir(temp_file_dir.name)})    

    if uploaded_file:                
        for file in uploaded_file:
            save_session_file(file.getbuffer(), file.name, temp_file_dir.name)
        temp_df = pd.DataFrame({"Filename":os.listdir(temp_file_dir.name)})    
            
    #temporary document list
    if len(temp_df) > 0 or temp_file_dir != "":
        temp_container = st.container()
        temp_container.subheader("Document analyzer database (temporary)", divider=True)
        temp_container.data_editor(temp_df)

    if len(df) > 0 or db_changes:
        #permanent document list container
        dtframe = st.container()
        dtframe.subheader("RAG database", divider=True)
        dtframe.data_editor(df)

if prompt:= st.chat_input():
    #write message to UI
    st.chat_message("user").write(prompt)

    #generating image
    if ask_img(prompt):
        #keyword
        # with st.status("Generating keywords", expanded=False) as status:
        with st.spinner('Generating keywords'):
            prompt, negative_prompt = bot.generate_image_keyword(prompt=prompt)
            # status.update(label="Generating Image", expanded=False)
        #img resolution
        #     w,h = bot.generate_image_resolution(prompt=prompt)
        # print(w,h)
        # update param
        # bot_img.update_param(width=w, height=h)
        #generate image
        try:
            with st.spinner('generating image'):
                img = bot_img.generate_image(prompt=prompt, negative=negative_prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role":"assistant", "content": "Generated Image as requested: ", "image":img })
            st.chat_message("assistant").write("Generated Image as requested: ")

            # Display the image immediately
            chat_block = st.chat_message("assistant")
            with chat_block:
                st.image(img, caption="Generated Image")
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
        # save_log(messages=log)
        except Exception as e: 
            message = f"Error on generating image, try to re-define the SD declaration to be the exact path of your SD model and Lora you're using. The program encountered following error: {e}"
            st.session_state.messages.append({"role":"assistant", "content": message})
            st.chat_message("assistant").write(message)

    elif any(item in available_command for item in prompt.split(" ")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if sum(1 for item in available_command if item in prompt.split(" ")) > 1:
            st.chat_message("assistant").write("Generated Image as requested: ")
        else:
            splitted_prompt = prompt.split(" ")
            if splitted_prompt[0] == '/save':
                if len(splitted_prompt) > 2 or len(splitted_prompt) == 1:
                    #error  response
                    st.session_state.messages.append({"role": "assistant", "content": "Use following syntax to insert uploaded file into RAG database '/save [filename_in_temporary_database]'"})
                    st.chat_message("assistant").write("Use following syntax to insert uploaded file into RAG database '/save [filename_in_temporary_database] \n note: file shouldn't have spacing'")            
                else:
                    db_changes = True
                    status = save_to_database(splitted_prompt[1], temp_file_dir.name)
                    # asyncio.run(bot.re_learn())
                    bot.re_learn()
                    st.chat_message("assistant").write(status)
                    st.session_state.messages.append({"role": "assistant", "content": status})
            
            elif splitted_prompt[0] == '/relearn':
                st.session_state.messages.append({"role": "assistant", "content": "Please wait during learning process.."})
                st.chat_message("assistant").write("Please wait during learning process..")
                # asyncio.run(bot.re_learn())
                bot.re_learn()
                st.session_state.messages.append({"role": "assistant", "content": "Learning completed.. You may continue our conversation"})
                st.chat_message("assistant").write("Learning completed.. You may continue our conversation")
                

            elif splitted_prompt[0] == '/rag':
                # content = asyncio.run(bot.search(prompt.split(" ",1)[-1]))
                if prompt.split(" ",1)[-1] in ['/rag']:
                    st.chat_message("assistant").write(f"The syntax is: /rag < what to do>")
                    st.session_state.messages.append({"role": "assistant", "content": f"The syntax is: /rag < what to do>"})
                else:
                    content = bot.search(prompt.split(" ",1)[-1])
                    st.session_state.messages.append({"role": "assistant", "content": content})
                    st.chat_message("assistant").write(content)
            
            elif splitted_prompt[0] == '/analyze':
                filename, prompt = prompt.split(" ", 2)[1], prompt.split(" ", 2)[-1]
                if filename == prompt:
                    prompt = "Analyze me the document to find insightful and general information of the document"
                chunks = bot.read_per_page(os.path.join(temp_file_dir.name, filename))
                response = bot.summarize(prompt=prompt, chunks=chunks)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

    #normal chat
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        respond = bot.send_chat_request(prompt=prompt)
        st.session_state.messages.append({"role": "assistant", "content": respond})
        st.chat_message("assistant").write(respond)
    
    if '/' in prompt.split(" ")[0] and not any(item in available_command for item in prompt.split(" ")):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("assistant").write(f"If you're trying to use command, here's available command list: {available_command}")
                st.session_state.messages.append({"role": "assistant", "content": f"If you're trying to use command, here's available command list: {available_command}"})

    #storing session log
    update_session_log(dict(st.session_state), temp_log)
    
