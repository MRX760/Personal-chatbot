from utils import connect, send_chat_request, save_log, read_log, send_img_request, ask_img
import os

Key = "hf_xxxx"
log_address = 'log_dump.txt'  # Could be changed
if os.path.isfile(log_address):
    log = read_log(log_address)
else:
    log = []

client = connect(API_KEY=Key)
while True:
    prompt = input("User: ")  # Removed str() since input() already returns a string
    if prompt=="":
        print("\nWrite something..\n")
    else:
        # if ask_img(prompt):
            # send_img_request(prompt=prompt, API_KEY=Key)
            # log.append({"role":"user", "content": prompt})
            # save_log(messages=log)
        respond, log = send_chat_request(prompt=prompt, client=client, logs=log)
        print("\nChatbot: ",respond, "\n")
        save_log(log, filename=log_address)
            #   if ask_img(respond):
            #       print(respond.split('"',3))[1]
                #   send_img_request(prompt=prompt, API_KEY=Key)
                  
