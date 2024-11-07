# PersonalChatbot
## Whatsapp based personal chatbot

### Instructions
  1. Clone this repository to your machine.
  2. Make sure you have all the required libraries and versions that appear in the requirements.txt file.
  3. Export desired conversation from Whatsapp and extract the text file. Name it "WhatsappChatOriginal.txt" (without the quotations), and place it in the "PersonalChatbot/src/data_preprocessing" directory.
  4. Fill in the folowing template with your desired model if different from the default,<br />
     the full path to where you are storing your data <br />
     (path must end with PersonalChatbot/src/data_prepocessing/final_outputs/formatted_data.jsonl),<br />
     and the chosen path to where you want to save the final model:<br />
<br />
<br />
      ###################################################<br />
      ################## Fill This In ########################<br />
      base_model_id = "yam-peleg/Hebrew-Mistral-7B"<br />
      path_to_data = "full/path/to/PersonalChatbot/src/data_prepocessing/final_outputs/formatted_data.jsonl"<br />
      saved_model_path = "path/to/save/model"<br />
      ###################################################<br />
      ###################################################<br />
<br />
<br />
     copy and paste this to the following 3 files in this project to replace the empty ones that are in the files by default:
       <ul>a. PersonalChatbot/src/chat/chat_bot.py</ul>
       <ul>b. PersonalChatbot/src/data_preprocessing/load_whatsapp.py</ul>
       <ul>c. PersonalChatbot/src/training/fine_tuning_model.py</ul>
  6. Change directory to PersonalChatbot/src/data_processing and run: python3 preprocess_start.sh
     <ul>Your processed data should appear in a final_outputs folder and should be called formatted_data.jsonl</ul>
  7. Change directory to PersonalChatbot/src/training
     <ul>run the following commant: python3 fine_tuning_model.py</ul>
     <ul>This can take a while and the duration depends on the amount of data in your Whatsapp chat.</ul>
  8. When step 6 is complete change directory to PersonalChatbot/src/chat</ul>
     <ul>run the following command: python3 chat_bot.py</ul>
     <ul>This will create a file named conversation.txt where your conversation with the model will appear. Type your input into the terminal and see the responses in the file conversation.txt</ul>

