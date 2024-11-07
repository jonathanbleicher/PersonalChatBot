python3 remove_media_omitted.py WhatsappChatOriginal.txt WhatsappChatNoMO.txt
python3 remove_code.py WhatsappChatNoMO.txt WhatsappChatNoCode.txt
python3 batch_process_whatsapp.py --input_path "./" -output_path "./final_outputs/formatted_data.jsonl"