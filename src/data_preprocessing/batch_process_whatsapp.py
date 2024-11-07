from pathlib import Path
from load_whatsapp_7b import preprocess_convo as process_conv_7b
from fire import Fire


def main(input_path, output_path):
    # convert all convos into jsonl
    fp = 'WhatsappChatNoCode.txt'
    process_conv_7b(fp, output_path, "person2")

if __name__ == "__main__":
    Fire(main)