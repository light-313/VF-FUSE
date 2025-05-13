from transformers import AutoModel, AutoTokenizer

def load_and_save_esm2(model_name, save_directory):
    """
    Load an ESM2 model from Hugging Face and save it locally.

    Args:
        model_name (str): The name of the ESM2 model on Hugging Face.
        save_directory (str): The directory to save the model and tokenizer.
    """
    # Load the model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer locally
    print(f"Saving model and tokenizer to: {save_directory}")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    # source /etc/network_turbo
    # esm2_t36_3B_UR50D esm2_t33_650M_UR50D esm2_t30_150M_UR50D esm2_t12_35M_UR50D
    esm2_model_name = "facebook/esm2_t36_3B_UR50D"  # Replace with the desired ESM2 model name
    save_path = "/root/autodl-tmp/.autodl/esm2_model/esm2_t36_3B_UR50D"  # Replace with your desired save path

    load_and_save_esm2(esm2_model_name, save_path)