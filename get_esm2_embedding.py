import torch
from tqdm import tqdm
from Bio import SeqIO
import h5py
import os
import warnings
from transformers import AutoModel, AutoTokenizer

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# You can define max_length here. This max_length is used by the tokenizer for truncation.
# Set this to an appropriate value depending on your data and model's maximum sequence length.
# ESM2 models typically have a max length of 1024. Be cautious with larger values if the model wasn't trained on them.
max_length = 1024

class ESM2FeatureExtractor:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = os.path.basename(model_path).split('.')[0]
        # Extract layer number from model name, assuming format like 'esm2_tX_...'
        try:
            # For esm2_t33, layer_num should be 33.
            self.layer_num = int(self.model_name.split('_')[1][1:])
            # Adjust layer index: transformers hidden_states include embedding output at index 0.
            # So, layer N's output is at index N. For a model with 33 layers (0-32), layer 33 means index 33.
            # The line `reps = out.hidden_states[self.layer_num]` correctly accesses layer `self.layer_num`.
        except (IndexError, ValueError):
            print(f"Warning: Could not parse layer number from model name '{self.model_name}'. Using default layer 33 for ESM2-T33.")
            self.layer_num = 33 # Default layer for T33 model if parsing fails

        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load model and tokenizer from a local path."""
        print(f"正在加载模型: {model_path}...")

        try:
            # Load the model and tokenizer from the local directory
            # trust_remote_code=True might be needed if the model was saved with custom code,
            # but for standard ESM2 models saved locally, it's usually not required and can be a security risk.
            # Use with caution if you are sure about the source of the saved model.
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
            print(f"模型已加载: {self.model_name}（层数: {self.layer_num}，设备: {self.device}）")
        except Exception as e:
            print(f"加载模型 {model_path} 时出错: {e}")
            raise # Re-raise the exception after printing

    @staticmethod
    def read_fasta(file_paths, max_sequence_length_filter=500000):
        """
        Reads FASTA files, extracting sequence IDs and sequences.
        Filters out sequences longer than max_sequence_length_filter.
        Does NOT extract or process labels.

        Args:
            file_paths (list): List of paths to FASTA files.
            max_sequence_length_filter (int): Maximum length of sequences to include.
                                               Sequences longer than this will be skipped.
                                               Note: This is different from the tokenizer's max_length.

        Returns:
            tuple: A tuple containing:
                - list: List of sequence IDs (str).
                - list: List of sequences (str).
        """
        sequence_ids = []
        sequences = []
        read_count = 0
        skipped_count = 0

        for path in file_paths:
            try:
                with open(path, "r") as f:
                    for record in SeqIO.parse(f, "fasta"):
                        read_count += 1
                        seq = str(record.seq)
                        if len(seq) > max_sequence_length_filter:
                            skipped_count += 1
                            continue  # Skip sequences that are too long for the filter

                        # Use record.id as the sequence identifier
                        sequence_ids.append(record.id)
                        sequences.append(seq)

            except FileNotFoundError:
                print(f"错误: 文件未找到 - {path}")
            except Exception as e:
                print(f"读取文件 {path} 时出错: {e}")

        print(f"从 {len(file_paths)} 个文件读取 {read_count} 条记录.")
        print(f"跳过 {skipped_count} 条长度超过 {max_sequence_length_filter} 的序列.")
        print(f"处理 {len(sequence_ids)} 条序列.")

        return sequence_ids, sequences

    def extract_features(self, sequence_ids, sequences, output_path, batch_size=8, save_format="h5"):
        """
        Extracts features for sequences using the MEAN pooling method and saves them.
        Saves sequence ID, features, and original sequence. Does NOT save label.

        Args:
            sequence_ids (list): List of sequence IDs.
            sequences (list): List of sequences.
            output_path (str): Path to save the output file.
            batch_size (int): Batch size for model inference.
            save_format (str): Format to save features ("pt" for PyTorch tensor, "h5" for HDF5).
        """
        results = {}
        pooling_method = "mean" # Fixed pooling method
        print(f"使用固定的 pooling 方法: {pooling_method}")

        with torch.no_grad():
            # Iterate through batches of sequences and IDs
            for i in tqdm(range(0, len(sequences), batch_size), desc="提取特征"):
                try:
                    batch_sequences = sequences[i:i+batch_size]
                    batch_ids = sequence_ids[i:i+batch_size]

                    # Tokenize sequences using the tokenizer's max_length
                    # padding=True adds padding tokens to the shorter sequences in the batch
                    # truncation=True truncates sequences longer than max_length
                    inputs = self.tokenizer(batch_sequences, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)

                    # Pass through the model
                    out = self.model(**inputs, output_hidden_states=True)
                    # Get representations from the specified layer.
                    # hidden_states list contains embeddings at index 0, then layer outputs from 1 to num_layers.
                    # So, for layer N, we access index N.
                    if self.layer_num >= len(out.hidden_states):
                         print(f"Error: Specified layer_num ({self.layer_num}) is out of range for model with {len(out.hidden_states)-1} layers.")
                         # Fallback to the last layer if specified layer is invalid
                         print(f"Using the last layer ({len(out.hidden_states)-1}) instead.")
                         reps = out.hidden_states[-1]
                    else:
                         reps = out.hidden_states[self.layer_num]  # [batch, seq_len, hidden_dim]


                    # Process each sequence in the batch
                    for j in range(len(batch_sequences)):
                        current_id = batch_ids[j]
                        current_sequence = batch_sequences[j]
                        attention_mask = inputs["attention_mask"][j]
                        # The true length of the sequence including CLS and SEP tokens, excluding padding
                        seq_len_with_special = attention_mask.sum().item()

                        # Apply the fixed MEAN pooling method
                        # Mean pooling: Average the representations of all amino acid tokens (excluding CLS and SEP)
                        if seq_len_with_special > 2: # Ensure there are actual tokens besides CLS and SEP
                            # real_reps is the slice excluding CLS (index 0) and SEP (last token before padding)
                            real_reps = reps[j, 1:seq_len_with_special-1]
                            feature = real_reps.mean(dim=0).cpu()  # [hidden_dim]
                        else: # Handle cases with only CLS/SEP or very short truncated sequences
                            # Fallback for mean pooling when no tokens to average: use CLS token representation
                            # This aligns with common practice for very short sequences
                            feature = reps[j, 0].cpu()
                            # Optional warning for fallback:
                            # print(f"Warning: Using CLS token for mean pooling fallback on sequence {current_id} due to insufficient tokens after tokenization.")

                        # Store results using sequence ID as the key
                        results[current_id] = {
                            "features": feature,
                            "sequence": current_sequence # Save the original sequence
                        }

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"\n[显存不足] 处理批次 {i//batch_size} (序列 {i}-{min(i+batch_size-1, len(sequences)-1)}) 时出错.")
                        # Find the longest sequence in the current batch for better error message
                        longest_seq_in_batch = max(len(s) for s in batch_sequences)
                        print(f"当前批次中最大原始序列长度: {longest_seq_in_batch}. 请减少 batch_size 或缩短 FASTA 文件中的序列。")
                        torch.cuda.empty_cache() # Clear CUDA cache
                        # Decide whether to continue or exit on OOM. Continuing might lead to more OOMs
                        # but allows processing smaller sequences in subsequent batches.
                        continue # Skip the rest of this batch and continue with the next one
                    else:
                        # Re-raise other types of RuntimeErrors
                        print(f"\n[错误] 处理批次 {i//batch_size} (序列 {i}-{min(i+batch_size-1, len(sequences)-1)}) 时发生运行时错误: {e}")
                        raise e # Exit on other unexpected runtime errors
                except Exception as e:
                     print(f"\n[错误] 处理批次 {i//batch_size} (序列 {i}-{min(i+batch_size-1, len(sequences)-1)}) 时发生未知错误: {e}")
                     # Decide whether to continue or exit on other errors
                     raise e # Exit on any other unexpected error


        print(f"提取特征完成，共 {len(results)} 条序列")
        # Print the shape of the first extracted feature as an example
        if len(results) > 0:
             first_feature_key = list(results.keys())[0]
             first_feature = results[first_feature_key]['features']
             print(f"第一个序列 ({first_feature_key}) 的特征维度: {first_feature.shape}")
        else:
             print("没有提取到任何特征。请检查输入文件、序列长度过滤和模型加载。")


        # Save the extracted features
        self._save_features(results, output_path, save_format)
        print(f"特征保存至 {output_path}（格式: {save_format}）")


    def _save_features(self, results, output_path, save_format):
        """
        Saves the extracted feature data according to the specified format.
        Saves features and sequence, does NOT save label.
        """
        if save_format == "pt":
            # Save results dictionary directly (contains ID, features, sequence)
            torch.save(results, output_path)
        elif save_format == "h5":
            try:
                with h5py.File(output_path, 'w') as f:
                    for sid, data in results.items():
                        # Use sequence ID as the group name
                        grp = f.create_group(str(sid)) # Ensure sid is a string for HDF5 group names
                        # Save features dataset
                        grp.create_dataset("features", data=data["features"].numpy())
                        # Save sequence as an attribute
                        grp.attrs["sequence"] = data["sequence"]
                        # Do NOT save label: grp.attrs["label"] = data["label"]

            except Exception as e:
                print(f"保存 HDF5 文件 {output_path} 时出错: {e}")
                raise # Re-raise the exception
        else:
            raise ValueError(f"不支持的保存格式: {save_format}. 支持的格式为 'pt' 或 'h5'.")


# Example Usage
if __name__ == "__main__":
    # --- Configuration ---
    # Fixed model path for esm2_t33_650M_UR50D
    model_dir_name = "esm2_t33_650M_UR50D"
    base_model_path = "/root/autodl-tmp/.autodl/esm2_model" # Adjust if your base path is different
    model_path = os.path.join(base_model_path, model_dir_name)

    # List of input FASTA file paths
    input_fasta_files = ["/root/VF-pred/raw_data/all_0.fa"]

    # Directory to save the output file
    output_dir = "/root/autodl-tmp/.autodl/embedding_data" # Adjust if your output path is different

    # Fixed output filename
    output_filename = "esm2embeding.h5" # Ensure the extension matches the save_format below
    output_path = os.path.join(output_dir, output_filename)

    # Maximum sequence length to read from FASTA (filter before tokenization)
    # Set this based on your data and available memory.
    # Note: Tokenizer will further truncate based on the model's max_length (defined at the top).
    fasta_read_max_length = 500000 # Example: Filter sequences longer than 500k

    # --- Execution ---
    try:
        # Create ESM2 feature extractor instance
        # The constructor will print the loaded model name and layer number
        extractor = ESM2FeatureExtractor(model_path=model_path, device="cuda")

        # Read FASTA files, getting only IDs and sequences (labels are ignored)
        print("\n读取 FASTA 文件...")
        sequence_ids, sequences = extractor.read_fasta(input_fasta_files, max_sequence_length_filter=fasta_read_max_length)

        # Check if any sequences were read
        if not sequences:
            print("\n没有读取到有效序列，程序退出。请检查输入文件和序列长度过滤设置。")
        else:
            # Extract features and save using the fixed mean pooling
            print("\n开始提取特征...")
            extractor.extract_features(
                sequence_ids=sequence_ids,
                sequences=sequences,
                output_path=output_path,
                batch_size=1, # Adjust batch size based on GPU memory and sequence length. Batch size 1 is safe but slow.
                save_format="h5" # Fixed save format
                # pooling_method is fixed to "mean" inside extract_features
            )
            print("\n特征提取和保存完成。")

    except FileNotFoundError:
        print(f"\n错误: 确保模型路径 {model_path} 和输入文件 {input_fasta_files} 存在。")
    except Exception as e:
        print(f"\n程序执行过程中发生错误: {e}")