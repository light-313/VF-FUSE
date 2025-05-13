import torch
from tqdm import tqdm
from Bio import SeqIO
import h5py
import os
import warnings
from transformers import AutoModel, AutoTokenizer

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
# You can define max_length here
max_length = 102400  # Set this to an appropriate value depending on your data and model's maximum sequence length

class ESM2FeatureExtractor:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = os.path.basename(model_path).split('.')[0]
        self.layer_num = int(self.model_name.split('_')[1][1:])
        self._load_model(model_path)

    def _load_model(self, model_path):
        """加载模型及字母表"""
        print(f"正在加载模型: {model_path}...")
        
        # 使用从本地加载模型的方式
        model, tokenizer = self.load_local_esm2(model_path)
        
        # 获取字母表（tokenizer）和模型
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer  # Store tokenizer for encoding sequences
        print(f"模型已加载: {self.model_name}（层数: {self.layer_num}，设备: {self.device}）")

    @staticmethod
    def load_local_esm2(save_directory):
        """
        Load an ESM2 model and tokenizer from a local directory.

        Args:
            save_directory (str): The directory where the model and tokenizer are saved.
        """
        # Load the model and tokenizer from the local directory
        print(f"Loading model from: {save_directory}")
        model = AutoModel.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)

        print("Model and tokenizer loaded successfully.")
        return model, tokenizer

    @staticmethod
    def read_fasta(file_paths, max_length=10000):
        """读取FASTA文件"""
        seqs, labels, label_map = [], [], {}
        for path in file_paths:
            try:
                with open(path, "r") as f:
                    for record in SeqIO.parse(f, "fasta"):
                        seq = str(record.seq)
                        if len(seq) > max_length:
                            continue  # Skip sequences that are too long
                        label = record.description.split("|")[1].strip()
                        if label not in label_map:
                            label_map[label] = len(label_map)
                        seqs.append(seq)
                        labels.append(label_map[label])
            except Exception as e:
                print(f"读取文件 {path} 时出错: {e}")
        
        print(f"读取 {len(seqs)} 条序列，共 {len(label_map)} 类")
        return seqs, labels

    # def extract_features(self, sequences, labels, output_path, batch_size=8, save_format="pt"):
    #     """提取序列的特征并保存"""
    #     results = {}
    #     with torch.no_grad():  # 不计算梯度
    #         for i in tqdm(range(0, len(sequences), batch_size), desc="提取特征"):
    #             try:
    #                 batch = sequences[i:i+batch_size]
    #                 batch_labels = labels[i:i+batch_size]

    #                 inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)

    #                 # Tokenize sequences using the tokenizer
                    
    #                 # Pass through the model
    #                 out = self.model(**inputs, output_hidden_states=True)
    #                 reps = out.hidden_states[self.layer_num]  # Get representations from the specified layer
                    
    #                 for j, sid in enumerate(inputs["input_ids"]):
    #                     seq_len = len(batch[j])
    #                     results[f"seq_{i+j}"] = {
    #                         "features": reps[j, 1:seq_len+1].cpu(),  # Exclude padding tokens
    #                         "label": batch_labels[j],
    #                         "sequence": batch[j]
    #                     }
    #             except RuntimeError as e:
    #                 if "CUDA out of memory" in str(e):
    #                     print(f"[显存不足] 当前批中最大序列长度: {max(len(s) for s in batch)}，请减少 batch_size 或缩短序列。")
    #                     torch.cuda.empty_cache()  # 清空显存
    #                 else:
    #                     raise e
    

        # # 保存特征
        # self._save_features(results, output_path, save_format)
        # print(f"特征保存至 {output_path}（格式: {save_format}）")
    def extract_features(self, sequences, labels, output_path, batch_size=8, save_format="pt", pooling_method="token"):
        """提取序列的特征并保存，支持 token / mean / cls 三种方式"""
        results = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="提取特征"):
                try:
                    batch = sequences[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]

                    inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
                    out = self.model(**inputs, output_hidden_states=True)
                    reps = out.hidden_states[self.layer_num]  # [batch, seq_len, hidden_dim]

                    for j in range(len(batch)):
                        input_ids = inputs["input_ids"][j]
                        attention_mask = inputs["attention_mask"][j]
                        seq_len = attention_mask.sum().item()  # 去除 padding 的真实长度

                        if pooling_method == "token":
                            # 去掉开头的 CLS 和 padding，只保留氨基酸对应的表示
                            feature = reps[j, 1:seq_len-1].cpu()  # [real_seq_len, hidden]
                        elif pooling_method == "cls":
                            feature = reps[j, 0].cpu()  # [hidden]
                        elif pooling_method == "mean":
                            real_reps = reps[j, 1:seq_len-1]  # 去掉 [CLS] 和 padding
                            feature = real_reps.mean(dim=0).cpu()  # [hidden]
                        else:
                            raise ValueError("pooling_method 只能是 token / cls / mean")

                        results[f"seq_{i+j}"] = {
                            "features": feature,
                            "label": batch_labels[j],
                            "sequence": batch[j]
                        }

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[显存不足] 当前批中最大序列长度: {max(len(s) for s in batch)}，请减少 batch_size 或缩短序列。")
                        torch.cuda.empty_cache()
                    else:
                        raise e
            print(f"提取特征完成，共 {len(results)} 条序列")
            print(f"特征维度: {feature.shape}")

        self._save_features(results, output_path, save_format)
        print(f"特征保存至 {output_path}（格式: {save_format}）")

    def _save_features(self, results, output_path, save_format):
        """根据指定格式保存特征数据"""
        if save_format == "pt":
            torch.save(results, output_path)
        elif save_format == "h5":
            with h5py.File(output_path, 'w') as f:
                for sid, data in results.items():
                    grp = f.create_group(sid)
                    grp.create_dataset("features", data=data["features"].numpy())
                    grp.attrs["label"] = data["label"]
                    grp.attrs["sequence"] = data["sequence"]
        else:
            raise ValueError(f"不支持的保存格式: {save_format}")


# 示例用法
if __name__ == "__main__":
    # esm2_t36_3B_UR50D esm2_t33_650M_UR50D esm2_t30_150M_UR50D esm2_t12_35M_UR50D
    pooling_method="mean" #    best:mean     token / mean / cls
    model_path = "/root/autodl-tmp/.autodl/esm2_model/esm2_t33_650M_UR50D"  # 本地模型路径
    model_name = os.path.basename(model_path).split('.')[0]
    input_fasta = ["/root/VF-pred/raw_data/all_0.fa"]
    output_path = os.path.join("/root/autodl-tmp/.autodl/embedding_data", f"all_0_{model_name}_{pooling_method}.h5")
    
    # input_fasta = ["/root/VF-pred/raw_data/test.fasta"]
    # output_path = os.path.join("/root/autodl-tmp/.autodl/embedding_data", f"test_{model_name}_{pooling_method}.h5")
    
    # output_path = "/root/VF-pred/raw_data/test.h5"  # 输出路径
    
    # 创建ESM2特征提取器
    extractor = ESM2FeatureExtractor(model_path=model_path, device="cuda")

    # 读取FASTA文件
    seqs, labels = extractor.read_fasta(input_fasta,max_length=500000)

    # 提取特征并保存
    extractor.extract_features(
        sequences=seqs,
        labels=labels,
        output_path=output_path,
        batch_size=1,
        save_format="h5",  # 可以选择 "pt" 或 "h5",
        pooling_method=pooling_method # token / mean / cls
    )
