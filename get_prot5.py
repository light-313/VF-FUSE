import os
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
import h5py
import time
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

def read_fasta(fasta_path, split_char="!", id_field=0):
    '''读取FASTA文件，返回序列字典和标签'''
    seqs = dict()
    labels = dict()
    
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                # 从头部提取ID和标签
                header = line.replace('>', '').strip()
                uniprot_id = header.split(split_char)[id_field]
                # 替换可能导致h5文件读取错误的字符
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                
                # 提取标签 (假设格式为 >id_label 或 >id|label)
                label = 1 if 'positive' in header.lower() else 0
                
                seqs[uniprot_id] = ''
                labels[uniprot_id] = label
            else:
                # 处理序列行
                seq = ''.join(line.split()).upper().replace("-","")
                # 替换非标准氨基酸
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[uniprot_id] += seq
                
    example_id = next(iter(seqs))
    print(f"Read {len(seqs)} sequences.")
    print(f"Example:\nID: {example_id}\nSequence: {seqs[example_id]}\nLabel: {labels[example_id]}")
    
    return seqs, labels

def get_embeddings(model, tokenizer, seqs, max_residues=4000, max_seq_len=1000, max_batch=256, slice_len=2000):
    '''获取序列的嵌入向量并对超过slice_len的序列进行切片'''
    protein_embs = dict()
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    
    # 使用tqdm显示进度
    progress_bar = tqdm(total=len(seq_dict), desc="Generating embeddings")
    
    start = time.time()
    batch = list()
    
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        
        # 如果序列长度超过slice_len，则进行切片
        if seq_len > slice_len:
            slices = [seq[i:i+slice_len] for i in range(0, seq_len, slice_len)]
        else:
            slices = [seq]
        
        for slice_idx, slice_seq in enumerate(slices):
            slice_seq_len = len(slice_seq)
            slice_seq = ' '.join(list(slice_seq))
            batch.append((pdb_id, slice_seq, slice_seq_len))
        
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        
        if (len(batch) >= max_batch or 
            n_res_batch >= max_residues or 
            seq_idx == len(seq_dict) or 
            seq_len > max_seq_len):
            
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()
            
            try:
                # 对批次进行编码
                token_encoding = tokenizer.batch_encode_plus(seqs, 
                                                           add_special_tokens=True, 
                                                           padding="longest")
                input_ids = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
                
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
                
                # 处理每个序列的嵌入向量
                for batch_idx, identifier in enumerate(pdb_ids):
                    s_len = seq_lens[batch_idx]
                    emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                    # 计算序列级别的表示（平均池化）
                    protein_emb = emb.mean(dim=0)
                    protein_embs[identifier] = protein_emb.detach().cpu().numpy()
                    
                    progress_bar.update(1)
                    
            except RuntimeError as e:
                print(f"RuntimeError during embedding for {pdb_ids} (max_len={max(seq_lens)})")
                print(f"Error message: {str(e)}")
                continue
    
    progress_bar.close()
    
    passed_time = time.time() - start
    avg_time = passed_time / len(protein_embs)
    
    print('\n========== EMBEDDING STATS ==========')
    print(f'Total sequences processed: {len(protein_embs)}')
    print(f'Time taken: {passed_time/60:.1f}min ({avg_time:.3f}s per sequence)')
    print('====================================')
    
    return protein_embs


def save_data(embeddings, sequences, labels, output_path):
    '''保存embeddings、sequences和labels到H5文件'''
    with h5py.File(output_path, 'w') as hf:
        # 创建组
        emb_group = hf.create_group('embeddings')
        seq_group = hf.create_group('sequences')
        label_group = hf.create_group('labels')
        
        # 保存数据
        for seq_id in embeddings.keys():
            emb_group.create_dataset(seq_id, data=embeddings[seq_id])
            seq_group.create_dataset(seq_id, data=sequences[seq_id].encode('ascii'))
            label_group.create_dataset(seq_id, data=labels[seq_id])
            
def main(fasta_path, output_path):
    '''主函数'''
    # 1. 加载模型和分词器
    print("Loading ProtT5 model and tokenizer...")
    model = T5EncoderModel.from_pretrained('Rostlab/ProstT5')
    
    model = model.to(device)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)

    # 2. 读取序列和标签
    print(f"\nReading sequences from {fasta_path}")
    sequences, labels = read_fasta(fasta_path)
    
    # 3. 生成嵌入向量
    print("\nGenerating embeddings...")
    embeddings = get_embeddings(model, tokenizer, sequences)
    
    # 4. 保存所有数据
    print(f"\nSaving data to {output_path}")
    save_data(embeddings, sequences, labels, output_path)
    print("Done!")
    
    return embeddings, sequences, labels

if __name__ == "__main__":
    # 设置输入输出路径
    seq_path = "/root/VF-pred/raw_data/all_0.fa"  # 训练集路径
    output_path = "/root/autodl-tmp/all_0_prot_features.h5"  # 输出路径
    
    # 运行主程序
    embeddings, sequences, labels = main(seq_path, output_path)