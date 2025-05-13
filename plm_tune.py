import os
import warnings
import torch
import json
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from plm_train import *

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ========= ✅ Ray Tune 训练函数 (使用交叉验证) =========
def train_tune(config, train_path, device, num_classes, prot5_path=None):
    # 使用交叉验证训练，不再需要test_path参数
    # 检查是否为融合模型
    is_fusion_model = config["classifier_type"].lower() == "deltaattfusion"
    
    report = train_one_experiment(
        train_path=train_path,
        classifier_type=config["classifier_type"],
        log_dir=session.get_trial_dir(),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        patience=config["patience"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        optimizer=config["optimizer"],
        scheduler=config["scheduler"],
        device=device,
        
        # 特有参数（如果配置中存在则传递）
        rank=config.get('rank'),
        steps=config.get('steps'),
        
        # 额外参数：ProtT5特征路径（仅用于融合模型）
        prot5_path=prot5_path,
        
        # 交叉验证参数
        n_folds=5,  # 使用5折交叉验证，加快调参速度
    )
    
    # 报告平均F1分数
    tune.report({
        "f1": report["f1"],
        "acc": report["acc"],
        "sn": report["sn"],
        "sp": report["sp"],
        "mcc": report["mcc"],
        "std_f1": report["f1_std"]  # 添加F1标准差
    })


def custom_trial_name(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"


# ========= ✅ 主入口 =========
if __name__ == "__main__":
    seed_everything(666)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_embed_dir = "/root/autodl-tmp/.autodl/embedding_data"
    save_base_dir = "/root/autodl-tmp/.autodl/prot5_tune_result"
    
    # ESM特征路径
    esm_train_path = os.path.join(base_embed_dir, "train_ba_esm2_t33_650M_UR50D_mean.h5")
    # ProtT5特征路径
    prot5_train_path = "/root/autodl-tmp/train_ba_prot_features_modified.h5"
    

    # 使用更激进的早停调度器，加快搜索速度
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=30,          # 最大epochs
        grace_period=5,    # 至少训练5个epochs才考虑停止
        reduction_factor=2 # 每次减少一半的trials
    )
    
    # 增加更多的指标列用于监控
    reporter = CLIReporter(
        metric_columns=["f1", "acc", "sn", "sp", "mcc", "std_f1", "training_iteration"]
    )

    model_list = [
        # 'cnn','gru','transformer',
        # 'mlp','mamba',
        # 'bilstm','lstm', 'biomamaba', 'delta',
        # 'delta',
        'deltaattfusion'  # 添加特征融合模型
    ]

    for model_name in model_list:
        print(f"🔍 开始调参: {model_name}")
        
        # 确定使用的特征路径
        is_fusion_model = model_name.lower() == "deltaattfusion"
        train_path = esm_train_path  # ESM作为基础特征

        # 根据模型类型创建基础配置
        base_config = {
            "classifier_type": model_name,  # 固定为当前模型名
            "hidden_dim": tune.choice([128, 256, 512, 1024]),
            "num_layers": tune.choice([1, 2, 3, 4, 5,6,7]),
            "dropout": tune.uniform(0.1, 0.7),
            "batch_size": tune.choice([64, 128, 256, 512,1024]),
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "optimizer": tune.choice(["Adam", "AdamW",'SGD','RMSprop']),  # 简化优化器选择
            "scheduler": tune.choice(["none", "step", "cosine"]),
            # 固定参数
            "num_epochs": 30,  # 减少epoch数以加快调参
            "patience": 5,
        }
        
        # 根据模型类型添加特有参数
        if model_name.lower() in ['delta', 'deltaattfusion']:
            base_config.update({
                'rank': tune.choice([4, 8, 16, 32]),
                'steps': tune.choice([1, 2, 3, 4]),
            })

        # 创建搜索算法
        search_algo = OptunaSearch(metric="f1", mode="max")
        
        # 设置试验数量
        num_samples = 30 if model_name in ['delta', 'mamba', 'bimamba', 'deltaattfusion'] else 30
        
        print(f"将进行 {num_samples} 次参数搜索...")
        
        # 运行调参
        result = tune.run(
            tune.with_parameters(
                train_tune,
                train_path=train_path,
                device=device,
                num_classes=2,
                prot5_path=prot5_train_path,  # 只为融合模型传递ProtT5路径
            ),
            search_alg=search_algo,
            scheduler=scheduler,
            config=base_config,
            num_samples=num_samples,
            resources_per_trial={"cpu": 2, "gpu": 0.15},  # 增加GPU资源以加快训练
            progress_reporter=reporter,
            storage_path=os.path.join(save_base_dir, "trails", model_name),
            trial_dirname_creator=custom_trial_name,
            name=f"{model_name}_tune",
            keep_checkpoints_num=1,          # 只保留最好的检查点
            checkpoint_score_attr="f1",      # 根据f1分数保留检查点
            verbose=1,                       # 输出更详细的日志
        )

        # 获取最佳试验及其配置
        best_trial = result.get_best_trial("f1", "max", "last")
        best_config = best_trial.config
        
        # 统计搜索结果，计算参数频率
        param_stats = {}
        for param in ["hidden_dim", "num_layers", "optimizer", "scheduler"]:
            values = [t.config[param] for t in result.trials if t.last_result]
            from collections import Counter
            counts = Counter(values)
            param_stats[param] = counts
            
        print("\n参数分布统计:")
        for param, counts in param_stats.items():
            print(f"{param}: {dict(counts)}")

        print(f"\n✅ {model_name} 最优参数: {best_config}")
        print(f"📈 最终 F1: {best_trial.last_result['f1']:.4f} (std: {best_trial.last_result.get('std_f1', 0):.4f})")

        # 保存最优参数
        result_dir = os.path.join(save_base_dir, "result", model_name)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "best_config.json"), "w") as f:
            json.dump(best_config, f, indent=4)

        # ========= ✅ 完整训练并保存 (使用更多折和更多epoch) =========
        print(f"🚀 开始完整训练：{model_name}")
        final_report = train_one_experiment(
            train_path=train_path,
            classifier_type=best_config["classifier_type"],
            log_dir=result_dir,
            hidden_dim=best_config["hidden_dim"],
            num_layers=best_config["num_layers"],
            dropout=best_config["dropout"],
            batch_size=best_config["batch_size"],
            num_epochs=50,  # 在最终训练中使用更多的epoch
            patience=6,    # 增加耐心参数
            learning_rate=best_config["learning_rate"],
            weight_decay=best_config["weight_decay"],
            optimizer=best_config["optimizer"],
            scheduler=best_config["scheduler"],
            device=device,
            # 特定模型的参数
            rank=best_config.get('rank'),
            steps=best_config.get('steps'),
            # 额外参数：ProtT5特征路径（仅用于融合模型）
            prot5_path=prot5_train_path,
            # 完整交叉验证使用10折
            n_folds=10,
        )

        # 打印完整训练结果
        print(f"\n📊 {model_name} 完整训练结果：")
        print(f"📈 平均F1: {final_report['f1']:.4f} ± {final_report['f1_std']:.4f}")
        print(f"📈 平均ACC: {final_report['acc']:.4f} ± {final_report['acc_std']:.4f}")
        print(f"📈 平均SN: {final_report['sn']:.4f} ± {final_report['sn_std']:.4f}")
        print(f"📈 平均SP: {final_report['sp']:.4f} ± {final_report['sp_std']:.4f}")
        print(f"📈 平均MCC: {final_report['mcc']:.4f} ± {final_report['mcc_std']:.4f}")
        
        # 如果是融合模型，打印注意力权重信息
        if is_fusion_model and "attention_weights" in final_report:
            print(f"📊 平均注意力权重：ESM={final_report['attention_weights'][0]:.4f}, ProtT5={final_report['attention_weights'][1]:.4f}")
        
        # 保存完整训练结果
        with open(os.path.join(result_dir, "final_report.json"), "w") as f:
            json.dump(final_report, f, indent=4)
            
        # 创建摘要文件
        with open(os.path.join(result_dir, "summary.md"), "w") as f:
            f.write(f"# {model_name} 模型性能摘要\n\n")
            f.write(f"## 最佳参数\n")
            for param, value in best_config.items():
                f.write(f"- **{param}**: {value}\n")
            
            f.write(f"\n## 交叉验证性能 (10折)\n")
            f.write(f"- **F1**: {final_report['f1']:.4f} ± {final_report['f1_std']:.4f}\n")
            f.write(f"- **ACC**: {final_report['acc']:.4f} ± {final_report['acc_std']:.4f}\n")
            f.write(f"- **SN**: {final_report['sn']:.4f} ± {final_report['sn_std']:.4f}\n")
            f.write(f"- **SP**: {final_report['sp']:.4f} ± {final_report['sp_std']:.4f}\n")
            f.write(f"- **MCC**: {final_report['mcc']:.4f} ± {final_report['mcc_std']:.4f}\n")
            
            # 如果是融合模型，添加注意力权重信息
            if is_fusion_model and "attention_weights" in final_report:
                f.write(f"\n## 特征融合权重\n")
                f.write(f"- **ESM权重**: {final_report['attention_weights'][0]:.4f}\n")
                f.write(f"- **ProtT5权重**: {final_report['attention_weights'][1]:.4f}\n")
            
            f.write(f"\n## 最佳折表现\n")
            f.write(f"- **最佳折**: {final_report['best_fold']}/10\n")
            f.write(f"- **最佳F1**: {final_report['best_fold_f1']:.4f}\n")
            
        print(f"✅ {model_name} 完整训练结果已保存到: {result_dir}")
        print("="*80)