import subprocess
import itertools
import csv
import time

# 超参数搜索空间
batch_sizes = [128, 256, 512]
learning_rates = [1e-4, 1e-3, 1e-2]
num_epochs_list = [10, 50, 100]

# 训练模式列表
modes = ["regression", "classification"]

# 输出结果 CSV
output_file = "hyperparam_results.csv"

# 打开 CSV 文件写入表头
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "mode", "batch_size", "learning_rate", "num_epochs", 
        "train_time_sec", "MAE", "R2", "F1", "AUC"
    ])

    # 遍历训练模式和超参数组合
    for mode in modes:
        for batch_size, lr, num_epochs in itertools.product(batch_sizes, learning_rates, num_epochs_list):
            print(f"Running: mode={mode}, batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}")

            cmd = [
                "python", "train.py",
                "--mode", mode,
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--num_epochs", str(num_epochs)
            ]

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            train_time = time.time() - start_time
            output = result.stdout

            # 初始化指标
            MAE, R2, F1, AUC = None, None, None, None

            # 解析输出指标
            for line in output.splitlines():
                if "Evaluation results on your eval set:" in line:
                    if mode == "regression":
                        parts = line.split("mae:")[1].split(", R2:")
                        MAE = float(parts[0].strip())
                        R2 = float(parts[1].strip())
                    else:  # classification
                        parts = line.split("F1:")[1].split(", AUC:")
                        F1 = float(parts[0].strip())
                        AUC = float(parts[1].strip())

            # 写入 CSV
            writer.writerow([mode, batch_size, lr, num_epochs, train_time, MAE, R2, F1, AUC])

            print(f"Completed: mode={mode}, batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, time={train_time:.2f}s")
