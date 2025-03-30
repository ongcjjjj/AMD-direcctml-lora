# AMD-cpu-lora
启动命令：$env:PYTORCH_DIRECTML=1
python train_single_file.py `
  --model_path "E:\aimodle\ai" `
  --train_data "E:\aimodle\ai\campus_dialogs.jsonl" `
  --output_dir "./lora_output"`



1.创建虚拟环境
python -m venv dml_env

2.激活虚拟环境
.\activate

3.pip安装依赖

transformers	         4.49.0	
peft	                 0.12.0
torch	                 2.4.1
datasets	             3.2.0	
tokenizers	           0.21.0	
accelerate	           1.2.1	                
safetensors	           0.5.3	
torch-directml	       0.2.5.dev240914

4.下载模型

5.准备数据集

6.虚拟环境powershell启动lora训练

7.等

8.加载
