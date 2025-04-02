# AMD-gpu-lora

directml-Gpu：
#详细版本后续贴，困死了
python src/train_lora.py



CPU微调：

启动命令：$env:PYTORCH_DIRECTML=1
python train_single_file.py `
  --model_path "E:\aimodle\ai" `
  --train_data "E:\aimodle\ai\campus_dialogs.jsonl" `
  --output_dir "./lora_output"`
(根据具体情况切换）


1.创建虚拟环境
python -m venv dml_env

2.激活虚拟环境
activate

3.pip安装依赖（之前试验品torch-directml也列出来了）

transformers	         4.49.0  

peft	                 0.12.0  

torch	                 2.4.1  

datasets	             3.2.0  

tokenizers	           0.21.0  

accelerate	           1.2.1  

safetensors	           0.5.3  

torch-directml。        0.2.5.dev240914


4.下载模型

5.准备数据集

6.虚拟环境powershell并激活输入命令启动lora训练

7.等

8.powershell加载Lora


#两个py一个是训练器一个加载器

