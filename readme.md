# Read Me
工程分几个文件夹:  
dataset 数据处理代码  
model 模型相关代码  
tweeteval 存放数据集  
utils 辅助函数和常量代码  

其他文件:  
- main.py 训练入口  
- requirements.txt 环境
- readme.md 说明文档
- changelog.md 更新日志

## 1. 准备数据和环境
```bash
git clone https://github.com/cardiffnlp/tweeteval.git &
pip install -r requirements.txt`
```

## 2. 运行模型训练demo
   1. 无参数, 默认运行所有tasks(7个) `python main.py`
   2. 带`--task`参数, 任务列表:0: 'emoji', 1: 'emotion', 2: 'hate', 3: 'irony', 4: 'offensive', 5: 'sentiment', 6: 'stance'. 例如:
      1. `python main.py --task 1` 运行单任务, 仅使用emotion数据集训练
      2. `python main.py --task 1 2 4` 运行多任务, 使用三个数据集