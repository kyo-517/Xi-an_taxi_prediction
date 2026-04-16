西安市出租车轨迹数据挖掘 (Xi'an Taxi Trajectory Mining)

本项目为数学建模课程组队作业。基于 2016 年 10 月西安市出租车 GPS 轨迹数据，完成两个核心数据挖掘任务：

任务 A：轨迹修复 (Trajectory Recovery) - 给定稀疏轨迹点，预测缺失的空间坐标。

任务 B：行程时间估计 (Travel Time Estimation, TTE) - 基于出发时间与行驶路径预测总耗时。

📁 目录结构与模块说明

项目采用了工业界标准的高度模块化设计，方便团队分工协作：

features_and_utils.py：公用模块。包含球面距离计算、针对任务B的多维特征提取（含 Geohash 与路网弯曲度计算）。

osm_map_matching.py：路网模块。负责自动下载西安市 OSM 路网，并利用有向图最短路径算法实现受物理路网约束的轨迹匹配。

task_a_main.py：任务A主程序。支持极速的时间加权线性插值，以及高精度的基于 OSM 物理路网拓扑约束的插值修复。

task_b_main.py：任务B主程序。基于提取的特征（引入 OD 网格先验均值），使用 XGBoost 梯度提升树进行高精度回归预测。

modeling_methodology_report.md：团队数学建模报告撰写参考（包含技术路线、方法论证等）。

🚀 快速开始 (Quick Start)

1. 环境配置

请确保你的 Python 版本 >= 3.8。克隆项目后，在根目录下执行以下命令安装依赖：

pip install -r requirements.txt


2. 数据准备

注意： 由于数据文件较大，未上传至 GitHub。请从课程平台下载以下数据，并将其放置在项目根目录：

data_ds15/ (包含 train.pkl, val.pkl)

task_A_recovery/ (包含输入数据和答案)

task_B_tte/ (包含输入数据和答案)

3. 运行程序

运行任务 A（轨迹修复）：

python task_a_main.py


(注：首次开启路网匹配模式时，系统会自动下载西安市机动车路网并缓存为 xian_drive.graphml，请耐心等待几分钟。)

运行任务 B（行程时间估计）：

python task_b_main.py


程序运行结束后，预测结果会自动以 .pkl 格式保存在对应的任务文件夹中。

🤝 团队协作规范

请基于 main 分支切出自己的功能分支进行开发（如：git checkout -b feature/model-tuning）。

数据文件（.pkl）和路网缓存（.graphml）已被 .gitignore 忽略，千万不要强制 git add -f 提交它们。

撰写代码时，请尽量保留关键步骤的中文注释。