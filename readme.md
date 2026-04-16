🚕 西安市出租车轨迹数据挖掘 (Xi'an Taxi Trajectory Mining)

本项目为数学建模课程组队作业。基于 2016 年 10 月西安市出租车 GPS 轨迹数据，完成两个核心的时空数据挖掘任务：

任务 A：轨迹修复 (Trajectory Recovery) - 给定稀疏轨迹点（1/8 或 1/16 保留率），预测缺失的空间坐标。

任务 B：行程时间估计 (Travel Time Estimation, TTE) - 基于出发时间与完整的行驶路径，预测总耗时。

✨ 核心技术亮点

混合驱动的轨迹修复 (Task A)：独创 k-NN 历史轨迹形状移植 + OSM 物理路网隐马尔可夫匹配 + 时序线性插值兜底 的三段式混合架构。

高维时空特征工程 (Task B)：利用 Geohash 算法构建全西安市的 OD（起终点）空间网格通行先验矩阵，并引入“轨迹弯曲度 (Sinuosity)”等高级形态学特征。

大数据管道化训练 (Task B)：针对海量数据，设计了离线预处理管道，并利用 XGBoost 的 get_booster() 机制实现了分批次增量学习 (Incremental Learning)，彻底规避内存溢出 (OOM) 问题。

📁 项目核心目录结构

taxi_trajectory_mining/
│
├── data_processor.py         # 预处理核心：构建 k-NN 历史库、OD网格矩阵与特征分批持久化
├── features_and_utils.py     # 工具库：球面距离计算、Geohash网格化、多维特征提取
├── osm_map_matching.py       # 路网匹配模块：解析本地 OSM 数据并执行受限最短路径插值
├── task_a_main.py            # 任务 A 主程序 (执行混合修补算法)
├── task_b_main.py            # 任务 B 主程序 (执行增量模型训练与推理)
│
├── data_ds15/                # 存放官方训练集 (train.pkl, val.pkl) [需手动放入]
├── task_A_recovery/          # 存放任务 A 输入数据与答案 [需手动放入]
├── task_B_tte/               # 存放任务 B 输入数据与答案 [需手动放入]
├── processed_data/           # 自动生成：存放预处理后的 npy 特征切片与 pkl 知识库
│
├── xian_map.osm              # 西安市 OSM 本地路网源文件 [需手动放入]
├── xian_drive.graphml        # 自动生成：解析后的本地路网图缓存
│
├── modeling_methodology_report.md  # 团队数学建模方法论证报告
├── requirements.txt          # Python 依赖清单
└── .gitignore                # Git 忽略配置 (已拦截所有超大文件)


🚀 快速接入与运行指南 (Quick Start)

1. 环境配置

请确保 Python 版本 >= 3.8。克隆项目后，在根目录下执行以下命令安装依赖：

pip install -r requirements.txt


2. 本地数据准备 (非常重要)

由于数据文件和路网文件体积巨大，未上传至 GitHub。请队友们手动在本地完成以下操作：

将官方下发的 data_ds15/, task_A_recovery/, task_B_tte/ 三个数据文件夹完整拷贝到项目根目录下。

获取西安市路网：从 BBBike Extract Service 下载西安市主城区的 OSM XML 格式数据。解压后重命名为 xian_map.osm，并直接放置在项目根目录。

3. 按顺序执行流水线

本项目的数据流具有严格的依赖关系，请务必按照以下顺序运行脚本：

Step 1: 数据预处理与知识库构建

python data_processor.py


(注：此步骤将在本地扫描十几万条轨迹，构建 knn_db、od_matrix 知识库，并生成用于训练的张量切片至 processed_data/ 目录。)

Step 2: 运行任务 A（轨迹修复）

python task_a_main.py


(注：首次运行会解析 xian_map.osm 并生成 xian_drive.graphml 缓存，请耐心等待1-2分钟。随后将自动生成 val_pred_8.pkl 等预测文件。)

Step 3: 运行任务 B（行程时间估计）

python task_b_main.py


(注：将自动加载切片数据进行 XGBoost 分批次增量训练，并输出对验证集的评估结果及 val_pred.pkl。)