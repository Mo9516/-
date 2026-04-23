猫爪识别系统 — 基于 YOLOv8 的猫爪分类与性格识别
项目简介
本项目是基于 YOLOv8 实现的猫爪细粒度识别系统，可对 6 类猫爪进行自动检测与分类，并根据爪形输出对应性格说明。系统包含数据预处理、模型训练、Web 可视化界面，适用于图像识别课程设计与目标检测实训。
识别类别
爱心型
富士山型
米粒型
饭团型
火箭型
三叶草型
文件说明
M1.py — 数据预处理
构建 YOLO 格式数据集目录
图片与标签配对检查
按 8:2 划分训练集与验证集
统计类别分布
M2.py — 配置文件生成
生成 YOLO 数据集配置文件
生成两阶段训练参数配置
自动路径验证
M3.py — 模型训练
两阶段精细化训练
针对小样本、极相似类别优化
数据增强与损失权重调节
GPU 显存优化、AMP 混合精度训练
M6.py — Web 可视化主程序
基于 Streamlit 的可视化界面
支持图片检测、视频检测
猫爪识别 + 性格自动解读
置信度调节、图像增强、日志展示
检测结果可视化输出
技术栈
目标检测：YOLOv8n
界面框架：Streamlit
图像处理：OpenCV、PIL
深度学习：PyTorch
数据处理：NumPy、Pandas
运行步骤
bash
运行
# 1. 数据预处理
python M1.py

# 2. 生成配置文件
python M2.py

# 3. 训练模型
python M3.py

# 4. 启动 Web 系统
streamlit run M6.py
核心功能
图片猫爪检测
短视频猫爪检测
6 类猫爪精细分类
自动输出性格描述
检测框、置信度、结果表格
图像增强、日志记录、参数可调
环境依赖
plaintext
ultralytics
streamlit
opencv-python
numpy
pillow
pandas
安装：
bash
运行
pip install ultralytics streamlit opencv-python numpy pillow pandas
使用说明
准备猫爪图片与标签
运行 M1 预处理数据
运行 M2 生成配置
运行 M3 训练模型
将模型路径配置到 M6
运行 M6 启动 Web 界面
项目用途
图像识别课程设计
目标检测实训项目
细粒度分类学习
宠物趣味识别应用

本项目仅用于学习、课程设计与实训使用。
