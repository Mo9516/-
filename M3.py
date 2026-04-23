import os
from ultralytics import YOLO
import torch
import gc
import random
import string


def setup_gpu():
    """GPU设置和优化"""
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将使用CPU训练")
        return 'cpu'

    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    print(f"检测到GPU: {gpu_name}")
    print(f"GPU内存: {gpu_memory:.1f} GB")

    # 针对2GB GPU的优化设置
    if gpu_memory <= 2:
        print("🔧 检测到2GB GPU，应用内存优化设置...")
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 设置较小的CUDA缓存
        torch.cuda.set_per_process_memory_fraction(0.7)

    return 'cuda'


def analyze_data():
    """详细分析训练数据"""
    base_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别"
    labels_train_path = os.path.join(base_path, "datasets", "labels", "train")

    if not os.path.exists(labels_train_path):
        print("错误: 训练标签路径不存在")
        return None

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_files = 0

    for label_file in os.listdir(labels_train_path):
        if label_file.endswith('.txt'):
            total_files += 1
            with open(os.path.join(labels_train_path, label_file), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue

    class_names = ['ANXIN', 'FUSHISHAN', 'MILI', 'FANTUAN', 'HUOJIAN', 'SANYECAO']

    print("\n" + "=" * 70)
    print("训练数据详细分析")
    print("=" * 70)
    print(f"总标签文件数: {total_files}")

    total_instances = sum(class_counts.values())
    print(f"总标注实例数: {total_instances}")

    print("\n各类别分布:")
    print("-" * 50)
    for class_id in range(6):
        count = class_counts[class_id]
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        status = " 少数" if count < 15 else " 充足" if count > 50 else "适中"
        print(f"  {class_names[class_id]}: {count:3d} 实例 ({percentage:5.1f}%) [{status}]")

    # 识别问题类别
    problem_classes = []
    for class_id, count in class_counts.items():
        if count < 10:
            problem_classes.append(class_names[class_id])

    if problem_classes:
        print(f"\n需要重点关注的极少数类别: {', '.join(problem_classes)}")
        print("   建议策略: 大幅提高分类损失权重，延长训练轮次")

    return class_counts


def check_1():
    """
    修补Ultralytics库，避免AMP检查时下载YOLO11n
    强制使用现有模型进行AMP训练
    """
    try:
        from ultralytics.utils import checks

        # 保存原始函数
        original_amp_check = getattr(checks, 'check_amp', None)

        def patched_amp_check(model):
            """修补后的AMP检查函数"""
            print("使用现有YOLOv8n进行AMP训练，跳过YOLO11n下载检查")
            return True  # 直接返回True，允许AMP训练

        # 应用补丁
        if original_amp_check:
            checks.check_amp = patched_amp_check
            print("已应用AMP检查补丁")
        else:
            print("无法找到AMP检查函数，将继续正常训练")

    except Exception as e:
        print(f"AMP补丁应用失败: {e}")
        print("训练将继续，但可能会尝试下载YOLO11n")


def train_1(data_yaml_path, device):
    """第一阶段启用AMP版本，使用大尺寸和批次"""
    print("\n" + "=" * 60)
    print("阶段1: 基础特征学习 (启用AMP, imgsz=640, batch=10)")
    print("=" * 60)

    # 确保yolov8n.pt在本地
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        print(f" 错误: {model_path} 不在当前目录")
        print("请确保 yolov8n.pt 文件存在")
        return None

    print(f"使用本地模型: {model_path}")
    model = YOLO(model_path)

    phase1_config = {
        'data': data_yaml_path,
        'epochs': 50,
        'imgsz': 512,  # 使用大尺寸图像
        'batch': 10,  # 使用大批次
        'device': device,
        'workers': 2,
        'patience': 20,
        'save': True,
        'exist_ok': True,
        'project': r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别",
        'name': 'phase1_baseline_amp_640',

        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,

        # 损失权重 - 适度关注分类
        'box': 6.0,
        'cls': 1.8,  # 适度增加分类权重
        'dfl': 1.5,

        # 温和的数据增强（适合极相似类别）
        'augment': True,
        'mosaic': 0.3,  # 降低mosaic概率
        'mixup': 0.0,  # 关闭mixup，避免类别混淆
        'copy_paste': 0.0,  # 关闭复制粘贴
        'fliplr': 0.3,  # 适度水平翻转
        'flipud': 0.0,  # 关闭垂直翻转
        'degrees': 5.0,  # 减小旋转角度
        'translate': 0.05,  # 减小平移
        'scale': 0.1,  # 减小缩放
        'shear': 2.0,  # 减小剪切
        'perspective': 0.0,  # 关闭透视变换

        # 颜色增强（对相似类别很重要）
        'hsv_h': 0.01,  # 减小色调变化
        'hsv_s': 0.5,  # 适度饱和度增强
        'hsv_v': 0.3,  # 适度亮度增强

        # 启用AMP但使用补丁避免下载
        'amp': True,

        # 其他优化
        'close_mosaic': 10,
        'dropout': 0.15,
        'overlap_mask': True,
    }

    print("阶段1配置 (启用AMP, imgsz=640, batch=10):")
    for key, value in {k: v for k, v in phase1_config.items() if k not in ['data']}.items():
        print(f"  {key}: {value}")

    try:
        # 应用AMP补丁
        check_1()

        results = model.train(**phase1_config)
        print("阶段1训练完成!")
        return model
    except Exception as e:
        print(f"阶段1AMP训练失败: {e}")
        if "out of memory" in str(e):
            print("显存不足，建议减小batch size或图像尺寸")
        print("尝试使用非AMP配置...")
        return train_1_1(data_yaml_path, device)


def train_1_1(data_yaml_path, device):
    """阶段1备选方案：不使用AMP，但保持大尺寸和批次"""
    print("\n切换到非AMP训练模式 (imgsz=640, batch=10)")

    model = YOLO('yolov8n.pt')

    phase1_config = {
        'data': data_yaml_path,
        'epochs': 50,
        'imgsz': 512,  #保持大尺寸
        'batch': 10,  # 保持大批次
        'device': device,
        'workers': 2,
        'patience': 20,
        'save': True,
        'exist_ok': True,
        'project': r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别",
        'name': 'phase1_baseline_no_amp_640',

        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,

        # 损失权重
        'box': 6.0,
        'cls': 1.8,
        'dfl': 1.5,

        # 数据增强配置
        'augment': True,
        'mosaic': 0.3,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'fliplr': 0.3,
        'flipud': 0.0,
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.1,
        'shear': 2.0,
        'perspective': 0.0,
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,

        # 明确禁用AMP
        'amp': False,

        # 其他配置
        'close_mosaic': 10,
        'dropout': 0.15,
        'overlap_mask': True,
    }

    try:
        results = model.train(**phase1_config)
        print("阶段1非AMP训练完成!")
        return model
    except Exception as e:
        print(f"阶段1非AMP训练也失败: {e}")
        if "out of memory" in str(e):
            print(" 显存不足，建议减小配置:")

        return None


def train_2(data_yaml_path, device, phase1_model_path):
    """第二阶段：精细分类训练 - 启用AMP版本，使用大尺寸"""
    print("\n" + "=" * 60)
    print("阶段2: 精细分类训练 (启用AMP, imgsz=640)")
    print("=" * 60)

    # 从阶段1的最佳模型继续训练
    if os.path.exists(phase1_model_path):
        model = YOLO(phase1_model_path)
        print(f"从阶段1模型继续训练: {phase1_model_path}")
    else:
        model = YOLO('yolov8n.pt')
        print("阶段1模型不存在，使用预训练模型")

    phase2_config = {
        'data': data_yaml_path,
        'epochs': 100,
        'imgsz': 512,  # 使用大尺寸图像
        'batch': 8,  # 阶段2使用稍小的批次
        'device': device,
        'workers': 1,
        'patience': 30,  # 更多耐心
        'save': True,
        'exist_ok': True,
        'project': r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别",
        'name': 'phase2_fine_grained_amp_640',

        # 优化器配置 - 更小的学习率
        'optimizer': 'AdamW',
        'lr0': 0.0001,  # 更小的学习率
        'lrf': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0001,

        # 关键   大幅提高分类损失权重
        'box': 4.0,#检测框损失权重
        'cls': 3.5,  # 大幅增加分类权重 强化少数类识别
        'dfl': 1.2,

        # 精细的数据增强
        'augment': True,
        'mosaic': 0.1,  # 进一步降低mosaic
        'mixup': 0.0,
        'copy_paste': 0.0,
        'fliplr': 0.1,  # 减少翻转
        'flipud': 0.0,
        'degrees': 2.0,  # 进一步减小旋转
        'translate': 0.02,
        'scale': 0.05,
        'shear': 1.0,
        'perspective': 0.0,

        # 增强颜色和纹理区分
        'hsv_h': 0.005,  # 更小的色调变化
        'hsv_s': 0.6,  # 增强饱和度（帮助颜色区分）
        'hsv_v': 0.4,  # 增强亮度（帮助纹理区分）

        # 启用AMP
        'amp': True,

        # 其他优化
        'close_mosaic': 5,
        'dropout': 0.25,  # 增加dropout防止过拟合
        'overlap_mask': True,
    }

    print("阶段2配置 (启用AMP, imgsz=640):")
    for key, value in {k: v for k, v in phase2_config.items() if k not in ['data']}.items():
        print(f"  {key}: {value}")

    try:
        # 应用AMP补丁
        check_1()

        results = model.train(**phase2_config)
        print("阶段2训练完成!")
        return model
    except Exception as e:
        print(f"阶段2AMP训练失败: {e}")
        print("尝试使用非AMP配置...")
        return train_2_2(data_yaml_path, device, phase1_model_path)


def train_2_2(data_yaml_path, device, phase1_model_path):
    """阶段2备选方案：不使用AMP，但保持大尺寸"""
    print("\n切换到非AMP训练模式 (imgsz=640)")

    if os.path.exists(phase1_model_path):
        model = YOLO(phase1_model_path)
    else:
        model = YOLO('yolov8n.pt')

    phase2_config = {
        'data': data_yaml_path,
        'epochs': 100,
        'imgsz': 512,  #保持大尺寸
        'batch': 8,  # 非AMP需要更小的批次
        'device': device,
        'workers': 1,
        'patience': 30,
        'save': True,
        'exist_ok': True,
        'project': r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别",
        'name': 'phase2_fine_grained_no_amp_640',

        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.0001,
        'lrf': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0001,

        # 损失权重
        'box': 4.0,
        'cls': 3.5,
        'dfl': 1.2,

        # 数据增强配置
        'augment': True,
        'mosaic': 0.1,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'fliplr': 0.1,
        'flipud': 0.0,
        'degrees': 2.0,
        'translate': 0.02,
        'scale': 0.05,
        'shear': 1.0,
        'perspective': 0.0,
        'hsv_h': 0.005,
        'hsv_s': 0.6,
        'hsv_v': 0.4,

        # 明确禁用AMP
        'amp': False,

        # 其他配置
        'close_mosaic': 5,
        'dropout': 0.25,
        'overlap_mask': True,
    }

    try:
        results = model.train(**phase2_config)
        print("阶段2非AMP训练完成!")
        return model
    except Exception as e:
        print(f"阶段2非AMP训练也失败: {e}")
        return None


def train_xiangsi():
    """完整的极相似类别训练流程 - 启用AMP版本，使用大尺寸和批次"""

    base_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别"
    data_yaml_path = os.path.join(base_path, "cat_paw_data.yaml")

    # 检查配置文件
    if not os.path.exists(data_yaml_path):
        print(f"错误: 配置文件 {data_yaml_path} 不存在!")
        print("请先运行 M2_modified.py 创建配置文件")
        return None

    print("开始极相似猫爪类别识别训练 (启用AMP, imgsz=640, batch=10)")

    # 设置GPU
    device = setup_gpu()

    # 分析数据
    class_counts = analyze_data()
    if not class_counts:
        print("数据分析失败，停止训练")
        return None

    # 检查极少数类别
    class_names = ['ANXIN', 'FUSHISHAN', 'MILI', 'FANTUAN', 'HUOJIAN', 'SANYECAO']
    minority_classes = []
    for class_id, count in class_counts.items():
        if count < 10:
            minority_classes.append(class_names[class_id])

    if minority_classes:
        print(f"\n训练策略调整:")
        print(f"针对极少数类别 {', '.join(minority_classes)} 进行优化")

    # 阶段1训练 - 尝试AMP版本
    phase1_model = train_1(data_yaml_path, device)
    if not phase1_model:
        print("阶段1训练失败，停止流程")
        return None

    # 阶段1模型路径
    phase1_model_path = os.path.join(
        base_path, "phase1_baseline_amp_640", "weights", "best.pt"
    )

    # 如果AMP版本失败，尝试非AMP版本
    if not os.path.exists(phase1_model_path):
        phase1_model_path = os.path.join(
            base_path, "phase1_baseline_no_amp_640", "weights", "best.pt"
        )

    # 阶段2训练 - 尝试AMP版本
    phase2_model = train_2(data_yaml_path, device, phase1_model_path)

    # 清理GPU内存
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    if phase2_model:
        best_model_path = os.path.join(
            base_path, "phase2_fine_grained_amp_640", "weights", "best.pt"
        )
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(
                base_path, "phase2_fine_grained_no_amp_640", "weights", "best.pt"
            )

        if os.path.exists(best_model_path):
            print(f"\n训练完成! 最佳模型保存在:")
            print(f"   {best_model_path}")

            return phase2_model
        else:
            print("未找到最终模型文件")
            return phase1_model  # 返回阶段1模型作为备选
    else:
        print("阶段2训练失败")
        return phase1_model  # 返回阶段1模型作为备选


if __name__ == "__main__":
    trained_model = train_xiangsi()
