import os
import yaml


def SHUJV_p():
    """创建YOLOv8n数据配置文件 """

    base_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别"
    datasets_path = os.path.join(base_path, "datasets")

    # 检查数据集是否存在
    if not os.path.exists(datasets_path):
        print(f"错误: 数据集路径 {datasets_path} 不存在!")
        return None

    data = {
        'path': datasets_path,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 6,  # 类别数量
        'names': {
            0: 'ANXIN',
            1: 'FUSHISHAN',
            2: 'MILI',
            3: 'FANTUAN',
            4: 'HUOJIAN',
            5: 'SANYECAO'
        }
    }

    yaml_path = os.path.join(base_path, "cat_paw_data.yaml")

    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"数据集配置文件已创建: {yaml_path}")

        # 显示配置内容
        print("\n配置文件内容:")
        print("-" * 30)
        for key, value in data.items():
            if key == 'names':
                print(f"  {key}:")
                for cls_id, cls_name in value.items():
                    print(f"    {cls_id}: {cls_name}")
            else:
                print(f"  {key}: {value}")

        # 验证文件是否可读
        with open(yaml_path, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)
            print(f"\n配置文件验证成功，包含 {loaded_data['nc']} 个类别")

        return yaml_path

    except Exception as e:
        print(f"创建配置文件失败: {e}")
        return None


def train_p():
    """创建训练配置文件"""
    base_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别"

    training_config = {
        'model_type': 'yolov8n',
        'description': '极相似猫爪类别识别专用配置',
        'training_strategy': '两阶段训练',
        'phase1': {
            'epochs': 50,
            'imgsz': 416,
            'batch_size': 8,
            'learning_rate': 0.001,
            'augmentation': '温和增强'
        },
        'phase2': {
            'epochs': 100,
            'imgsz': 512,
            'batch_size': 4,
            'learning_rate': 0.0001,
            'augmentation': '精细增强'
        }
    }

    config_path = os.path.join(base_path, "training_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True)

    print(f"训练配置文件已创建: {config_path}")
    return config_path


if __name__ == "__main__":
    print("=== 创建配置文件 ===")
    yaml_path = SHUJV_p()
    if yaml_path:
        config_path = train_p()
        print(f"\n 所有配置文件创建完成!")
        print(f"   数据配置: {yaml_path}")
        print(f"   训练配置: {config_path}")