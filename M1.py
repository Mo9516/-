import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import yaml


def prepare_data():
    """数据准备和整理"""

    # 定义路径
    base_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别"
    image_source_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别\zonghe1"
    label_source_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别\ZONGHE"

    # 创建YOLOv8要求的目录结构
    datasets_path = os.path.join(base_path, "datasets")
    images_path = os.path.join(datasets_path, "images")
    labels_path = os.path.join(datasets_path, "labels")

    for path in [images_path, labels_path]:
        os.makedirs(os.path.join(path, "train"), exist_ok=True)
        os.makedirs(os.path.join(path, "val"), exist_ok=True)

    print("目录结构创建完成！")

    # 检查数据
    def check_and_organize_data():
        image_files = glob.glob(os.path.join(image_source_path, "*.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"找到 {len(image_files)} 张图片")

        missing_labels = []
        valid_pairs = []

        for img_path in image_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_source_path, f"{img_name}.txt")

            if os.path.exists(label_path):
                valid_pairs.append((img_path, label_path))
            else:
                missing_labels.append(img_name)

        print(f"有效图片-标签对: {len(valid_pairs)}")
        print(f"缺失标签的图片: {len(missing_labels)}")

        if missing_labels:
            print("前5个缺失标签的图片:", missing_labels[:5])

        return valid_pairs

    valid_pairs = check_and_organize_data()

    # 划分数据集
    image_paths = [pair[0] for pair in valid_pairs]
    label_paths = [pair[1] for pair in valid_pairs]

    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42, stratify=None
    )

    print(f"训练集: {len(train_images)} 张图片")
    print(f"验证集: {len(val_images)} 张图片")

    # 复制文件
    def copy_files():
        # 复制训练集
        for img_path, label_path in zip(train_images, train_labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)

            shutil.copy2(img_path, os.path.join(images_path, "train", img_name))
            shutil.copy2(label_path, os.path.join(labels_path, "train", label_name))

        # 复制验证集
        for img_path, label_path in zip(val_images, val_labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)

            shutil.copy2(img_path, os.path.join(images_path, "val", img_name))
            shutil.copy2(label_path, os.path.join(labels_path, "val", label_name))

        print("文件复制完成！")

    copy_files()

    # 分析类别分布
    analyze_class_distribution()

    print("数据准备完成！")
    return len(train_images), len(val_images)


def analyze_class_distribution():
    """分析训练集类别分布"""
    base_path = r"F:\OPENCV\实训作业\zhuayinshibie\猫爪识别"
    labels_train_path = os.path.join(base_path, "datasets", "labels", "train")

    if not os.path.exists(labels_train_path):
        print("警告: 训练标签路径不存在")
        return

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_instances = 0

    for label_file in os.listdir(labels_train_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_train_path, label_file), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                            total_instances += 1
                        except (ValueError, IndexError):
                            continue

    class_names = ['ANXIN', 'FUSHISHAN', 'MILI', 'FANTUAN', 'HUOJIAN', 'SANYECAO']

    print("\n" + "=" * 60)
    print("训练集类别分布详细分析:")
    print("=" * 60)
    for class_id in range(6):
        count = class_counts[class_id]
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"  {class_names[class_id]}: {count:3d} 个标注 ({percentage:5.1f}%)")

    print(f"\n总标注实例数: {total_instances}")

    # 识别少数类别
    minority_classes = []
    for class_id in range(6):
        if class_counts[class_id] < 10:  # 少于10个样本视为极少数
            minority_classes.append(class_names[class_id])

    if minority_classes:
        print(f"需要特别关注的极少数类别: {', '.join(minority_classes)}")

    return class_counts


if __name__ == "__main__":
    train_count, val_count = prepare_data()
    print(f"最终数据统计: 训练集 {train_count} 张, 验证集 {val_count} 张")