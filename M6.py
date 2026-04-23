import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

# 设置页面
st.set_page_config(
    page_title="🐱 猫咪爪印性格识别系统",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 解决中文显示问题
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    * {
        font-family: 'Noto Sans SC', sans-serif;
    }
    .stInfo {
        font-family: 'Noto Sans SC', sans-serif;
    }
    .stMarkdown, .stText, .stHeader, .stSubheader {
        font-family: 'Noto Sans SC', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.title("🐱 猫咪爪印性格识别系统")
st.markdown("---")

# 初始化session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_video' not in st.session_state:
    st.session_state.current_video = None
if 'detection_logs' not in st.session_state:
    st.session_state.detection_logs = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = 'image'
if 'run_image_detection' not in st.session_state:
    st.session_state.run_image_detection = False
if 'run_video_detection' not in st.session_state:
    st.session_state.run_video_detection = False

# 类别名称
CLASS_NAMES = {
    0: 'ANXIN(爱心)',
    1: 'FUSHISHAN(富士山)',
    2: 'MILI(米粒)',
    3: 'FANTUAN(饭团)',
    4: 'HUOJIAN(火箭)',
    5: 'SANYECAO(三叶草)'
}

# 猫爪形状介绍
PAW_DESCRIPTIONS = {
    'ANXIN(爱心)': {
        'name': '爱心型猫爪',
        'description': '你的猫咪属于爱心型猫爪：表现性格为温和，亲近人，喜欢撒娇比较粘人，对陌生人友好并且也能和其他宠物友好相处，比较需要主人的时间陪伴。',
        'emoji': '❤️'
    },
    'FUSHISHAN(富士山)': {
        'name': '富士山型猫爪',
        'description': '你的猫咪属于富士山型猫爪：表现性格为傲娇，比较高冷非常有个性，不太粘人，性格沉稳，无论是对陌生人还是主人都不会表示特别亲近。',
        'emoji': '🗻'
    },
    'FANTUAN(饭团)': {
        'name': '饭团型猫爪',
        'description': '你的猫咪属于饭团形猫爪：表现性格为活泼，比较爱打架，争强好胜，内心充满野心，是猫界中的"正义使者"，对主人忠心耿耿且领地意识很强。',
        'emoji': '🍙'
    },
    'MILI(米粒)': {
        'name': '米粒型猫爪',
        'description': '你的猫咪属于米粒型猫爪：表现性格为温柔且胆小，感情细腻会在主人不开心时陪伴主人，比较稳重依赖主人，比较容易相信人类，但对其他动物会有警惕心且容易受到外界惊吓。',
        'emoji': '🍚'
    },
    'HUOJIAN(火箭)': {
        'name': '火箭型猫爪',
        'description': '你的猫咪属于火箭型猫爪：表现性格为好奇心旺盛、探索欲强，喜欢扒拉东西，对新鲜事物有浓厚的兴趣，比较我行我素。',
        'emoji': '🚀'
    },
    'SANYECAO(三叶草)': {
        'name': '三叶草型猫爪',
        'description': '你的猫咪属于三叶草型猫爪：表现性格为比较外向，智商高，喜欢蹭人也喜欢跟人互动，感情比较丰富，还特别爱撒娇，喜欢吃醋。',
        'emoji': '☘️'
    }
}


# 工具函数
def log_message(message):
    """添加日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    st.session_state.detection_logs.append(formatted_message)
    if len(st.session_state.detection_logs) > 50:
        st.session_state.detection_logs.pop(0)


def load_model():
    """加载模型"""
    try:
        if st.session_state.model is None:
            with st.spinner("正在加载模型..."):
                # 模型路径 - 请根据实际情况修改
                model_path = r'F:\OPENCV\实训作业\zhuayinshibie\猫爪识别\phase2_fine_grained_amp_640\weights\best.pt'

                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    st.error(f"模型文件不存在，请检查路径: {model_path}")
                    log_message(f"❌ 模型文件不存在: {model_path}")
                    return False

                st.session_state.model = YOLO(model_path)
                log_message(f"✅ 模型加载成功！")
                return True
        return True
    except Exception as e:
        log_message(f"❌ 模型加载失败: {str(e)}")
        st.error(f"模型加载失败: {str(e)}")
        return False


def check_video_duration(file_path):
    """检查视频时长"""
    try:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        if duration < 3:
            return False, f"视频时长({duration:.1f}秒)太短，请上传3-5秒的视频"
        elif duration > 5:
            return False, f"视频时长({duration:.1f}秒)太长，请上传3-5秒的视频"
        else:
            return True, f"视频时长({duration:.1f}秒)符合要求"
    except Exception as e:
        return False, f"视频检查错误: {str(e)}"


def process_detection_results(image, results):
    """处理检测结果"""
    processed_image = image.copy()
    detection_info = []

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = CLASS_NAMES.get(cls, f'Class_{cls}')

                # 绘制检测框
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 绘制标签 - 使用支持中文的字体
                label = f"{class_name} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                cv2.rectangle(processed_image, (x1, y1 - text_height - 5),
                              (x1 + text_width, y1), (0, 255, 0), -1)

                cv2.putText(processed_image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detection_info.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

    return processed_image, detection_info


def detect_image(image, confidence, use_enhancement):
    """图片检测"""
    if st.session_state.model is None:
        st.error("模型未加载，请先加载模型")
        return None, None

    try:
        # 图像预处理
        if use_enhancement:
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)

        # 进行检测
        results = st.session_state.model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            device='cpu',
            verbose=False
        )

        # 处理结果
        processed_image, detection_info = process_detection_results(image, results)

        if detection_info:
            log_message(f"✅ 检测到 {len(detection_info)} 个猫爪")
        else:
            log_message("❌ 未检测到猫爪")

        return processed_image, detection_info

    except Exception as e:
        error_msg = f"❌ 图片检测失败: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)
        return None, None


def process_video(video_path, confidence, use_enhancement):
    """处理视频检测"""
    if st.session_state.model is None:
        st.error("模型未加载，请先加载模型")
        return None, None

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        best_detection_frame = None
        best_detection_info = None
        best_confidence = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"处理中: {frame_count}/{total_frames} 帧")

            # 图像预处理
            if use_enhancement:
                frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=20)
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                frame = cv2.filter2D(frame, -1, kernel)

            # 调整尺寸
            detection_frame = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

            # 进行检测
            results = st.session_state.model.predict(
                source=rgb_frame,
                conf=confidence,
                imgsz=640,
                device='cpu',
                verbose=False
            )

            # 处理检测结果
            detections = []
            current_max_confidence = 0

            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # 调整坐标
                        scale_x = frame.shape[1] / 640
                        scale_y = frame.shape[0] / 640
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': cls
                        })
                        current_max_confidence = max(current_max_confidence, float(conf))

            # 更新最佳检测结果
            if detections and current_max_confidence > best_confidence:
                best_confidence = current_max_confidence

                # 绘制检测结果
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    class_id = detection['class_id']
                    class_name = CLASS_NAMES.get(class_id, f'Class_{class_id}')

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    label = f"{class_name} {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )

                    cv2.rectangle(frame, (x1, y1 - text_height - 10),
                                  (x1 + text_width, y1), (0, 255, 0), -1)

                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                best_detection_frame = frame.copy()
                best_detection_info = detections.copy()

        cap.release()
        progress_bar.empty()
        status_text.empty()

        if best_detection_frame is not None:
            log_message(f"✅ 视频检测完成，最佳检测置信度: {best_confidence:.4f}")
        else:
            log_message("❌ 视频检测完成，但未检测到猫爪")

        return best_detection_frame, best_detection_info

    except Exception as e:
        error_msg = f"❌ 视频检测失败: {str(e)}"
        log_message(error_msg)
        st.error(error_msg)
        return None, None


def display_paw_description(class_name):
    """显示猫爪形状的性格介绍"""
    if class_name in PAW_DESCRIPTIONS:
        description = PAW_DESCRIPTIONS[class_name]
        with st.expander(f"{description['emoji']} {description['name']} 性格介绍", expanded=True):
            st.markdown(f"**{description['name']}**")
            st.info(description['description'])


# 侧边栏 - 控制面板
with st.sidebar:
    st.header("🔧 控制面板")

    # 模型状态
    st.subheader("模型信息")
    if st.button("加载模型", use_container_width=True):
        if load_model():
            st.success("模型加载成功!")

    # 检测设置
    st.subheader("⚙️ 检测设置")
    confidence = st.slider("置信度阈值", 0.05, 1.0, 0.1, 0.05)
    use_enhancement = st.checkbox("启用图像增强", value=True)

    st.markdown("---")

    # 操作按钮
    st.subheader("💾 结果操作")
    if st.button("清空结果", use_container_width=True):
        st.session_state.current_image = None
        st.session_state.current_video = None
        st.session_state.detection_logs = []
        st.success("结果已清空!")

    # 系统信息
    st.markdown("---")
    st.subheader("📊 系统信息")
    st.markdown("""
    - **识别类别**: 爱心、富士山、米粒、饭团、火箭、三叶草
    - **视频要求**: 3-5秒，猫爪清晰可见
    - **支持格式**: JPG、PNG、MP4、AVI
    """)

# 主内容区域 - 分栏布局
col1, col2 = st.columns([1, 3])

with col1:
    st.header("🔍 检测模式")

    # 检测模式选择
    mode = st.radio(
        "选择检测模式:",
        ["🖼️ 图片检测", "🎥 视频检测"],
        index=0 if st.session_state.current_mode == 'image' else 1
    )

    # 更新当前模式
    if "图片检测" in mode:
        st.session_state.current_mode = 'image'
    else:
        st.session_state.current_mode = 'video'

    st.markdown("---")

    # 显示当前模式的上传组件
    if st.session_state.current_mode == 'image':
        st.subheader("🖼️ 图片上传")
        uploaded_image = st.file_uploader(
            "选择图片文件（支持JPG、PNG等格式）",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_uploader"
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.session_state.current_image = np.array(image)
            st.image(image, caption="上传的图片", use_column_width=True)

        if st.session_state.current_image is not None:
            if st.button("开始图片检测", use_container_width=True, type="primary"):
                if st.session_state.model is None:
                    st.error("请先加载模型！")
                else:
                    st.session_state.run_image_detection = True

    else:  # video mode
        st.subheader("🎥 视频上传")
        uploaded_video = st.file_uploader(
            "选择视频文件（3-5秒，支持MP4、AVI等格式）",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )

        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            duration_ok, duration_msg = check_video_duration(video_path)
            if not duration_ok:
                st.warning(duration_msg)
            else:
                st.success(duration_msg)

            st.video(uploaded_video)
            st.session_state.current_video = video_path

        if st.session_state.current_video is not None:
            if st.button("开始视频检测", use_container_width=True, type="primary"):
                if st.session_state.model is None:
                    st.error("请先加载模型！")
                else:
                    st.session_state.run_video_detection = True

with col2:
    if st.session_state.current_mode == 'image':
        st.header("🖼️ 图片检测结果")

        if st.session_state.run_image_detection:
            if st.session_state.current_image is not None and st.session_state.model is not None:
                with st.spinner("图片检测中..."):
                    image_cv = st.session_state.current_image
                    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

                    processed_image, detection_info = detect_image(
                        image_cv,
                        confidence,
                        use_enhancement
                    )

                    if processed_image is not None:
                        result_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(result_image, caption="检测结果", use_column_width=True)

                        if detection_info:
                            st.subheader("📊 检测结果")
                            df = pd.DataFrame(detection_info)
                            st.dataframe(df, use_container_width=True)

                            # 显示猫爪形状介绍
                            st.subheader("🐾 猫爪形状性格分析")
                            detected_classes = set(df['class'].tolist())
                            for class_name in detected_classes:
                                display_paw_description(class_name)

                            class_counts = df['class'].value_counts()
                            st.bar_chart(class_counts)
                        else:
                            st.warning("❌ 未检测到猫爪")

                st.session_state.run_image_detection = False
        else:
            st.info("👈 请在左侧上传图片并点击'开始图片检测'按钮")

    else:  # video mode
        st.header("🎥 视频检测结果")

        if st.session_state.run_video_detection:
            if st.session_state.current_video is not None and st.session_state.model is not None:
                with st.spinner("视频检测中..."):
                    best_frame, detection_info = process_video(
                        st.session_state.current_video,
                        confidence,
                        use_enhancement
                    )

                    if best_frame is not None:
                        result_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
                        st.image(result_frame, caption="最佳检测结果", use_column_width=True)

                        if detection_info:
                            st.subheader("📊 检测结果")
                            detection_df = pd.DataFrame([
                                {
                                    'class': CLASS_NAMES.get(det['class_id'], f'Class_{det["class_id"]}'),
                                    'confidence': det['confidence'],
                                    'bbox': det['bbox']
                                } for det in detection_info
                            ])
                            st.dataframe(detection_df, use_container_width=True)

                            # 显示猫爪形状介绍
                            st.subheader("🐾 猫爪形状性格分析")
                            detected_classes = set(
                                [CLASS_NAMES.get(det['class_id'], f'Class_{det["class_id"]}') for det in
                                 detection_info])
                            for class_name in detected_classes:
                                display_paw_description(class_name)

                            max_conf = max(det['confidence'] for det in detection_info)
                            st.success(f"最高检测置信度: {max_conf:.4f}")
                        else:
                            st.warning("❌ 未检测到猫爪")

                st.session_state.run_video_detection = False
        else:
            st.info("👈 请在左侧上传视频并点击'开始视频检测'按钮")

# 日志显示
st.markdown("---")
st.header("📋 检测日志")

if st.button("清空日志"):
    st.session_state.detection_logs = []
    st.rerun()

for log in st.session_state.detection_logs:
    st.text(log)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🐱 猫咪爪印性格识别系统 | 基于YOLOv8n开发"
    "</div>",
    unsafe_allow_html=True
)

# 初始加载模型
if st.session_state.model is None:
    load_model()