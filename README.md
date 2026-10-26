 国内场景无人机行人检测数据集
7417张数据集
训练集：5223 验证集：1512 测试集：744
类别数量：1个对象类别 ：行人
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c9d0d592dbda4d4c9f756870c9421b5a.png)

类别类型：['无人机视角下的行人]
文件格式：JPG
标签格式：txt（需要COCO、VOC、Tensoorflow的CSV等格式请私信，否则默认发TXT）
图片大小：2000 x 1500



| **项目**                | **内容**                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **数据集名称**          | 国内场景无人机人员目标检测数据集                                           |
| **总图像数量**          | 7,417 张                                                                 |
| **图像分辨率**          | 2000 × 1500 像素（宽 × 高）                                               |
| **文件格式**            | JPG                                                                      |
| **目标类别数量**        | 1 类                                                                     |
| **类别名称**            | `无人机视角下的行人`（即：行人 / person，从无人机俯视角度拍摄）             |
| **标签格式（默认）**    | TXT（YOLO 格式：`class_id center_x center_y width height`，归一化坐标）     |
| **其他格式支持**        | COCO JSON、VOC XML、TensorFlow CSV（需私信申请，不默认提供）               |
| **数据划分**            |                                                                          |
| &nbsp;&nbsp;– 训练集   | 5,223 张（约占 70.4%）                                                   |
| &nbsp;&nbsp;– 验证集   | 1,512 张（约占 20.4%）                                                   |
| &nbsp;&nbsp;– 测试集   | 744 张（约占 10.0%）                                                     |
| **应用场景**            | 无人机巡检、人群密度估计、应急搜救、智慧城市、交通监控、大型活动安防等       |
| **视角特点**            | 高空俯拍、小目标密集、遮挡常见、光照与背景复杂（城市/乡村/道路/广场等国内典型场景） |

> ✅ **备注**：
> - 所有标注均为人工精细标注，确保边界框紧贴行人目标；
> - 适用于 YOLOv5/v8、Faster R-CNN、RT-DETR 等主流目标检测框架；
> - 因图像分辨率高（3MP），适合训练对小目标敏感的检测模型。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a052c26d986249f6a179fd87ab49f7c1.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d1c2c0040204a0f9c99a2547a92e675.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1cf36ad48b164ab1ba4a7e53d7406c7e.png)
# 🚁 无人机视角行人检测系统（YOLOv8 目标检测）  
**数据集说明**：7417 张高分辨率（2000×1500）JPG 图像，1 类目标 —— **“无人机视角下的行人”**  
✅ 提供完整 **YOLOv8 训练 + 推理代码**  
✅ 支持 YOLO 格式 TXT 标签（默认）  
✅ 适用于小目标、密集人群、复杂背景场景  

---

## 📁 一、项目结构

```
drone_person_yolov8/
├── datasets/
│   └── drone_person/
│       ├── images/
│       │   ├── train/      # 5223 张
│       │   ├── val/        # 1512 张
│       │   └── test/       # 744 张
│       └── labels/
│           ├── train/      # 对应 .txt 标签
│           ├── val/
│           └── test/
├── data.yaml               # 数据配置文件
├── train.py                # YOLOv8 训练脚本
├── detect.py               # 推理脚本（图片/视频/摄像头）
└── requirements.txt
```

> 💡 假设你已将图像和标签按上述结构组织好（YOLO格式：`class_id x_center y_center w h`，归一化）

---

## 🔧 二、环境安装

```bash
# 创建虚拟环境
conda create -n yolov8_drone python=3.9 -y
conda activate yolov8_drone

# 安装官方 YOLOv8
pip install ultralytics opencv-python numpy matplotlib tqdm
```

---

## 📄 三、创建 `data.yaml`

```yaml
# data.yaml
path: ./datasets/drone_person
train: images/train
val: images/val
test: images/test

nc: 1
names: ['person']
```

> ✅ 注意：
> - `class_id` 在标签中必须为 `0`
> - 所有坐标已归一化到 [0, 1]

---

## 🚀 四、YOLOv8 训练代码（`train.py`）

```python
# train.py
from ultralytics import YOLO

# 加载预训练模型（推荐 yolov8n 或 yolov8s 平衡速度与精度）
model = YOLO('yolov8n.pt')  # 可替换为 'yolov8s.pt' 提升精度

# 开始训练
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,              # 输入尺寸（YOLOv8 自动缩放，不影响原始高分辨率）
    batch=16,               # 根据 GPU 显存调整（RTX 3060 建议 16）
    name='drone_person_yolov8n',
    device=0,               # 使用 GPU 0；设为 'cpu' 则用 CPU
    patience=20,            # 早停：20 轮无提升则停止
    save=True,
    plots=True,
    
    # 数据增强（针对无人机小目标优化）
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,           # 随机旋转（模拟无人机姿态变化）
    translate=0.1,
    scale=0.5,
    mosaic=1.0,             # Mosaic 增强（提升小目标检测能力）
    flipud=0.0,             # 不上下翻转（无人机视角通常无需）
    fliplr=0.5              # 左右翻转（合理）
)

print("✅ 训练完成！")
print("最佳权重路径: runs/detect/drone_person_yolov8n/weights/best.pt")
```

> ⚠️ **重要提示**：  
> 虽然原始图像是 **2000×1500**，但 YOLOv8 在训练时会自动 resize 到 `imgsz=640`。  
> 若需更好检测小目标，可尝试：
> ```python
> imgsz=1280  # 更大输入尺寸（显存允许前提下）
> ```

---

## 🔍 五、推理检测（`detect.py`）

```python
# detect.py
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/detect/drone_person_yolov8n/weights/best.pt')

# 1. 检测单张高分辨率图像
def detect_image(image_path, output_path="result.jpg"):
    results = model(image_path, imgsz=1280)  # 推理时用更大尺寸提升小目标召回
    annotated = results[0].plot()
    cv2.imwrite(output_path, annotated)
    print(f"✅ 检测结果已保存至 {output_path}")

# 2. 检测视频
def detect_video(video_path, output_path="output.mp4"):
    model.predict(
        source=video_path,
        save=True,
        project="runs/predict",
        name="video_result",
        imgsz=1280
    )
    print(f"✅ 视频结果保存在 runs/predict/video_result/")

# 3. 实时摄像头（或 RTSP 流）
def detect_stream(stream_url=0):  # 0 表示本地摄像头
    model.predict(
        source=stream_url,
        show=True,
        imgsz=640,
        conf=0.4  # 置信度阈值（可调）
    )

if __name__ == "__main__":
    # detect_image("test_drone.jpg")
    # detect_video("drone_footage.mp4")
    detect_stream()
```

---

## 📊 六、预期性能（基于 7417 张数据）

| 指标 | 预期值 | 说明 |
|------|--------|------|
| **mAP@0.5** | ≥ 0.85 | 高密度人群下仍保持良好定位 |
| **Recall** | ≥ 0.80 | 小目标检出率高 |
| **推理速度** | ~45 FPS（RTX 3060, imgsz=640） | 满足实时无人机回传分析 |
| **适用模型** | YOLOv8n / YOLOv8s 最佳 | YOLOv8m/l 可用于离线高精度场景 |

---

## 🛠️ 七、针对无人机场景的优化建议

1. **使用更大的输入尺寸**（如 `imgsz=1280`）以保留小目标细节；
2. **开启 Mosaic 增强**：让模型学习多图拼接中的小目标；
3. **降低置信度阈值**（推理时 `conf=0.3~0.4`）提升召回；
4. **后处理优化**：对密集区域使用 Soft-NMS 替代标准 NMS（可通过 `ultralytics` 的 `agnostic_nms=False` 控制）；
5. **TTA（Test-Time Augmentation）**：
   ```python
   results = model(image_path, augment=True)  # 提升精度（速度变慢）
   ```

---



> 🛰️ **总结**：本系统专为 **高空无人机行人检测** 设计，适用于应急搜救、大型活动监控、智慧城市等场景。利用 YOLOv8 的高效架构，在保持实时性的同时，有效应对小目标、遮挡、密集人群等挑战。
