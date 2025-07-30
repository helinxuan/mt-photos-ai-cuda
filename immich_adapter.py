import os
import sys
import asyncio
from typing import Any, Dict, List
import numpy as np
from PIL import Image
from io import BytesIO
import orjson
import cv2
import logging
import torch

# 设置日志
logger = logging.getLogger(__name__)

# 添加immich机器学习模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'machine-learning'))

from immich_ml.models import from_model_type
from immich_ml.models.transforms import decode_pil, serialize_np_array
from immich_ml.schemas import ModelType, ModelTask
from immich_ml.config import settings

class ImmichAdapter:
    """Immich机器学习模块适配器，为mtphotos提供CLIP和人脸识别功能"""
    
    def __init__(self):
        self.clip_visual_model = None
        self.clip_textual_model = None
        self.face_detector = None
        self.face_recognizer = None
        
        # 从环境变量获取模型配置
        self.clip_model_name = os.getenv("CLIP_MODEL_NAME", "XLM-Roberta-Large-Vit-B-16Plus")
        self.face_model_name = os.getenv("FACE_MODEL_NAME", "antelopev2")
        self.face_threshold = float(os.getenv("FACE_THRESHOLD", "0.7"))
        
    def load_clip_visual_model(self):
        """加载CLIP视觉编码器"""
        if self.clip_visual_model is None:
            logger.info(f"Loading visual model '{self.clip_model_name}' to memory")
            self.clip_visual_model = from_model_type(
                self.clip_model_name,
                ModelType.VISUAL,
                ModelTask.SEARCH
            )
            if torch.cuda.is_available():
                logger.info("Setting execution providers to ['CUDAExecutionProvider', 'CPUExecutionProvider'], in descending order of preference")
            else:
                logger.info("Setting execution providers to ['CPUExecutionProvider'], in descending order of preference")
        return self.clip_visual_model
    
    def load_clip_textual_model(self):
        """加载CLIP文本编码器"""
        if self.clip_textual_model is None:
            logger.info(f"Loading textual model '{self.clip_model_name}' to memory")
            self.clip_textual_model = from_model_type(
                self.clip_model_name,
                ModelType.TEXTUAL,
                ModelTask.SEARCH
            )
            if torch.cuda.is_available():
                logger.info("Setting execution providers to ['CUDAExecutionProvider', 'CPUExecutionProvider'], in descending order of preference")
            else:
                logger.info("Setting execution providers to ['CPUExecutionProvider'], in descending order of preference")
        return self.clip_textual_model
    
    def load_face_detector(self):
        """加载人脸检测器"""
        if self.face_detector is None:
            logger.info(f"Loading face detection model '{self.face_model_name}' to memory")
            self.face_detector = from_model_type(
                self.face_model_name,
                ModelType.DETECTION,
                ModelTask.FACIAL_RECOGNITION,
                minScore=self.face_threshold
            )
            if torch.cuda.is_available():
                logger.info("Setting execution providers to ['CUDAExecutionProvider', 'CPUExecutionProvider'], in descending order of preference")
            else:
                logger.info("Setting execution providers to ['CPUExecutionProvider'], in descending order of preference")
        return self.face_detector
    
    def load_face_recognizer(self):
        """加载人脸识别器"""
        if self.face_recognizer is None:
            logger.info(f"Loading face recognition model '{self.face_model_name}' to memory")
            self.face_recognizer = from_model_type(
                self.face_model_name,
                ModelType.RECOGNITION,
                ModelTask.FACIAL_RECOGNITION
            )
            if torch.cuda.is_available():
                logger.info("Setting execution providers to ['CUDAExecutionProvider', 'CPUExecutionProvider'], in descending order of preference")
            else:
                logger.info("Setting execution providers to ['CPUExecutionProvider'], in descending order of preference")
        return self.face_recognizer
    
    def encode_image(self, image_data: bytes) -> List[str]:
        """对图像进行CLIP编码，返回特征向量"""
        try:
            model = self.load_clip_visual_model()
            # 将bytes转换为PIL Image
            image = Image.open(BytesIO(image_data))
            # 使用immich模型进行预测
            result = model.predict(image)
            # 解析序列化的numpy数组
            features = orjson.loads(result)
            # 转换为字符串格式，保持与原API兼容
            return ["{:.16f}".format(float(vec)) for vec in features]
        except Exception as e:
            logger.error(f"CLIP image encoding error: {e}")
            return []
    
    def encode_text(self, text: str) -> List[str]:
        """对文本进行CLIP编码，返回特征向量"""
        try:
            model = self.load_clip_textual_model()
            # 使用immich模型进行预测
            result = model.predict(text)
            # 解析序列化的numpy数组
            features = orjson.loads(result)
            # 转换为字符串格式，保持与原API兼容
            return ["{:.16f}".format(float(vec)) for vec in features]
        except Exception as e:
            logger.error(f"CLIP text encoding error: {e}")
            return []
    
    def detect_faces(self, image_data: bytes) -> Dict[str, Any]:
        """检测人脸，返回检测结果"""
        try:
            detector = self.load_face_detector()
            recognizer = self.load_face_recognizer()
            
            # 将bytes转换为numpy数组
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 人脸检测 - 使用predict方法而不是_predict
            detection_result = detector.predict(img)
            
            # 如果检测到人脸，进行人脸识别
            if detection_result["boxes"].shape[0] > 0:
                recognition_result = recognizer.predict(img, detection_result)
                
                # 转换为MT-Photos兼容格式，确保所有numpy类型都转换为Python原生类型
                faces = []
                for i, face_data in enumerate(recognition_result):
                    # 获取边界框
                    box = detection_result["boxes"][i]
                    landmarks = detection_result["landmarks"][i] if "landmarks" in detection_result else None
                    
                    face_obj = {
                        "embedding": face_data.tolist() if hasattr(face_data, 'tolist') else list(face_data),
                        "facial_area": {
                            "x": int(box[0]),
                            "y": int(box[1]), 
                            "w": int(box[2] - box[0]),
                            "h": int(box[3] - box[1]),
                            "left_eye": [int(landmarks[0]), int(landmarks[1])] if landmarks is not None else None,
                            "right_eye": [int(landmarks[2]), int(landmarks[3])] if landmarks is not None else None
                        },
                        "face_confidence": float(detection_result["scores"][i])
                    }
                    faces.append(face_obj)
                
                return {
                    "faces": faces,
                    "detection": {
                        "boxes": detection_result["boxes"].tolist(),
                        "scores": detection_result["scores"].tolist(),
                        "landmarks": detection_result["landmarks"].tolist() if "landmarks" in detection_result else []
                    }
                }
            else:
                return {
                    "faces": [],
                    "detection": {
                        "boxes": [],
                        "scores": [],
                        "landmarks": []
                    }
                }
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {"faces": [], "detection": {"boxes": [], "scores": [], "landmarks": []}}

# 全局适配器实例
immich_adapter = ImmichAdapter()