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

# 模型配置现在通过immich_ml.config.settings统一管理

# 根据人脸模型名称确定检测器后端
def get_detector_backend(face_model_name):
    """根据人脸模型名称返回对应的检测器后端"""
    if 'antelope' in face_model_name.lower():
        return 'retinaface'
    elif 'buffalo' in face_model_name.lower():
        return 'retinaface'
    else:
        return 'retinaface'  # 默认使用retinaface

class ImmichAdapter:
    """Immich机器学习模块适配器，为mtphotos提供CLIP和人脸识别功能"""
    
    def __init__(self):
        self.clip_visual_model = None
        self.clip_textual_model = None
        self.face_detector = None
        self.face_recognizer = None
        
        # 从config.py的settings获取模型配置
        self.clip_model_name = settings.clip_model_name
        self.face_model_name = settings.face_model_name
        self.face_threshold = settings.face_threshold
        
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
            logger.info(f"Detection result: boxes shape={detection_result['boxes'].shape}, scores shape={detection_result['scores'].shape}")
            logger.info(f"Detection result types: boxes type={type(detection_result['boxes'])}, scores type={type(detection_result['scores'])}")
            
            # 如果检测到人脸，进行人脸识别
            if detection_result["boxes"].shape[0] > 0:
                recognition_result = recognizer.predict(img, detection_result)
                logger.info(f"Recognition result type: {type(recognition_result)}, length: {len(recognition_result)}")
                if len(recognition_result) > 0:
                    logger.info(f"First recognition result: {type(recognition_result[0])}, keys: {recognition_result[0].keys() if isinstance(recognition_result[0], dict) else 'not dict'}")
                
                # 转换为MT-Photos兼容格式，确保所有numpy类型都转换为Python原生类型
                faces = []
                for i, face_data in enumerate(recognition_result):
                    logger.info(f"Processing face {i}: type={type(face_data)}, data={face_data}")
                    # recognition_result返回的是DetectedFace字典格式
                    # 需要从embedding字符串中解析出numpy数组
                    try:
                        # face_data是字典，包含boundingBox, embedding(字符串), score
                        embedding_str = face_data["embedding"]
                        # 解析embedding字符串为numpy数组，然后转为列表
                        embedding_array = orjson.loads(embedding_str)
                        embedding_list = embedding_array.tolist() if hasattr(embedding_array, 'tolist') else list(embedding_array)
                        
                        # 使用boundingBox信息
                        bbox = face_data["boundingBox"]
                        logger.info(f"Bbox for face {i}: type={type(bbox)}, data={bbox}")
                        
                        # 获取landmarks信息（眼部坐标）
                        landmarks = detection_result["landmarks"][i] if "landmarks" in detection_result and i < len(detection_result["landmarks"]) else None
                        left_eye = None
                        right_eye = None
                        if landmarks is not None and len(landmarks) >= 4:
                            # landmarks格式通常是 [left_eye_x, left_eye_y, right_eye_x, right_eye_y, ...]
                            # 需要先转换numpy数组元素为Python标量
                            logger.info(f"Landmarks for face {i}: type={type(landmarks)}, shape={landmarks.shape if hasattr(landmarks, 'shape') else 'no shape'}, data={landmarks}")
                            # 如果landmarks是多维数组，需要flatten
                            if hasattr(landmarks, 'flatten'):
                                landmarks_flat = landmarks.flatten()
                            else:
                                landmarks_flat = landmarks
                            
                            # 确保有足够的坐标点
                            if len(landmarks_flat) >= 4:
                                left_eye = [int(landmarks_flat[0]), int(landmarks_flat[1])]
                                right_eye = [int(landmarks_flat[2]), int(landmarks_flat[3])]
                        
                        face_obj = {
                            "embedding": embedding_list,
                            "facial_area": {
                                "x": int(bbox["x1"].item() if hasattr(bbox["x1"], 'item') else bbox["x1"]),
                                "y": int(bbox["y1"].item() if hasattr(bbox["y1"], 'item') else bbox["y1"]), 
                                "w": int((bbox["x2"] - bbox["x1"]).item() if hasattr(bbox["x2"], 'item') else (bbox["x2"] - bbox["x1"])),
                                "h": int((bbox["y2"] - bbox["y1"]).item() if hasattr(bbox["y2"], 'item') else (bbox["y2"] - bbox["y1"])),
                                "left_eye": left_eye,
                                "right_eye": right_eye
                            },
                            "face_confidence": float(face_data["score"])
                        }
                        faces.append(face_obj)
                    except Exception as e:
                        logger.error(f"Error processing face data {i}: {e}")
                        logger.error(f"Face data details: {face_data}")
                        logger.error(f"Detection result details: boxes={detection_result['boxes'][i] if i < len(detection_result['boxes']) else 'index out of range'}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        continue
                
                return {
                    "detector_backend": get_detector_backend(self.face_model_name),
                    "recognition_model": self.face_model_name, 
                    "result": faces
                }
            else:
                return {
                    "detector_backend": get_detector_backend(self.face_model_name),
                    "recognition_model": self.face_model_name,
                    "result": []
                }
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return {
                "detector_backend": get_detector_backend(self.face_model_name),
                "recognition_model": self.face_model_name,
                "result": []
            }

# 全局适配器实例
immich_adapter = ImmichAdapter()