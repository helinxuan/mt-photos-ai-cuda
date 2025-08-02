from dotenv import load_dotenv
import os
import sys
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
import logging
# from paddleocr import PaddleOCR
import torch
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
from paddleocr import PaddleOCR  # 使用PaddleOCR GPU版本
# import cn_clip.clip as clip  # 注释掉原有的clip导入
ImageFile.LOAD_TRUNCATED_IMAGES = True

on_linux = sys.platform.startswith('linux')

# 必须在导入immich_adapter之前加载.env文件
load_dotenv()
from immich_adapter import immich_adapter  # 使用immich适配器

# 配置日志
logger = logging.getLogger("uvicorn")
# 设置日志级别
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 是否自动加载CLIP文本模型，开启可以优化第一次搜索时的响应速度,文本模型占用700多m内存

# clip_model_name = os.getenv("CLIP_MODEL")  # 移到immich_adapter中管理


ocr_model = None
# clip_processor = None  # 不再需要，使用immich适配器
# clip_model = None  # 不再需要，使用immich适配器

restart_task = None
restart_lock = None

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    """预加载OCR模型"""
    global ocr_model
    if ocr_model is None:
        logger.info("Loading OCR model 'PaddleOCR' to memory")
        
        # 根据 PaddleOCR 3.0 官方文档，使用默认配置
        # 默认使用 PP-OCRv5_server 模型，支持中英文识别
        # PaddleOCR 会自动下载模型到系统默认缓存目录
        # 注意：只有在已有完整模型文件时才能使用 text_detection_model_dir 参数
        ocr_model = PaddleOCR()
        if torch.cuda.is_available():
            logger.info("PaddleOCR initialized with GPU acceleration")
        else:
            logger.info("PaddleOCR initialized with CPU")
        logger.info("PaddleOCR models will be automatically cached by the system")
        # https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html

def load_clip_model():
    """预加载CLIP模型 - 使用immich适配器"""
    try:
        # 预加载视觉和文本模型
        immich_adapter.load_clip_visual_model()
        if env_auto_load_txt_modal:
            immich_adapter.load_clip_textual_model()
    except Exception as e:
        print(f"Error loading CLIP models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    global restart_lock
    restart_lock = asyncio.Lock()
    import onnxruntime as ort
    logger.info("Using PaddleOCR with GPU support")
    logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'ERROR')}")
    logger.info(f"MODEL_TTL: {os.getenv('MODEL_TTL', '0')}")
    logger.info(f"FACE_MODEL_NAME: {immich_adapter.face_model_name}")
    logger.info(f"CLIP_MODEL_NAME: {immich_adapter.clip_model_name}")
    logger.info(f"FACE_THRESHOLD: {immich_adapter.face_threshold}")
    logger.info(f"FACE_MAX_DISTANCE: {immich_adapter.face_max_distance}")
    logger.info(f"DEVICE: {device}")
    logger.info(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
    # 输出ONNX Runtime执行提供程序信息
    available_providers = ort.get_available_providers()

    logger.info(f"ONNX_PROVIDERS: {available_providers}")
    # 检查CUDA执行提供程序
    if 'CUDAExecutionProvider' in available_providers:
        logger.info(f"CUDA_RUNTIME: Available")
    else:
        logger.info(f"CUDA_RUNTIME: Not Available")
    if env_auto_load_txt_modal:
        load_clip_model()
        logger.info("Auto-loaded CLIP text model")
    
    yield
    
    # 关闭事件
    if restart_task and not restart_task.done():
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass

# 创建 FastAPI 应用实例
app = FastAPI(lifespan=lifespan)

async def restart_timer():
    await asyncio.sleep(server_restart_time)
    restart_program()

@app.middleware("http")
async def activity_monitor(request, call_next):
    global restart_task

    # 确保 restart_lock 已初始化
    if restart_lock is not None:
        async with restart_lock:
            if restart_task and not restart_task.done():
                restart_task.cancel()

            restart_task = asyncio.create_task(restart_timer())

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))

def convert_paddleocr_to_json(paddleocr_output):
    """
    将 PaddleOCR 3.0 的输出转换为 JSON 格式
    
    Args:
        paddleocr_output: PaddleOCR 3.0 的原始输出
        
    Returns:
        dict: 包含 texts, scores, boxes 的字典
    """
    logger.debug(f"convert_paddleocr_to_json input type: {type(paddleocr_output)}")
    logger.debug(f"convert_paddleocr_to_json input content: {paddleocr_output}")
    
    texts = []
    scores = []
    boxes = []
    
    try:
        # PaddleOCR 3.0 新格式：返回字典包含 rec_texts, rec_scores, rec_polys 等字段
        if isinstance(paddleocr_output, list) and len(paddleocr_output) > 0:
            # 检查是否为新的字典格式
            if isinstance(paddleocr_output[0], dict):
                result_dict = paddleocr_output[0]
                
                # 提取文本
                if 'rec_texts' in result_dict:
                    texts = [str(text) for text in result_dict['rec_texts']]
                
                # 提取置信度
                if 'rec_scores' in result_dict:
                    scores = [f"{float(score):.2f}" for score in result_dict['rec_scores']]
                
                # 提取边界框坐标
                if 'rec_polys' in result_dict:
                    for poly in result_dict['rec_polys']:
                        if hasattr(poly, 'tolist'):
                            # 处理 numpy 数组
                            poly = poly.tolist()
                        
                        if isinstance(poly, list) and len(poly) == 4:
                            # 计算矩形边界框
                            xs = [point[0] for point in poly]
                            ys = [point[1] for point in poly]
                            
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            boxes.append({
                                'x': to_fixed(x_min),
                                'y': to_fixed(y_min),
                                'width': to_fixed(width),
                                'height': to_fixed(height)
                            })
                
                logger.info(f"PaddleOCR 3.0 new format processed: {len(texts)} texts, {len(scores)} scores, {len(boxes)} boxes")
            
            # 兼容旧格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]]
            elif isinstance(paddleocr_output[0], list) and len(paddleocr_output[0]) == 2:
                for line in paddleocr_output:
                    if isinstance(line, list) and len(line) == 2:
                        bbox, text_info = line
                        
                        # 处理边界框坐标
                        if isinstance(bbox, list) and len(bbox) == 4:
                            # 计算矩形边界框
                            xs = [point[0] for point in bbox]
                            ys = [point[1] for point in bbox]
                            
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            boxes.append({
                                'x': to_fixed(x_min),
                                'y': to_fixed(y_min),
                                'width': to_fixed(width),
                                'height': to_fixed(height)
                            })
                        
                        # 处理文本和置信度
                        if isinstance(text_info, list) and len(text_info) >= 2:
                            text, confidence = text_info[0], text_info[1]
                            texts.append(str(text))
                            scores.append(f"{float(confidence):.2f}")
                        else:
                            logger.warning(f"Unexpected text_info format: {text_info}")
                            texts.append("")
                            scores.append("0.00")
                    else:
                        logger.warning(f"Unexpected line format: {line}")
                
                logger.info(f"PaddleOCR legacy format processed: {len(texts)} texts, {len(scores)} scores, {len(boxes)} boxes")
            else:
                logger.warning(f"Unexpected paddleocr_output list format: {paddleocr_output[0]}")
        else:
            logger.warning(f"Unexpected paddleocr_output format: {type(paddleocr_output)}")
    
    except Exception as e:
        logger.error(f"Error in convert_paddleocr_to_json: {e}")
        logger.error(f"paddleocr_output: {paddleocr_output}")
    
    logger.debug(f"Conversion result - texts: {len(texts)}, scores: {len(scores)}, boxes: {len(boxes)}")
    
    return {
        'texts': texts,
        'scores': scores,
        'boxes': boxes
    }

@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务 (集成Immich)",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "device": device,
        "face_model": immich_adapter.face_model_name,
        "clip_model": immich_adapter.clip_model_name,
        "detector_backend": immich_adapter.face_model_name,
        "recognition_model": immich_adapter.face_model_name
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # cuda版本 OCR没有显存未释放问题，这边可以关闭重启
    return {'result': 'unsupported'}
    # restart_program()

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    logger.info(f"ocr_process_image Received {file.content_type} file: {file.filename}")
    load_ocr_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        # 根据 PaddleOCR 3.0 官方文档，直接调用 predict 方法
        _result = await asyncio.get_running_loop().run_in_executor(None, ocr_model.predict, img)
        logger.info(f"Raw PaddleOCR result for {file.filename}: {_result}")
        result = convert_paddleocr_to_json(_result)
        del img
        del _result
        logger.info(f"OCR processing completed for {file.filename}, texts found: {len(result.get('texts', []))}, scores: {len(result.get('scores', []))}, boxes: {len(result.get('boxes', []))}")
        if len(result.get('texts', [])) == 0:
            logger.warning(f"No text extracted from {file.filename}, check convert_paddleocr_to_json function")
        return {'result': result}
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return {'result': [], 'msg': str(e)}

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    logger.info(f"clip_process_image Received {file.content_type} file: {file.filename}")
    image_bytes = await file.read()
    try:
        # 使用immich适配器进行图像编码
        result = await asyncio.get_running_loop().run_in_executor(
            None, immich_adapter.encode_image, image_bytes
        )
        logger.info(f"CLIP image processing completed for {file.filename}, result: {result[:3] if len(result) > 3 else result}...")
        return {'result': result}
    except Exception as e:
        logger.error(f"CLIP image processing error: {e}")
        return {'result': [], 'msg': str(e)}

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    logger.info(f"clip_process_text Received text query: {request.text[:50]}...")
    try:
        # 使用immich适配器进行文本编码
        result = await asyncio.get_running_loop().run_in_executor(
            None, immich_adapter.encode_text, request.text
        )
        logger.info(f"CLIP text processing completed, result: {result[:3] if len(result) > 3 else result}...")
        return {'result': result}
    except Exception as e:
        logger.error(f"CLIP text processing error: {e}")
        return {'result': [], 'msg': str(e)}

@app.post("/represent")
async def face_represent(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    """人脸特征提取API - 兼容MT-Photos格式，使用Immich后端"""
    logger.info(f"face_represent Received {file.content_type} file: {file.filename}")
    content_type = file.content_type
    image_bytes = await file.read()
    
    try:
        img = None
        if content_type == 'image/gif':
            # 处理GIF文件的第一帧
            with Image.open(BytesIO(image_bytes)) as pil_img:
                if pil_img.is_animated:
                    pil_img.seek(0)
                frame = pil_img.convert('RGB')
                np_arr = np.array(frame)
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        
        if img is None:
            # 处理其他图像格式
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            err = f"The uploaded file {file.filename} is not a valid image format or is corrupted."
            logger.error(err)
            return {'result': [], 'msg': str(err)}
        
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}
        
        data = {
            "detector_backend": immich_adapter.face_model_name,
            "recognition_model": immich_adapter.face_model_name
        }
        
        # 使用Immich适配器进行人脸特征提取
        embedding_objs = await asyncio.get_running_loop().run_in_executor(
            None, _immich_represent, image_bytes
        )
        
        del img
        data["result"] = embedding_objs
        logger.info(f"Face representation completed for {file.filename}, embeddings count: {len(embedding_objs)}")
        return data
        
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        logger.error(f"Face representation error: {e}")
        return {'result': [], 'msg': str(e)}

def _immich_represent(image_bytes):
    """使用Immich适配器进行人脸特征提取"""
    try:
        face_result = immich_adapter.detect_faces(image_bytes)
        # 直接返回result数组，保持与DeepFace.represent格式兼容
        if face_result and 'result' in face_result:
            return face_result['result']
        return []
    except Exception as e:
        logger.error(f"Immich face representation error: {e}")
        return []

async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host=None, port=http_port)
