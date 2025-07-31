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
restart_lock = asyncio.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        logger.info("Loading OCR model 'PaddleOCR' to memory")
        ocr_model = PaddleOCR(
            use_angle_cls=True,  # 使用角度分类器
            lang='ch',  # 支持中文识别，也支持英文
            use_gpu=torch.cuda.is_available(),  # 使用GPU加速
            show_log=True,  # 显示详细日志用于调试
            det=True,  # 启用文本检测
            rec=True,  # 启用文本识别
            cls=True,  # 启用文本方向分类
            ocr_version='PP-OCRv4'  # 使用PP-OCRv4文字识别
        )
        if torch.cuda.is_available():
            logger.info("PaddleOCR initialized with GPU acceleration")
        else:
            logger.info("PaddleOCR initialized with CPU")
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
    import onnxruntime as ort
    logger.info("Using PaddleOCR with GPU support")
    logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'ERROR')}")
    logger.info(f"MODEL_TTL: {os.getenv('MODEL_TTL', '0')}")
    logger.info(f"FACE_MODEL_NAME: {immich_adapter.face_model_name}")
    logger.info(f"CLIP_MODEL_NAME: {immich_adapter.clip_model_name}")
    logger.info(f"FACE_THRESHOLD: {immich_adapter.face_threshold}")
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
    texts = []
    scores = []
    boxes = []
    
    logger.debug(f"PaddleOCR raw output: {paddleocr_output}")
    logger.debug(f"PaddleOCR output type: {type(paddleocr_output)}")
    
    if not paddleocr_output or len(paddleocr_output) == 0:
        logger.warning("PaddleOCR returned empty output")
        return {'texts': [], 'scores': [], 'boxes': []}
    
    # PaddleOCR 返回格式: (boxes_list, texts_with_scores_list, timing_dict)
    if isinstance(paddleocr_output, tuple) and len(paddleocr_output) >= 2:
        boxes_list = paddleocr_output[0]  # 边界框列表
        texts_with_scores = paddleocr_output[1]  # 文本和置信度列表
        
        logger.debug(f"Boxes list length: {len(boxes_list) if boxes_list else 0}")
        logger.debug(f"Texts with scores length: {len(texts_with_scores) if texts_with_scores else 0}")
        
        if boxes_list and texts_with_scores and len(boxes_list) == len(texts_with_scores):
            try:
                for i, (bbox, (text, confidence)) in enumerate(zip(boxes_list, texts_with_scores)):
                    logger.debug(f"Processing item {i}: text='{text}', confidence={confidence}")
                    
                    texts.append(text)
                    scores.append(f"{confidence:.2f}")
                    
                    # 处理边界框坐标
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
            except Exception as e:
                logger.error(f"Error parsing PaddleOCR tuple format: {e}")
                return {'texts': [], 'scores': [], 'boxes': []}
        else:
            logger.warning(f"Mismatch in PaddleOCR output lengths: boxes={len(boxes_list) if boxes_list else 0}, texts={len(texts_with_scores) if texts_with_scores else 0}")
    else:
        # 兼容旧格式处理
        logger.debug(f"PaddleOCR output[0]: {paddleocr_output[0]}")
        logger.debug(f"PaddleOCR output[0] type: {type(paddleocr_output[0])}")
        logger.debug(f"PaddleOCR output[0] length: {len(paddleocr_output[0]) if paddleocr_output[0] else 0}")
        
        try:
            for i, line in enumerate(paddleocr_output[0]):  # PaddleOCR返回的是嵌套列表
                logger.debug(f"Processing line {i}: {line}, type: {type(line)}")
                # 检查line的类型和长度，PaddleOCR可能返回不同格式
                if isinstance(line, (list, tuple)) and len(line) == 2:  # 格式: [bbox, (text, confidence)]
                    bbox, (text, confidence) = line
                elif isinstance(line, (list, tuple)) and len(line) == 3:  # 格式: [bbox, text, confidence]
                    bbox, text, confidence = line
                elif isinstance(line, np.ndarray):  # 只有边界框坐标的情况
                    logger.debug(f"PaddleOCR returned only bbox coordinates: {line}")
                    continue  # 跳过只有坐标没有文本的结果
                else:
                    logger.warning(f"Unexpected PaddleOCR output format: {line}")
                    continue
                
                texts.append(text)
                scores.append(f"{confidence:.2f}")
                
                # 处理边界框坐标
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
        except Exception as e:
            logger.error(f"Error parsing PaddleOCR output: {e}, output: {paddleocr_output}")
            return {'texts': [], 'scores': [], 'boxes': []}

    output = {
        'texts': texts,
        'scores': scores,
        'boxes': boxes
    }

    return output

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

        _result = await predict(ocr_model, img)
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

@app.post("/face/detect")
async def face_detect(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    """人脸检测和识别API - 使用Immich"""
    logger.info(f"face_detect Received {file.content_type} file: {file.filename}")
    image_bytes = await file.read()
    try:
        # 使用immich适配器进行人脸检测和识别
        result = await asyncio.get_running_loop().run_in_executor(
            None, immich_adapter.detect_faces, image_bytes
        )
        logger.info(f"Face detection completed for {file.filename}, faces found: {len(result.get('faces', []))}")
        return {'result': result}
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return {'result': {'faces': [], 'detection': {'boxes': [], 'scores': [], 'landmarks': []}}, 'msg': str(e)}

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
