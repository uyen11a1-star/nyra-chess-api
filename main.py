import os
import subprocess
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any

SERVICE_NAME = "Nyra Chess API (C++ Core)"
VERSION = "1.0.2"

app = FastAPI(title=SERVICE_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://uyen11a1-star.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

@app.on_event("startup")
def startup_event():
    model_path = "nyra_net.onnx"
    if not os.path.exists(model_path):
        print("Không tìm thấy model ONNX, đang tải từ HuggingFace...")
        hf_token = os.getenv("HF_TOKEN", "").strip()
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        url = "https://huggingface.co/huynek231/nyra-chess-weights/resolve/main/nyra_net.onnx"
        r = requests.get(url, headers=headers, stream=True)
        if r.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            print("Tải model ONNX thành công!")
    
    # Kiểm tra xem file C++ đã được biên dịch chưa
    if os.path.exists("nyra_engine"):
        print("File nyra_engine đã sẵn sàng.")
    else:
        print("CẢNH BÁO: Không tìm thấy file nyra_engine! Kiểm tra lại build.sh")

class MoveRequest(BaseModel):
    fen: str
    legal_moves: list[Any] = Field(default_factory=list)

@app.get("/health")
def health():
    return {"status": "ready", "engine": "C++"}

@app.post("/move")
def choose_move(request: MoveRequest):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_dir, "nyra_engine")
        
        env = os.environ.copy()
        lib_path = os.path.join(current_dir, "onnxruntime", "lib")
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            [engine_path, request.fen],
            capture_output=True, text=True, timeout=20, env=env, cwd=current_dir
        )
        
        # Nếu C++ trả về mã lỗi khác 0, báo lỗi chi tiết
        if result.returncode != 0:
            error_details = f"C++ Error (Code {result.returncode}): {result.stderr.strip()}"
            print(f"[API ERROR] {error_details} | FEN: {request.fen}")
            raise HTTPException(status_code=500, detail=error_details)
            
        best_move = result.stdout.strip().split('\n')[-1]
        if not best_move or best_move == "NONE":
            raise HTTPException(status_code=400, detail="No move found")
            
        return {"uci": best_move, "search": {"depth": 3, "method": "C++ Alpha-Beta"}}
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="C++ Engine Timeout (quá 20 giây)")
    except Exception as e:
        # Bắt mọi lỗi khác (ví dụ không tìm thấy file)
        raise HTTPException(status_code=500, detail=str(e))
