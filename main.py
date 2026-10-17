import os
import subprocess
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any

SERVICE_NAME = "Nyra Chess API (C++ Core)"
VERSION = "1.0.1"

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
        else:
            print(f"Lỗi tải model: HTTP {r.status_code}")

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

        # Tăng timeout lên 20s
        result = subprocess.run(
            [engine_path, request.fen],
            capture_output=True, text=True, timeout=20, env=env, cwd=current_dir
        )
        
        if result.returncode != 0:
            print(f"[C++ CRASH] FEN: {request.fen}")
            print(f"[C++ CRASH] Error: {result.stderr}")
            raise Exception(f"Engine error: {result.stderr}")
            
        best_move = result.stdout.strip().split('\n')[-1]
        if not best_move or best_move == "NONE":
            raise HTTPException(status_code=400, detail="No move found")
            
        return {"uci": best_move, "search": {"depth": 3, "method": "C++ Alpha-Beta"}}
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] C++ chạy quá 20s ở FEN: {request.fen}")
        raise HTTPException(status_code=504, detail="Engine Timeout")
    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
