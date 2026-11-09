import uvicorn

if __name__ == "__main__":
    print("Abrindo em http://127.0.0.1:8000 (endpoints para teste em http://127.0.0.1:8000/docs)")
    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000)
