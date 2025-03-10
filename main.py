# from fastapi import FastAPI
#
# app = FastAPI()
#
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}

from fastapi import FastAPI
from routes import classify

app = FastAPI(title="Text Classification API")

# API 라우터 등록
app.include_router(classify.router, prefix="/api", tags=["Classification"])

@app.get("/")
def root():
    return {"message": "FastAPI KO-BERT 문장 분류 API"}
