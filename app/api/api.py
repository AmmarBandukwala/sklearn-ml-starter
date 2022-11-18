import uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run("api:app")
    
# TO RUN GO TO app/api folder, from terminal run 'uvicorn api:app --reload' for local development.