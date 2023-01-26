from data_preprocessing import load_word_embeddings
from summarizers import pagerank_summarizer
from fastapi import FastAPI, Form
from jinja2 import Environment, FileSystemLoader
from fastapi.responses import HTMLResponse
import uvicorn


app = FastAPI()
word_embeddings = load_word_embeddings()
env = Environment(loader=FileSystemLoader("templates/"))


@app.post("/evaluate")
def evaluation(text_area: str = Form(...)):
    result = pagerank_summarizer(word_embeddings, text_area)
    template = env.get_template("evaluate.html")
    return HTMLResponse(content=template.render(result=result))


@app.get("/evaluate")
async def evaluation_get():
    print("No Post Back Call")
    template = env.get_template("index.html")
    return HTMLResponse(content=template.render())


@app.get("/")
async def root():
    template = env.get_template("index.html")
    return HTMLResponse(content=template.render())


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
