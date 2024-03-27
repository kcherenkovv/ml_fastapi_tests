from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel

#Добавим async версии для FastAPI, чтобы улучшить асинхронную обработку запросов.
#Разделим создание экземпляров модели и токенизатора от основного потока выполнения, 
#чтобы уменьшить время загрузки при старте сервера.

class Item(BaseModel):
    text: str



app = FastAPI()
# Инициализация вне основного потока выполнения
classifier = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")





@app.get("/", response_model=dict)
async def root() -> dict:
    return {"message": "Hello World"}


@app.post("/predict/", response_model=dict)
async def predict(item: Item) -> dict:
    try:
        prediction = classifier(item.text)[0]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/", response_model=dict)
async def translate_text(item: Item) -> dict:
    try:
        inputs = tokenizer(item.text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
