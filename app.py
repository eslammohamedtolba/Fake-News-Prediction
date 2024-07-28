from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nltk.stem.porter import PorterStemmer
from Prepare_model.Fake_News_classifier import stemming
import uvicorn
import joblib

# File paths
model_path = 'Prepare_model\\GradientBoostingClassifier_model.sav'
vectorizer_path = 'Prepare_model\\TfidfVectorizer.sav'

# Load the vectorizer and model
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
# Create porter stemmer to stemming data
stemmer = PorterStemmer()

# Create application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Route for the home page
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for the predict page
@app.post("/predict")
async def predict(request: Request, title: str = Form(...), text: str = Form(...)):
    # Prepare the input data for prediction
    content = title + ': ' + text

    # Stemming content
    content = stemming(content, stemmer)
    # Vectorize data
    content_vectorized = vectorizer.transform([content])

    # Make prediction using the loaded model
    prediction = model.predict(content_vectorized)[0]
    if prediction == 1:
        prediction = "Fake News"
    else:
        prediction = "Real News"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


