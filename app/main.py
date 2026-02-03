from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <title>Judging App</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
      </head>
      <body style="font-family: sans-serif; text-align: center; padding: 2rem;">
        <h1>Judging App is Live 🏆</h1>
        <p>If you can see this, Render is running your Python app.</p>
      </body>
    </html>
    """
