import os
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Убираем app.run и даем возможность запускать приложение через gunicorn.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Используем переменную окружения PORT
    app.run(debug=True, host="0.0.0.0", port=port)
