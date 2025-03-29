import os
from flask import Flask, render_template

# Указываем Flask, где искать шаблоны и статические файлы
app = Flask(__name__,
            template_folder='templates',  # Папка с шаблонами
            static_folder='static')       # Папка со статическими файлами

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Используем переменную окружения PORT
    app.run(debug=True, host="0.0.0.0", port=port)
