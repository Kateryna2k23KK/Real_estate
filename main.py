<<<<<<< HEAD
from flask import Flask, render_template

# Создание экземпляра приложения Flask
app = Flask(__name__)

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для страницы "О нас"
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Flask будет искать файл в папке templates

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 39f4f1a (Update Procfile to use main.py)
