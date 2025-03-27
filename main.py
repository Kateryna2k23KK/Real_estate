from flask import Flask

# Создаём объект Flask приложения
app = Flask(__name__)

# Определяем маршрут для главной страницы
@app.route('/')
def home():
    return "Hello, World!"

# Если файл запускается напрямую, запускаем сервер
if __name__ == '__main__':
    app.run(debug=True)
