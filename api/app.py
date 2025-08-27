from flask import Flask
from api.routes import register_routes

def create_app():
    app = Flask(__name__)
    register_routes(app)
    return app

app = create_app()
app.config['JSON_AS_ASCII'] = False
