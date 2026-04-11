import os
from flask import Flask

# Import blueprints
from routes.auth import auth_bp
from routes.upload import upload_bp
from routes.dashboard import dashboard_bp
from routes.model import model_bp

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Buat folder yang diperlukan
for folder in ['uploads', 'models/model_utama']:
    os.makedirs(folder, exist_ok=True)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(model_bp)

if __name__ == '__main__':
    app.run(debug=True)
