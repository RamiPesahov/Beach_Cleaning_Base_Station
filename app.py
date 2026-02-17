from flask  import Flask
from robot  import robot_bp
from laptop import laptop_bp

app = Flask(__name__)

app.register_blueprint(robot_bp, url_prefix='/robot')
app.register_blueprint(laptop_bp, url_prefix='/laptop')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 