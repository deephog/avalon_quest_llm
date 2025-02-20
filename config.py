import os

class Config:
    # 基础配置
    SECRET_KEY = 'your-secret-key'
    
    # 数据库配置
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DB_PATH = os.environ.get('AVALON_DB_PATH', os.path.join(BASE_DIR, 'data'))
    DB_NAME = 'avalon.db'
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_PATH, DB_NAME)}?check_same_thread=False'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GEMINI_API_KEY = 'AIzaSyDJYUOEewmdMLj54ZJ1XYa6nGVKBTFIntc'  # 从环境变量或其他安全位置获取
    FIREWORKS_API_KEY = 'fw_3Zjs7pZN9mYqpSXZQVUzenXi'  # 从环境变量或其他安全位置获取

    # 管理员账号配置
    ADMIN_USERNAME = os.environ.get('AVALON_ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('AVALON_ADMIN_PASSWORD', 'admin123')

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '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') 