from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from flask_socketio import SocketIO, emit
from game_logic import AvalonSimulator, set_model_api
from game_output import WebSocketOutput
from models import db, User, Game, GamePlayer, GameRound, Speech, AIThought, IdentityGuess
from config import Config
import queue
import threading
import os
import logging
import pandas as pd
import io
import json
from datetime import datetime
from flask_migrate import Migrate

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*")

db.init_app(app)
migrate = Migrate(app, db)  # 添加 Flask-Migrate

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_db_directory():
    """确保数据库目录存在并且有正确的权限"""
    try:
        os.makedirs(Config.DB_PATH, exist_ok=True)
        db_file = os.path.join(Config.DB_PATH, Config.DB_NAME)
        logger.info(f"Database path: {db_file}")
        # 如果数据库文件已存在，确保有写权限
        if os.path.exists(db_file):
            os.chmod(db_file, 0o666)
    except Exception as e:
        logger.error(f"Error setting up database directory: {e}")
        raise

def init_db():
    ensure_db_directory()
    db.create_all()
    # 创建管理员账号（如果不存在）
    admin = User.query.filter_by(username=Config.ADMIN_USERNAME).first()
    if not admin:
        admin = User(username=Config.ADMIN_USERNAME, is_admin=True)
        admin.set_password(Config.ADMIN_PASSWORD)
        db.session.add(admin)
        db.session.commit()
        print(f"Created admin user: {Config.ADMIN_USERNAME}, is_admin: {admin.is_admin}")  # 调试信息
    else:
        print(f"Admin user exists: {Config.ADMIN_USERNAME}, is_admin: {admin.is_admin}")  # 调试信息

# 使用 with app.app_context() 来初始化数据库
with app.app_context():
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

class GameManager:
    def __init__(self):
        self.reset_game()
        self.output = WebSocketOutput(socketio, self)
        self.current_user = None
        print("GameManager initialized")

    def set_language(self, lang):
        """设置语言"""
        if self.output:
            self.output.set_language(lang)
        print(f"Game manager language set to: {lang}")

    def reset_game(self):
        self.game = None
        self.input_queue = queue.Queue()
        self.current_player = None
        self.awaiting_input = False
        self.current_round = 0
        self.output = WebSocketOutput(socketio, self)

    def run_game(self, test_mode=False, p5_is_morgan=False, player_models=None):
        print("Game started")
        with app.app_context():
            if not self.game:
                self.game = AvalonSimulator(
                    output=self.output,
                    player_models=player_models,
                    test_mode=test_mode
                )
            self.game.run_game(test_mode=test_mode, p5_is_morgan=p5_is_morgan)
            print("Game finished")
            self.reset_game()

    def wait_for_input(self):
        """等待玩家输入"""
        print("Waiting for player input...")  # 调试信息
        while True:
            try:
                input_text = self.input_queue.get(timeout=300)  # 5分钟超时
                print(f"Received input: {input_text}")  # 调试信息
                return input_text
            except queue.Empty:
                print("Input timeout")  # 调试信息
                return ""

    def update_stats(self, role, won):
        """更新玩家统计数据"""
        if not self.current_user:
            return
        
        user = User.query.get(self.current_user)
        if role == 'blue':
            if won:
                user.blue_wins += 1
            else:
                user.blue_losses += 1
        else:
            if won:
                user.red_wins += 1
            else:
                user.red_losses += 1
        
        db.session.commit()

game_manager = GameManager()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            session['user_id'] = user.id
            session['is_admin'] = user.is_admin
            print(f"User logged in: {user.username}, is_admin: {user.is_admin}")  # 调试信息
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # 检查用户名长度
        if len(username) < 3:
            flash('Username must be at least 3 characters long')
            return redirect(url_for('register'))

        # 检查密码长度
        if len(password) < 6:
            flash('Password must be at least 6 characters long')
            return redirect(url_for('register'))

        # 检查密码确认
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))

        if User.query.filter_by(username=request.form['username']).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    print(f"Rendering index for user: {user.username}, is_admin: {user.is_admin}")  # 调试信息
    return render_template('index.html', user=user)

@socketio.on('connect')
def handle_connect():
    lang = request.args.get('lang', 'zh')
    print(f"Client connected with language: {lang}")
    if game_manager and game_manager.output:
        game_manager.output.set_language(lang)

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")  # 调试信息

@socketio.on('start_game')
def handle_start_game(data):
    include_human = data.get('include_human', True)
    player_models = data.get('player_models', {})
    random_team = data.get('random_team', True)
    player_teams = data.get('player_teams', {})
    simulation_count = data.get('simulation_count', 1)
    
    print("Starting game")
    print(f"Simulation count: {simulation_count}")
    
    def run_multiple_games():
        with app.app_context():  # 添加应用上下文
            for i in range(simulation_count):
                print(f"\n开始第 {i+1}/{simulation_count} 次模拟")
                
                game_manager.game = AvalonSimulator(
                    output=game_manager.output,
                    human_player_id="P5" if include_human else None,
                    player_models=player_models,
                    random_team=random_team,
                    player_teams=player_teams
                )
                
                game_manager.run_game(
                    test_mode=False,
                    p5_is_morgan=False,
                    player_models=player_models
                )
                
                if i < simulation_count - 1:  # 不是最后一次模拟
                    game_manager.output.send_message("\n=== 准备开始下一轮模拟 ===\n", 'info')
    
    game_thread = threading.Thread(target=run_multiple_games)
    game_thread.start()

@socketio.on('player_input')
def handle_player_input(data):
    """接收玩家输入并放入队列"""
    if game_manager.game and game_manager.input_queue:
        game_manager.input_queue.put(data['input'])

@socketio.on('start_test_game')
def handle_test_game(data):
    try:
        p5_is_morgan = data.get('p5_is_morgan', False)
        include_human = data.get('include_human', False)
        print(f"Received test game request: p5_is_morgan={p5_is_morgan}, include_human={include_human}")  # 调试日志
        
        game = AvalonSimulator(
            output=WebSocketOutput(socketio, game_manager),
            test_mode=True,
            p5_is_morgan=p5_is_morgan,
            human_player_id="P5" if include_human else None
        )
        
        game_manager.current_game = game
        game.run_game()
    except Exception as e:
        print(f"Error starting test game: {e}")

@app.route('/export/<format>')
def export_data(format):
    if 'user_id' not in session or not session.get('is_admin'):
        return {'error': 'Unauthorized'}, 403

    if format == 'excel':
        # 创建一个 Excel 文件，包含多个 sheet
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # 使用 SQLAlchemy 引擎
            engine = db.session.get_bind()

            # 游戏基本信息
            games_df = pd.read_sql("""
                SELECT id, start_time, end_time, winner, blue_wins, red_wins 
                FROM game
            """, engine)
            games_df.to_excel(writer, sheet_name='Games', index=False)

            # 玩家信息
            players_df = pd.read_sql("""
                SELECT game_id, player_id, role, is_ai, character, strategy, team_mates, morgan
                FROM game_player
            """, engine)
            players_df.to_excel(writer, sheet_name='Players', index=False)

            # 轮次信息
            rounds_df = pd.read_sql("""
                SELECT game_id, round_number, leader_id, team_members, fail_votes, result
                FROM game_round
            """, engine)
            rounds_df.to_excel(writer, sheet_name='Rounds', index=False)

            # 发言记录
            speeches_df = pd.read_sql("""
                SELECT r.game_id, r.round_number, s.player_id, s.content
                FROM speech s
                JOIN game_round r ON s.round_id = r.id
            """, engine)
            speeches_df.to_excel(writer, sheet_name='Speeches', index=False)

            # AI 思考记录
            thoughts_df = pd.read_sql("""
                SELECT r.game_id, r.round_number, t.player_id, t.summary
                FROM ai_thought t
                JOIN game_round r ON t.round_id = r.id
            """, engine)
            thoughts_df.to_excel(writer, sheet_name='AI_Thoughts', index=False)

            # 身份猜测记录
            guesses_df = pd.read_sql("""
                SELECT r.game_id, r.round_number, g.guesser_id, g.target_id, g.guessed_role
                FROM identity_guess g
                JOIN game_round r ON g.round_id = r.id
            """, engine)
            guesses_df.to_excel(writer, sheet_name='Identity_Guesses', index=False)

        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'avalon_game_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

    elif format == 'training':
        # 导出训练数据格式
        training_data = []
        games = Game.query.all()
        
        for game in games:
            game_data = {
                'game_id': game.id,
                'winner': game.winner,
                'rounds': []
            }
            
            # 获取游戏中的所有玩家信息
            players = {p.player_id: {
                'role': p.role,
                'is_ai': p.is_ai,
                'character': p.character,
                'strategy': p.strategy,
                'team_mates': p.team_mates.split(',') if p.team_mates else [],
                'morgan': p.morgan
            } for p in game.players}
            
            game_data['players'] = players
            
            # 获取每一轮的详细信息
            for round in game.rounds:
                round_data = {
                    'round_number': round.round_number,
                    'leader': round.leader_id,
                    'team': round.team_members.split(','),
                    'result': round.result,
                    'fail_votes': round.fail_votes,
                    'speeches': {},
                    'thoughts': {},
                    'identity_guesses': {}
                }
                
                # 添加发言
                for speech in round.speeches:
                    round_data['speeches'][speech.player_id] = speech.content
                
                # 添加AI思考
                for thought in round.ai_thoughts:
                    round_data['thoughts'][thought.player_id] = thought.summary
                
                # 添加身份猜测
                for guess in round.identity_guesses:
                    if guess.guesser_id not in round_data['identity_guesses']:
                        round_data['identity_guesses'][guess.guesser_id] = {}
                    round_data['identity_guesses'][guess.guesser_id][guess.target_id] = guess.guessed_role
                
                game_data['rounds'].append(round_data)
            
            training_data.append(game_data)
        
        return jsonify(training_data)

    return {'error': 'Invalid format'}, 400

@app.route('/change_model', methods=['POST'])
def change_model():
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    model = data.get('model')
    
    if model not in ['chatgpt', 'ollama-32b', 'ollama-7b', 'glm-zero', 'deepseek-reasoner', 'doubao-lite', 'siliconflow']:
        return jsonify({'success': False, 'error': 'Invalid model'}), 400
    
    # 通过函数设置模型
    set_model_api(model)
    
    # 重新初始化游戏
    game_manager.game = AvalonSimulator(output=game_manager.output)
    
    return jsonify({'success': True})

@app.route('/create_game', methods=['POST'])
def create_game():
    test_mode = request.form.get('test_mode') == 'true'
    p5_is_morgan = request.form.get('p5_is_morgan') == 'true'
    include_human = request.form.get('include_human') == 'true'
    
    # 创建新游戏
    game = AvalonSimulator(
        output=WebSocketOutput(socketio, game_manager),
        test_mode=test_mode,
        p5_is_morgan=p5_is_morgan,
        human_player_id="P5" if include_human else None
    )

@app.route('/set_language', methods=['POST'])
def set_language():
    lang = request.json.get('lang')
    print(f"Language change request received: {lang}")
    if lang in ['en', 'zh']:
        session['lang'] = lang
        if game_manager:
            game_manager.set_language(lang)
            if game_manager.game:
                game_manager.game.lang = lang
                print(f"Game language set to: {lang}")  # 调试日志
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Unsupported language'})

# 添加规则文件的路径配置
RULES_PATH = os.path.dirname(os.path.abspath(__file__))

@app.route('/rules/<lang>')
def get_rules(lang):
    try:
        filename = 'game_rules.md' if lang == 'zh' else 'game_rules_en.md'
        file_path = os.path.join(RULES_PATH, filename)
        print(f"Attempting to read rules file: {file_path}")  # 调试信息
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return jsonify({'content': content})
    except Exception as e:
        print(f"Error loading rules file: {str(e)}")  # 调试信息
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
