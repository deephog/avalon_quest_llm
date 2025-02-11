from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    
    # 游戏统计
    blue_wins = db.Column(db.Integer, default=0)
    blue_losses = db.Column(db.Integer, default=0)
    blue_winrate = db.Column(db.Float, default=0)
    red_wins = db.Column(db.Integer, default=0)
    red_losses = db.Column(db.Integer, default=0)
    red_winrate = db.Column(db.Float, default=0)

    # 关联
    games = db.relationship('GamePlayer', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def blue_winrate(self):
        total = self.blue_wins + self.blue_losses
        return round(self.blue_wins / total * 100, 2) if total > 0 else 0

    @property
    def red_winrate(self):
        total = self.red_wins + self.red_losses
        return round(self.red_wins / total * 100, 2) if total > 0 else 0

    def get_game_history(self, limit=10):
        """获取用户最近的游戏历史"""
        return GamePlayer.query.filter_by(user_id=self.id).order_by(
            GamePlayer.id.desc()
        ).limit(limit).all()

    def export_game_data(self, format='json'):
        """导出用户的游戏数据"""
        games = self.get_game_history(limit=None)
        if format == 'json':
            return self._export_as_json(games)
        # 可以添加其他格式的导出

    def _export_as_json(self, games):
        data = []
        for game_player in games:
            game_data = {
                'game_id': game_player.game_id,
                'player_id': game_player.player_id,
                'role': game_player.role,
                'rounds': []
            }
            
            for round in game_player.game.rounds:
                round_data = {
                    'round_number': round.round_number,
                    'leader': round.leader_id,
                    'team': round.team_members.split(','),
                    'result': round.result,
                    'fail_votes': round.fail_votes,
                    'speech': next((s.content for s in round.speeches 
                                  if s.player_id == game_player.player_id), None),
                    'identity_guesses': [
                        {'target': g.target_id, 'guess': g.guessed_role}
                        for g in round.identity_guesses
                        if g.guesser_id == game_player.player_id
                    ]
                }
                game_data['rounds'].append(round_data)
            
            data.append(game_data)
        return data

class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    winner = db.Column(db.String(10))  # 'blue' or 'red'
    blue_wins = db.Column(db.Integer)
    red_wins = db.Column(db.Integer)
    
    # 关联
    rounds = db.relationship('GameRound', backref='game', lazy=True)
    players = db.relationship('GamePlayer', backref='game', lazy=True)

class GamePlayer(db.Model):
    __tablename__ = 'game_player'  # 确保表名正确
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('game.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    player_id = db.Column(db.String(10))
    role = db.Column(db.String(10))
    is_ai = db.Column(db.Boolean, default=True)
    character = db.Column(db.String(50), nullable=True)
    strategy = db.Column(db.String(50), nullable=True)
    team_mates = db.Column(db.String(50))
    morgan = db.Column(db.Boolean, default=False)

class GameRound(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('game.id'))
    round_number = db.Column(db.Integer)
    leader_id = db.Column(db.String(10))
    next_leader_id = db.Column(db.String(10))
    team_members = db.Column(db.String(50))  # 存储为逗号分隔的字符串
    fail_votes = db.Column(db.Integer)
    result = db.Column(db.String(10))  # 'success' or 'fail'
    
    # 关联
    speeches = db.relationship('Speech', backref='round', lazy=True)
    ai_thoughts = db.relationship('AIThought', backref='round', lazy=True)
    identity_guesses = db.relationship('IdentityGuess', backref='round', lazy=True)

class Speech(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('game_round.id'))
    player_id = db.Column(db.String(10))
    content = db.Column(db.Text)

class AIThought(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('game_round.id'))
    player_id = db.Column(db.String(10))
    summary = db.Column(db.Text)

class IdentityGuess(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('game_round.id'))
    guesser_id = db.Column(db.String(10))
    target_id = db.Column(db.String(10))
    guessed_role = db.Column(db.String(10))  # 'red', 'blue', or 'unknown' 