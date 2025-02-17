from models import db, Game, GamePlayer, IdentityGuess, GameRound
from sqlalchemy import and_, func
from flask import Flask
from config import Config

class GameAnalyzer:
    def __init__(self):
        self.db = db

    def analyze_games(self, start_id: int, end_id: int):
        """分析指定ID范围内的游戏数据"""
        games = Game.query.filter(
            and_(
                Game.id >= start_id,
                Game.id <= end_id,
                Game.winner.isnot(None)  # 确保游戏已完成
            )
        ).all()

        if not games:
            return {"error": "No games found in the specified range"}

        total_games = len(games)
        
        # 统计完全猜对的局数
        perfect_guess_games = 0
        
        # 获取所有蓝方胜利的游戏
        blue_win_games = [game for game in games if game.winner == "blue"]
        
        for game in blue_win_games:
            # 获取这局游戏的红方玩家ID
            red_players = set(gp.player_id for gp in GamePlayer.query.filter_by(
                game_id=game.id, 
                role="red"
            ).all())
            
            # 获取蓝方玩家
            blue_players = GamePlayer.query.filter_by(
                game_id=game.id,
                role="blue"
            ).all()
            
            # 获取这局游戏的最后一轮
            last_round = GameRound.query.filter_by(
                game_id=game.id
            ).order_by(GameRound.round_number.desc()).first()
            
            if not last_round:
                continue
            
            # 检查每个蓝方玩家的猜测
            all_correct = True
            for blue_player in blue_players:
                # 获取该蓝方玩家的最终猜测
                guesses = IdentityGuess.query.filter_by(
                    guesser_id=blue_player.player_id,
                    round_id=last_round.id,  # 只看最后一轮的猜测
                    guessed_role="red"  # 使用正确的字段名
                ).all()
                
                print(f"Game {game.id}, Player {blue_player.player_id}, Round {last_round.id}")
                print(f"Found {len(guesses)} guesses: {[g.target_id for g in guesses]}")
                print(f"Actual red players: {red_players}")
                
                # 添加更多调试信息
                print(f"Querying with: guesser_id={blue_player.player_id}, round_id={last_round.id}")
                # 检查数据库中是否有任何猜测记录
                all_guesses = IdentityGuess.query.filter_by(round_id=last_round.id).all()
                print(f"All guesses in this round: {[(g.guesser_id, g.target_id, g.guessed_role) for g in all_guesses]}")
                
                # 如果没有足够的猜测记录，或猜测不完全正确
                guessed_reds = set(g.target_id for g in guesses)
                if len(guessed_reds) != 2 or guessed_reds != red_players:
                    print(f"Incorrect guess: {guessed_reds} != {red_players}")
                    all_correct = False
                    break
            
            if all_correct:
                print(f"Perfect guess in game {game.id}!")
                perfect_guess_games += 1

        # 统计阵营胜率
        team_stats = {
            "blue": {"wins": 0},
            "red": {"wins": 0}
        }

        # 统计每个玩家的表现
        player_stats = {
            f"P{i}": {
                "blue_wins": 0, "blue_games": 0,
                "red_wins": 0, "red_games": 0
            } for i in range(1, 6)
        }

        # 遍历每场游戏
        for game in games:
            winner = game.winner
            if winner:
                team_stats[winner]["wins"] += 1

            # 获取该游戏的所有玩家记录
            game_players = GamePlayer.query.filter_by(game_id=game.id).all()
            
            for gp in game_players:
                player_id = gp.player_id
                role = gp.role
                
                # 更新玩家统计
                if role == "blue":
                    player_stats[player_id]["blue_games"] += 1
                    if winner == "blue":
                        player_stats[player_id]["blue_wins"] += 1
                else:  # red
                    player_stats[player_id]["red_games"] += 1
                    if winner == "red":
                        player_stats[player_id]["red_wins"] += 1

        # 计算胜率
        results = {
            "game_range": f"{start_id}-{end_id}",
            "total_games": total_games,
            "perfect_guesses": {
                "count": perfect_guess_games,
                "total_blue_wins": len(blue_win_games),
                "percentage": round(perfect_guess_games / len(blue_win_games) * 100, 2) if blue_win_games else 0
            },
            "team_winrates": {
                "blue": {
                    "winrate": round(team_stats["blue"]["wins"] / total_games * 100, 2),
                    "wins": team_stats["blue"]["wins"],
                    "total": total_games
                },
                "red": {
                    "winrate": round(team_stats["red"]["wins"] / total_games * 100, 2),
                    "wins": team_stats["red"]["wins"],
                    "total": total_games
                }
            },
            "player_stats": {}
        }

        # 计算每个玩家的胜率
        for pid, stats in player_stats.items():
            results["player_stats"][pid] = {
                "blue": {
                    "winrate": round(stats["blue_wins"] / stats["blue_games"] * 100, 2) if stats["blue_games"] > 0 else 0,
                    "wins": stats["blue_wins"],
                    "games": stats["blue_games"]
                },
                "red": {
                    "winrate": round(stats["red_wins"] / stats["red_games"] * 100, 2) if stats["red_games"] > 0 else 0,
                    "wins": stats["red_wins"],
                    "games": stats["red_games"]
                },
                "overall": {
                    "winrate": round((stats["blue_wins"] + stats["red_wins"]) / 
                                   (stats["blue_games"] + stats["red_games"]) * 100, 2) 
                                   if (stats["blue_games"] + stats["red_games"]) > 0 else 0,
                    "total_games": stats["blue_games"] + stats["red_games"]
                }
            }

        return results

    def print_analysis(self, results):
        """打印分析结果"""
        print(f"\n=== 游戏分析报告 (游戏 {results['game_range']}) ===")
        print(f"总场次: {results['total_games']}")
        
        # 打印完全猜对的统计
        perfect = results['perfect_guesses']
        print(f"\n蓝方完全猜对统计:")
        print(f"在蓝方胜利的 {perfect['total_blue_wins']} 局中，有 {perfect['count']} 局完全猜对")
        print(f"完全猜对率: {perfect['percentage']}%")
        
        print("\n阵营胜率:")
        for team, stats in results['team_winrates'].items():
            print(f"{team.upper()}: {stats['winrate']}% ({stats['wins']}/{stats['total']})")

        print("\n玩家表现:")
        for pid, stats in results['player_stats'].items():
            print(f"\n{pid}:")
            print(f"  蓝方: {stats['blue']['winrate']}% ({stats['blue']['wins']}/{stats['blue']['games']})")
            print(f"  红方: {stats['red']['winrate']}% ({stats['red']['wins']}/{stats['red']['games']})")
            print(f"  总体: {stats['overall']['winrate']}% (共{stats['overall']['total_games']}场)")

def create_app():
    """创建一个临时的 Flask 应用来初始化数据库连接"""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 Avalon 游戏数据')
    parser.add_argument('start_id', type=int, help='起始游戏 ID')
    parser.add_argument('end_id', type=int, help='结束游戏 ID')
    args = parser.parse_args()
    
    app = create_app()
    with app.app_context():
        analyzer = GameAnalyzer()
        results = analyzer.analyze_games(args.start_id, args.end_id)
        analyzer.print_analysis(results) 