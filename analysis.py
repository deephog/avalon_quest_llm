from models import db, Game, GamePlayer, IdentityGuess, GameRound
from sqlalchemy import and_, func
from flask import Flask
from config import Config
import json
import matplotlib.pyplot as plt  # 新增
import numpy as np  # 新增

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

    def analyze_model_performance(self, start_id: int, end_id: int):
        """分析指定ID范围内每个位置(P1-P5)的表现"""
        # 查询时过滤掉winner为空的游戏
        games = Game.query.filter(
            Game.id >= start_id,
            Game.id <= end_id,
            Game.winner.isnot(None)
        ).all()
        
        # 打印总游戏数和有效游戏数
        total_games = Game.query.filter(
            Game.id >= start_id,
            Game.id <= end_id
        ).count()
        valid_games = len(games)
        print(f"\n分析范围: 游戏ID {start_id}-{end_id}")
        print(f"总游戏数: {total_games}, 有效游戏数: {valid_games}")
        
        # 初始化统计数据
        stats = {}
        for i in range(1, 6):
            stats[f"P{i}"] = {
                "total_games": 0,
                "total_wins": 0,
                "red_games": 0,
                "red_wins": 0,
                "blue_games": 0,
                "blue_wins": 0,
                "perfect_guesses": 0,  # 完全猜中次数
                "total_fails": 0,  # 个人不合规次数
                "total_api_calls": 0,  # 个人API调用次数
                # 新增加权表现值相关字段
                "expected_wins": 0,  # 期望胜利次数
                "performance_score": 0,  # 表现值
                "standard_score": 0,  # 标准分
                "selected_by_blue": 0,  # 被蓝方队长选中次数
                "selected_as_red": 0,   # 作为红方被蓝方队长选中次数
                "blue_leader_rounds": 0, # 蓝方担任队长的轮次总数
                "selected_by_red": 0,    # 作为蓝方被红方队长选中次数
                "red_leader_rounds": 0,  # 红方担任队长的轮次总数
                "red_failed_missions": 0,  # 作为红方参与的失败任务数
                "blue_leader_count": 0,     # 作为蓝方队长的次数
                "blue_leader_fails": 0,     # 作为蓝方队长时任务失败的次数
                "amulet_red_hits": 0,       # 作为蓝方队长时魔法指示物命中红方的次数
                "red_leader_count": 0,      # 作为红方队长的次数
                "red_epic_fails": 0,        # 作为红方队长时的致命失误次数
                "red_early_inactive": 0,     # 作为红方时前两轮未参与任务的次数
                "red_early_suspects": 0,     # 前两轮未参与时被蓝方玩家猜测为红方的总人次
                "red_mistaken_as_blue": 0,  # 作为红方被误认为蓝方的总人轮次
                "red_identified_as_red": 0,  # 作为红方被对手正确识别的总人次
                "red_undetected": 0,  # 作为红方未被蓝方玩家识破的总人次（猜测为蓝方或unknown）
                "first_selected_total_rounds": 0,  # 首次被蓝方队长选中的总轮次
                "first_selected_games": 0,         # 被蓝方队长选中过的游戏数
                "red_first_leader_total_rounds": 0,  # 作为红方首次被蓝方选为队长的总轮次
                "red_first_leader_games": 0,         # 作为红方被蓝方选为队长的游戏数
                "first_leader_total_rounds": 0,  # 首次被蓝方选为队长的总轮次
                "first_leader_games": 0,         # 被蓝方选为队长的游戏数
                "morgan_leader_count": 0,      # 作为摩根担任队长的次数
                "morgan_self_amulet": 0,       # 作为摩根队长给自己魔法指示物的次数
                "total_leader_count": 0,       # 作为队长的总次数
                "total_self_amulet": 0,        # 作为队长给自己魔法指示物的总次数
                "red_nonmorgan_leader": 0,     # 作为红方非摩根队长的次数
                "red_nonmorgan_self": 0,       # 作为红方非摩根队长给自己魔法的次数
                "red_leader_success": 0,  # 作为红方队长任务成功的次数
                "actual_leader_total_rounds": 0,  # 实际被选为队长的总轮次
                "actual_leader_games": 0,         # 实际被选为队长的游戏数
                "red_teammates": {},  # 新增：初始化红方队友统计
                "morgan": 0,         # 记录摩根次数
                "red_nonstart_leader_rounds": 0,  # 新增：非首轮队长的轮次总和
                "red_nonstart_leader_count": 0,   # 新增：非首轮队长次数
                "red_nonstart_games": 0,          # 新增：非首轮队长的游戏数
                "red_second_round_leader_from_blue": 0,  # 新增：由蓝方交接的第二轮队长次数
                "red_third_round_leader_from_blue": 0,   # 新增：由蓝方交接的第三轮队长次数
                "blue_correctly_identified": 0,  # 作为蓝方被其他蓝方正确识别的总人次
                "blue_correctly_identified_round": 0,  # 作为蓝方被其他蓝方正确识别的总人轮次
                "total_blue_players_sum": 0,     # 新增：统计遇到的蓝方玩家总数
                "total_blue_players_games": 0,   # 新增：统计有效局数
                "red_leader_count": 0,  # 作为红方时成为队长的次数
                "blue_leader_count": 0,  # 作为蓝方时成为队长的次数
                "red_first_round_leader": 0,  # 作为红方时第一轮随机到队长的次数
                "blue_first_round_leader": 0,  # 作为蓝方时第一轮随机到队长的次数
                "red_three_success": 0,  # 作为红方时赢得3次任务的局数
                "red_three_success_failed": 0,  # 作为红方时赢得3次任务但最终被翻盘的局数
                # 为每个统计指标添加队友维度
                "red_mistaken_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},  # 每个队友情况下被误认为蓝方的次数
                "red_identified_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},  # 每个队友情况下被对手正确识别为红方的次数
                "blue_correctly_identified_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},  # 每个队友情况下被其他蓝方正确识别为蓝方的次数
                "red_games_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},     # 与每个队友一起玩的红方局数
                "red_leader_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},    # 每个队友情况下成为队长的次数
                "red_first_leader_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},  # 每个队友情况下第一轮随机到队长的次数
                "red_three_success_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},  # 每个队友情况下赢得3次任务的次数
                "red_three_success_failed_by_teammate": {f"P{j}": 0 for j in range(1, 6) if j != i},  # 每个队友情况下被翻盘的次数
            }
        
        # 统计每个位置的表现
        for game in games:
            winner = game.winner
            players = GamePlayer.query.filter_by(game_id=game.id).all()
            
            # 获取本局轮数
            rounds_count = GameRound.query.filter_by(game_id=game.id).count()
            
            # 获取红方玩家ID列表
            red_players = {p.player_id for p in players if p.role == "red"}
            
            # 获取本局最后一轮
            last_round = (GameRound.query
                .filter_by(game_id=game.id)
                .order_by(GameRound.id.desc())
                .first())
            
            # 处理猜测数据
            guesses = {}
            if last_round:
                all_guesses = IdentityGuess.query.filter_by(round_id=last_round.id).all()
                for guess in all_guesses:
                    if guess.guesser_id not in guesses:
                        guesses[guess.guesser_id] = {}
                    guesses[guess.guesser_id][guess.target_id] = guess.guessed_role
            
            # 记录本局每个玩家是否已被选中
            selected_in_game = {pid: False for pid in stats}
            current_round = 0
            
            # 获取每一轮的数据
            rounds = GameRound.query.filter_by(game_id=game.id).order_by(GameRound.id).all()
            for game_round in rounds:
                current_round += 1
                leader_id = game_round.leader_id
                leader_player = next((p for p in players if p.player_id == leader_id), None)
                if not leader_player:
                    continue
                
                # 获取队员信息（共用）
                raw_team_members = game_round.team_members.split('_')[0] if game_round.team_members else ""
                team_members = [m.strip() for m in raw_team_members.split(',')] if raw_team_members else []
                
                # 部分1: 统计被蓝方队长选中的情况
                if leader_player.role == "blue":
                    # 统计首次被选中的轮次
                    for member_id in team_members:
                        if member_id == leader_id:
                            continue
                        if not selected_in_game[member_id]:
                            selected_in_game[member_id] = True
                            stats[member_id]["first_selected_total_rounds"] += current_round
                            stats[member_id]["first_selected_games"] += 1
                
                # 部分2: 统计队长相关数据（无论红蓝方）
                stats[leader_id]["total_leader_count"] += 1
                
                if '_' in game_round.team_members:
                    amulet_target = game_round.team_members.split('_')[1].strip()
                    if amulet_target == leader_id:  # 给自己魔法指示物
                        stats[leader_id]["total_self_amulet"] += 1
                        
                        if leader_player.morgan:  # 是摩根
                            stats[leader_id]["morgan_leader_count"] += 1
                            stats[leader_id]["morgan_self_amulet"] += 1
                        elif leader_player.role == "red":  # 是红方非摩根
                            stats[leader_id]["red_nonmorgan_leader"] += 1
                            stats[leader_id]["red_nonmorgan_self"] += 1
                    else:  # 没给自己魔法指示物
                        if leader_player.morgan:
                            stats[leader_id]["morgan_leader_count"] += 1
                        elif leader_player.role == "red":
                            stats[leader_id]["red_nonmorgan_leader"] += 1
                
                if leader_player.role == "blue":
                    # 统计蓝方队长次数
                    stats[leader_id]["blue_leader_count"] += 1
                    
                    # 更新所有玩家的蓝方队长轮次计数
                    for pid in stats:
                        stats[pid]["blue_leader_rounds"] += 1
                    
                    # 统计被选中情况
                    for member_id in team_members:
                        if member_id == leader_id:
                            continue  # 不统计队长自己
                        
                        member_player = next((p for p in players if p.player_id == member_id), None)
                        if not member_player:
                            continue
                        
                        stats[member_id]["selected_by_blue"] += 1
                        if member_player.role == "red":  # 如果被选中的是红方玩家
                            stats[member_id]["selected_as_red"] += 1
                    
                    # 检查魔法指示物使用情况
                    if '_' in game_round.team_members:
                        amulet_target = game_round.team_members.split('_')[1].strip()
                        target_player = next((p for p in players if p.player_id == amulet_target), None)
                        
                        # 如果目标是红方，记录命中
                        if target_player and target_player.role == "red":
                            stats[leader_id]["amulet_red_hits"] += 1
                    
                    # 统计任务失败
                    if game_round.result == "fail":
                        stats[leader_id]["blue_leader_fails"] += 1
                else:  # 红方队长
                    # 统计红方队长次数
                    stats[leader_id]["red_leader_count"] += 1
                    
                    # 统计任务成功的情况
                    if game_round.result == "success":
                        stats[leader_id]["red_leader_success"] += 1
                    
                    # 更新所有玩家的红方队长轮次计数
                    for pid in stats:
                        stats[pid]["red_leader_rounds"] += 1
                    
                    # 统计被选中情况
                    for member_id in team_members:
                        if member_id == leader_id:
                            continue  # 不统计队长自己
                        
                        member_player = next((p for p in players if p.player_id == member_id), None)
                        if not member_player:
                            continue
                        
                        # 如果是蓝方玩家，记录被红方队长选中
                        if member_player.role == "blue":
                            stats[member_id]["selected_by_red"] += 1
                    
                    # 检查是否发生致命失误
                    if '_' in game_round.team_members:
                        raw_team_members = game_round.team_members.split('_')[0]
                        team_members = [m.strip() for m in raw_team_members.split(',')] if raw_team_members else []
                        amulet_target = game_round.team_members.split('_')[1].strip()
                        
                        # 检查是否选择了红方队友
                        has_red_teammate = False
                        for member_id in team_members:
                            if member_id == leader_id:
                                continue
                            member_player = next((p for p in players if p.player_id == member_id), None)
                            if member_player and member_player.role == "red":
                                has_red_teammate = True
                                break
                        
                        # 检查魔法指示物是否给了摩根
                        amulet_player = next((p for p in players if p.player_id == amulet_target), None)
                        gave_amulet_to_morgan = amulet_player and amulet_player.morgan
                        
                        # 如果两个条件都满足，记录致命失误
                        if has_red_teammate and gave_amulet_to_morgan:
                            stats[leader_id]["red_epic_fails"] += 1
            
            # 处理每个玩家的基本统计
            for player in players:
                pid = player.player_id
                role = player.role
                fails = player.user_id or 0
                
                # 更新总场次和胜负
                stats[pid]["total_games"] += 1
                if role == winner:
                    stats[pid]["total_wins"] += 1
                
                # 更新阵营场次
                if role == "red":
                    stats[pid]["red_games"] += 1
                    if winner == "red":
                        stats[pid]["red_wins"] += 1
                else:
                    stats[pid]["blue_games"] += 1
                    if winner == "blue":
                        stats[pid]["blue_wins"] += 1
                
                # 如果是蓝方玩家，检查猜测准确性
                if role == "blue" and pid in guesses:
                    player_red_guesses = {
                        target_id for target_id, guessed_role in guesses[pid].items()
                        if guessed_role.lower() == "red"
                    }
                    if player_red_guesses == red_players:  # 两个红方都猜中
                        stats[pid]["perfect_guesses"] += 1
                
                # 更新个人统计
                stats[pid]["total_fails"] += fails
                stats[pid]["total_api_calls"] += (rounds_count + 1)
                
            
            # 获取每一轮的数据并统计
            rounds = GameRound.query.filter_by(game_id=game.id).order_by(GameRound.id).all()
            for game_round in rounds:
                # 只统计失败的任务
                if game_round.result != "fail":
                    continue
                
                # 获取该轮队员信息
                raw_team_members = game_round.team_members.split('_')[0] if game_round.team_members else ""
                team_members = [m.strip() for m in raw_team_members.split(',')] if raw_team_members else []
                
                # 统计参与失败任务的红方玩家
                for member_id in team_members:
                    member_player = next((p for p in players if p.player_id == member_id), None)
                    if not member_player or member_player.role != "red":
                        continue
                    
                    stats[member_id]["red_failed_missions"] += 1
            
            # 获取前两轮数据
            early_rounds = (GameRound.query
                .filter_by(game_id=game.id)
                .order_by(GameRound.id)
                .limit(2)
                .all())
            
            if len(early_rounds) < 2:  # 跳过不足两轮的游戏
                continue
            
            # 记录每个红方玩家在前两轮的参与情况
            for player in players:
                if player.role != "red":
                    continue
                
                # 检查是否参与前两轮
                participated = False
                for round in early_rounds:
                    if player.player_id == round.leader_id:  # 是队长
                        participated = True
                        break
                        
                    # 检查是否是队员
                    raw_team_members = round.team_members.split('_')[0] if round.team_members else ""
                    team_members = [m.strip() for m in raw_team_members.split(',')] if raw_team_members else []
                    if player.player_id in team_members:  # 是队员
                        participated = True
                        break
                
                if not participated:  # 前两轮都没参与
                    stats[player.player_id]["red_early_inactive"] += 1
                    
                    # 获取第二轮的猜测数据
                    round2_guesses = IdentityGuess.query.filter_by(round_id=early_rounds[1].id).all()
                    suspect_count = 0  # 记录猜测为红方的蓝方玩家数量
                    
                    # 检查每个蓝方玩家的猜测
                    for guesser in players:
                        if guesser.role != "blue":  # 只看蓝方玩家的猜测
                            continue
                            
                        # 找到这个蓝方玩家对目标玩家的猜测
                        player_guess = next(
                            (g for g in round2_guesses if g.guesser_id == guesser.player_id and g.target_id == player.player_id), 
                            None
                        )
                        
                        # 如果明确猜测为红方，增加计数
                        if player_guess and player_guess.guessed_role.lower() == "red":
                            suspect_count += 1
                    
                    # 累加被怀疑次数
                    stats[player.player_id]["red_early_suspects"] += suspect_count
            
            # 记录本局每个玩家是否已当过队长
            been_leader = {pid: False for pid in stats}
            current_round = 0
            
            # 获取每一轮的数据
            rounds = GameRound.query.filter_by(game_id=game.id).order_by(GameRound.id).all()
            for game_round in rounds:
                current_round += 1
                if current_round == 1:  # 跳过第一轮
                    continue
                
                # 获取上一轮的队长（选择者）
                prev_round = rounds[current_round - 2]  # current_round从1开始，所以这里是-2
                prev_leader = next((p for p in players if p.player_id == prev_round.leader_id), None)
                
                # 只统计蓝方队长选择的情况
                if not prev_leader or prev_leader.role != "blue":
                    continue
                
                # 获取当前轮的队长（被选择者）
                current_leader = next((p for p in players if p.player_id == game_round.leader_id), None)
                if not current_leader:
                    continue
                
                # 如果是该玩家首次被选为队长
                if not been_leader[current_leader.player_id]:
                    been_leader[current_leader.player_id] = True
                    stats[current_leader.player_id]["first_leader_total_rounds"] += current_round
                    stats[current_leader.player_id]["first_leader_games"] += 1
                    # 记录实际被选为队长的情况
                    stats[current_leader.player_id]["actual_leader_total_rounds"] += current_round
                    stats[current_leader.player_id]["actual_leader_games"] += 1
            
            # 如果是红方玩家，统计队友
            for player in players:
                pid = player.player_id
                if player.role == "red":
                    # 找出该局游戏中的其他红方玩家
                    red_teammates = [p for p in players if p.role == "red" and p.player_id != pid]

                    for teammate in red_teammates:
                        if teammate.player_id not in stats[pid]["red_teammates"]:
                            stats[pid]["red_teammates"][teammate.player_id] = 0
                        stats[pid]["red_teammates"][teammate.player_id] += 1
                    
                    if 'morgan' not in stats[pid]:
                        stats[pid]["morgan"] = 0
                    if player.morgan:
                        stats[pid]["morgan"] += 1

            # 获取第一轮队长
            first_round_leader = rounds[0].leader_id if rounds else None
            
            # 记录每个红方玩家在这局是否已经当过队长
            red_leader_in_game = {p.player_id: False for p in players if p.role == "red"}
            
            # 遍历每一轮
            for round_num, game_round in enumerate(rounds, 1):
                leader_id = game_round.leader_id
                leader_player = next((p for p in players if p.player_id == leader_id), None)
                
                if not leader_player or leader_player.role != "red":
                    continue
                    
                # 如果是红方队长且不是第一轮
                if leader_id != first_round_leader:
                    stats[leader_id]["red_nonstart_leader_rounds"] += round_num
                    stats[leader_id]["red_nonstart_leader_count"] += 1
                    red_leader_in_game[leader_id] = True
                    
                    # 获取上一轮队长信息
                    if round_num > 1:
                        prev_round = rounds[round_num - 2]  # round_num从1开始，所以这里是-2
                        prev_leader = next((p for p in players if p.player_id == prev_round.leader_id), None)
                        
                        # 如果上一轮是蓝方队长
                        if prev_leader and prev_leader.role == "blue":
                            # 统计第二轮和第三轮队长
                            if round_num == 2:
                                stats[leader_id]["red_second_round_leader_from_blue"] += 1
                            elif round_num == 3:
                                stats[leader_id]["red_third_round_leader_from_blue"] += 1
            
            # 在每局结束时，更新非首轮队长的游戏计数
            for pid in red_leader_in_game:
                if pid != first_round_leader:
                    stats[pid]["red_nonstart_games"] += 1

            # 对每个蓝方玩家，统计被其他蓝方玩家正确识别的情况
            blue_team = [p.player_id for p in players if p.role == "blue"]
                
            # 遍历每个蓝方玩家  
            for player in players:
                if player.role != "blue":  # 只统计蓝方玩家
                    continue
                    
                correctly_identified = 0

                for game_round in rounds:
                    round_guesses = IdentityGuess.query.filter_by(round_id=game_round.id).all()
                # 检查每个其他蓝方玩家的猜测
                    for guesser in players:
                        if guesser.role != "blue" or guesser.player_id == player.player_id:  # 排除自己和非蓝方玩家
                            continue
                        
                        # 找到这个蓝方玩家对当前玩家的猜测
                        guess = next(
                            (g for g in round_guesses if g.guesser_id == guesser.player_id and g.target_id == player.player_id),
                            None
                        )               
                        # 如果明确猜测为蓝方，计入正确识别
                        if guess and guess.guessed_role.lower() == "blue":
                            correctly_identified += 1
                        
                        for blue_player in blue_team:
                            if (blue_player != player.player_id) and (blue_player != guesser.player_id):
                                stats[player.player_id]["blue_correctly_identified_by_teammate"][blue_player] += 1
                
                # 累加统计数据
                stats[player.player_id]["blue_correctly_identified"] += correctly_identified
            

            # 计算本局蓝方玩家数量
            blue_players_count = sum(1 for p in players if p.role == "blue")
            
            # 更新蓝方玩家数量统计
            if role == "red":
                stats[pid]["total_blue_players_sum"] += blue_players_count
                stats[pid]["total_blue_players_games"] += 1
            elif role == "blue":
                stats[pid]["total_blue_players_sum"] += (blue_players_count - 1)  # 减去自己
                stats[pid]["total_blue_players_games"] += 1

            # 对每个红方玩家，统计每一轮被误认为蓝方的情况
            for player in players:
                teammate = player.team_mates
                if player.role != "red":
                    continue
                
                # 遍历每一轮
                for game_round in rounds:
                    # 获取该轮的猜测数据
                    round_guesses = IdentityGuess.query.filter_by(round_id=game_round.id).all()
                    
                    # 检查每个蓝方玩家在这一轮的猜测
                    for guesser in players:
                        if guesser.role != "blue":  # 只看蓝方玩家的猜测
                            continue
                        
                        # 找到这个蓝方玩家在这一轮对该红方玩家的猜测
                        guess = next(
                            (g for g in round_guesses if g.guesser_id == guesser.player_id and g.target_id == player.player_id),
                            None
                        )
                        
                        # 如果在这一轮明确猜测为蓝方，计入误判
                        if guess and guess.guessed_role.lower() == "blue":
                            stats[player.player_id]["red_mistaken_as_blue"] += 1
                            stats[player.player_id]["red_mistaken_by_teammate"][teammate] += 1
                        if guess and guess.guessed_role.lower() == "red":
                            stats[player.player_id]["red_identified_as_red"] += 1
                            stats[player.player_id]["red_identified_by_teammate"][teammate] += 1
            # 获取第一轮数据
            if rounds:
                first_round = rounds[0]
                first_round_leader = next((p for p in players if p.player_id == first_round.leader_id), None)
                
                # 统计第一轮队长
                if first_round_leader:
                    if first_round_leader.role == "red":
                        stats[first_round_leader.player_id]["red_first_round_leader"] += 1
                    else:  # blue
                        stats[first_round_leader.player_id]["blue_first_round_leader"] += 1

            # 统计本局红方赢得的任务次数

            red_success_count = sum(1 for r in game.rounds if r.result == "fail")
            
            # 如果红方赢得3次或以上任务
            if red_success_count >= 3:
                # 统计每个红方玩家
                for player in players:
                    if player.role == "red":
                        stats[player.player_id]["red_three_success"] += 1
                        # 如果最终蓝方胜利，说明被翻盘
                        if game.winner == "blue":
                                stats[player.player_id]["red_three_success_failed"] += 1

        # 计算全局基准胜率
        total_red_games = sum(data["red_games"] for data in stats.values())
        total_red_wins = sum(data["red_wins"] for data in stats.values())
        total_blue_games = sum(data["blue_games"] for data in stats.values())
        total_blue_wins = sum(data["blue_wins"] for data in stats.values())
        
        red_base_winrate = total_red_wins / total_red_games if total_red_games > 0 else 0
        blue_base_winrate = total_blue_wins / total_blue_games if total_blue_games > 0 else 0
        
        # 计算每个玩家的加权表现值
        for pid, data in stats.items():
            # 计算期望胜利次数
            expected_red_wins = data["red_games"] * red_base_winrate
            expected_blue_wins = data["blue_games"] * blue_base_winrate
            data["expected_wins"] = expected_red_wins + expected_blue_wins
            
            # 计算实际表现值
            actual_wins = data["total_wins"]
            data["performance_score"] = actual_wins - data["expected_wins"]
            
            # 计算标准分
            total_games = data["total_games"]
            data["standard_score"] = (data["performance_score"] / total_games * 100) if total_games > 0 else 0
        
        return stats

    def _calculate_logic_stats(self, stats):
        """计算所有玩家的逻辑推理统计数据"""
        # 收集所有玩家的数据
        all_stats = {
            'guess_rates': [],
            'amulet_rates': [],
            'fail_rates': [],
            'red_success_rates': []
        }
        
        for data in stats.values():
            # 猜中红方率
            if data["blue_games"] > 0:
                guess_rate = (data["perfect_guesses"] / data["blue_games"] * 100)
                all_stats['guess_rates'].append(guess_rate)
                
            # 魔法命中率
            if data["blue_leader_count"] > 0:
                amulet_rate = (data["amulet_red_hits"] / data["blue_leader_count"] * 100)
                all_stats['amulet_rates'].append(amulet_rate)
                
            # 蓝方队长失败率
            if data["blue_leader_count"] > 0:
                fail_rate = (data["blue_leader_fails"] / data["blue_leader_count"] * 100)
                all_stats['fail_rates'].append(fail_rate)
                
            # 红方队长成功率
            if data["red_leader_count"] > 0:
                success_rate = (data["red_leader_success"] / data["red_leader_count"] * 100)
                all_stats['red_success_rates'].append(success_rate)
        
        # 计算平均值和标准差
        stats = {}
        for key, values in all_stats.items():
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                stats[key] = {'mean': mean, 'std': std}
            else:
                stats[key] = {'mean': 0, 'std': 1}  # 避免除以0
                
        return stats

    def _calculate_logic_score(self, player_stats, base_stats):
        """计算单个玩家的逻辑推理得分"""
        scores = {}
        
        # 猜中红方率
        if player_stats["blue_games"] > 0:
            guess_rate = (player_stats["perfect_guesses"] / player_stats["blue_games"] * 100)
            scores['guess'] = (guess_rate - base_stats['guess_rates']['mean']) / base_stats['guess_rates']['std']
        else:
            scores['guess'] = 0
            
        # 魔法命中率
        if player_stats["blue_leader_count"] > 0:
            amulet_rate = (player_stats["amulet_red_hits"] / player_stats["blue_leader_count"] * 100)
            scores['amulet'] = (amulet_rate - base_stats['amulet_rates']['mean']) / base_stats['amulet_rates']['std']
        else:
            scores['amulet'] = 0
            
        # 蓝方队长失败率（反指标）
        if player_stats["blue_leader_count"] > 0:
            fail_rate = (player_stats["blue_leader_fails"] / player_stats["blue_leader_count"] * 100)
            scores['fail'] = -(fail_rate - base_stats['fail_rates']['mean']) / base_stats['fail_rates']['std']
        else:
            scores['fail'] = 0
            
        # 红方队长成功率（反指标）
        if player_stats["red_leader_count"] > 0:
            success_rate = (player_stats["red_leader_success"] / player_stats["red_leader_count"] * 100)
            scores['red_success'] = -(success_rate - base_stats['red_success_rates']['mean']) / base_stats['red_success_rates']['std']
        else:
            scores['red_success'] = 0
            
        # 计算加权总分
        total_score = (scores['guess'] * 0.3 + 
                      scores['amulet'] * 0.3 + 
                      scores['fail'] * 0.3 + 
                      scores['red_success'] * 0.1)
                      
        return scores, total_score

    def _calculate_persuasion_stats(self, stats):
        """计算所有玩家的说服力/欺骗性统计数据"""
        all_stats = {
            'selection_rates': [],      # 1. 平均每几轮被带入一次
            'suspect_counts': [],       # 2. 前两轮未参与时被怀疑人数
            'fail_rates': [],          # 3. 作为红方参与失败任务(+1后)
            'undetected_counts': [],    # 4. 未被识破的人数
            'first_selected': [],       # 5. 首次被带入任务轮次
            'leader_rounds': []         # 6. 被选为队长的轮次
        }
        
        for data in stats.values():
            # 1. 平均被带入频率
            if data["blue_leader_rounds"] > 0 and data["selected_by_blue"] > 0:
                selection_rate = data["blue_leader_rounds"] / data["selected_by_blue"]
                all_stats['selection_rates'].append(selection_rate)
            
            # 2. 前两轮未参与被怀疑
            if data["red_early_inactive"] > 0:
                suspect_rate = data["red_early_suspects"] / data["red_early_inactive"]
                all_stats['suspect_counts'].append(suspect_rate)
            
            # 3. 红方参与失败任务
            if data["red_games"] > 0:
                fail_rate = (data["red_failed_missions"] / data["red_games"]) + 1
                all_stats['fail_rates'].append(fail_rate)
            
            # 4. 未被识破人数
            if data["red_games"] > 0:
                undetected = data["red_undetected"] / data["red_games"]
                all_stats['undetected_counts'].append(undetected)
            
            # 5. 首次被带入任务轮次
            if data["first_selected_games"] > 0:
                first_selected = data["first_selected_total_rounds"] / data["first_selected_games"]
                all_stats['first_selected'].append(first_selected)
            
            # 6. 被选为队长的轮次
            if data["actual_leader_games"] > 0:
                leader_round = data["actual_leader_total_rounds"] / data["actual_leader_games"]
                all_stats['leader_rounds'].append(leader_round)
        
        # 计算平均值和标准差
        stats = {}
        for key, values in all_stats.items():
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                stats[key] = {'mean': mean, 'std': std}
            else:
                stats[key] = {'mean': 0, 'std': 1}
            
        return stats

    def _calculate_api_stats(self, stats):
        """计算所有玩家的API稳定性统计数据"""
        all_stats = {
            'error_rates': []  # 每多少次API调用产生一次不合规
        }
        
        for data in stats.values():
            if data["total_fails"] > 0:  # 只统计有不合规记录的情况
                error_rate = data["total_api_calls"] / data["total_fails"]
                all_stats['error_rates'].append(error_rate)
        
        # 计算平均值和标准差
        stats = {}
        if all_stats['error_rates']:
            mean = sum(all_stats['error_rates']) / len(all_stats['error_rates'])
            std = (sum((x - mean) ** 2 for x in all_stats['error_rates']) / len(all_stats['error_rates'])) ** 0.5
            stats['error_rates'] = {'mean': mean, 'std': std}
        else:
            stats['error_rates'] = {'mean': 0, 'std': 1}
        
        return stats

    def _calculate_winrate_stats(self, stats):
        """计算所有玩家的胜率统计数据"""
        all_stats = {
            'blue_winrates': [],  # 蓝方胜率
            'red_winrates': [],   # 红方胜率
            'total_winrates': []  # 总体胜率（加权）
        }
        
        for data in stats.values():
            # 蓝方胜率
            if data["blue_games"] > 0:
                blue_wr = (data["blue_wins"] / data["blue_games"] * 100)
                all_stats['blue_winrates'].append(blue_wr)
            
            # 红方胜率
            if data["red_games"] > 0:
                red_wr = (data["red_wins"] / data["red_games"] * 100)
                all_stats['red_winrates'].append(red_wr)
            
            # 总体胜率（根据实际参与局数加权）
            if data["total_games"] > 0:
                total_wr = (data["total_wins"] / data["total_games"] * 100)
                all_stats['total_winrates'].append(total_wr)
        
        # 计算平均值和标准差
        stats = {}
        for key, values in all_stats.items():
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                stats[key] = {'mean': mean, 'std': std}
            else:
                stats[key] = {'mean': 0, 'std': 1}
            
        return stats

    def print_model_analysis(self, stats):
        """打印模型表现分析结果"""
        print("\n=== 模型表现分析 ===")
        
        # 计算全局基准胜率
        total_red_games = sum(data["red_games"] for data in stats.values())
        total_red_wins = sum(data["red_wins"] for data in stats.values())
        total_blue_games = sum(data["blue_games"] for data in stats.values())
        total_blue_wins = sum(data["blue_wins"] for data in stats.values())
        
        red_base_winrate = total_red_wins / total_red_games * 100 if total_red_games > 0 else 0
        blue_base_winrate = total_blue_wins / total_blue_games * 100 if total_blue_games > 0 else 0
        
        print(f"\n基准胜率:")
        print(f"  红方: {red_base_winrate:.1f}%")
        print(f"  蓝方: {blue_base_winrate:.1f}%")
        print(f"  阵营差异: {abs(blue_base_winrate - red_base_winrate):.1f}%")
        
        # 按标准分排序
        sorted_players = sorted(
            stats.items(),
            key=lambda x: x[1]["standard_score"],
            reverse=True
        )
        
        # 计算基准统计数据
        base_stats = self._calculate_logic_stats(stats)
        persuasion_stats = self._calculate_persuasion_stats(stats)
        api_stats = self._calculate_api_stats(stats)
        winrate_stats = self._calculate_winrate_stats(stats)  # 添加胜率统计
        
        for pid, data in sorted_players:
            red_wr = (data["red_wins"] / data["red_games"] * 100) if data["red_games"] > 0 else 0
            blue_wr = (data["blue_wins"] / data["blue_games"] * 100) if data["blue_games"] > 0 else 0
            guess_rate = (data["perfect_guesses"] / data["blue_games"] * 100) if data["blue_games"] > 0 else 0
            
            # 计算个人的API调用错误率
            avg_fails = data["total_fails"] / data["total_games"] if data["total_games"] > 0 else 0
            
            # 计算平均每局未被识破和误判人数（使用所有红方局数）
            avg_mistaken = (data["red_mistaken_as_blue"] / data["red_games"]) if data["red_games"] > 0 else 0
            
            # 修改平均首次被选中轮次的计算
            # 如果从未被选中，就当作是第5轮
            if data["first_selected_games"] == 0:
                data["first_selected_total_rounds"] = 5  # 设为第5轮
                data["first_selected_games"] = 1         # 设为1次，避免除以0
            
            # 修改平均首次被选为队长轮次的计算
            # 如果从未被选为队长，就当作是第5轮
            if data["first_leader_games"] == 0:
                data["first_leader_total_rounds"] = 5  # 设为第5轮
                data["first_leader_games"] = 1         # 设为1次，避免除以0
            
            print(f"\n位置 {pid}:")
            print(f"  总体胜率表现：")
            
            winrate_scores = {}  # 用于存储胜率得分
            
            # 计算并显示红方胜率
            if data["red_games"] > 0:
                red_wr = (data["red_wins"] / data["red_games"] * 100)
                red_score = (red_wr - winrate_stats['red_winrates']['mean']) / winrate_stats['red_winrates']['std']
                winrate_scores['red'] = red_score
                print(f"  红方胜率: {red_wr:.1f}% ({data['red_wins']}/{data['red_games']}), {red_score:+.2f}σ")
            
            # 计算并显示蓝方胜率
            if data["blue_games"] > 0:
                blue_wr = (data["blue_wins"] / data["blue_games"] * 100)
                blue_score = (blue_wr - winrate_stats['blue_winrates']['mean']) / winrate_stats['blue_winrates']['std']
                winrate_scores['blue'] = blue_score
                print(f"  蓝方胜率: {blue_wr:.1f}% ({data['blue_wins']}/{data['blue_games']}), {blue_score:+.2f}σ")
            
            # 计算并显示胜率总体得分
            if winrate_scores:
                winrate_total_score = sum(winrate_scores.values()) / len(winrate_scores.values())
                print(f"  胜率总体得分: {winrate_total_score:+.2f}σ")
            
            print(f"  \n  逻辑推理表现：")
            scores, total_score = self._calculate_logic_score(data, base_stats)
            
            if data["blue_games"] > 0:
                guess_rate = (data["perfect_guesses"] / data["blue_games"] * 100)
                print(f"  作为蓝方玩家猜中两名红方: {guess_rate:.1f}% ({data['perfect_guesses']}/{data['blue_games']}), {scores['guess']:+.2f}σ")
            
            if data["blue_leader_count"] > 0:
                amulet_rate = (data["amulet_red_hits"] / data["blue_leader_count"] * 100)
                print(f"  魔法指示物命中红方: {data['amulet_red_hits']}/{data['blue_leader_count']}次 ({amulet_rate:.1f}%), {scores['amulet']:+.2f}σ")
                
                fail_rate = (data["blue_leader_fails"] / data["blue_leader_count"] * 100)
                print(f"  作为蓝方队长任务失败: {data['blue_leader_fails']}/{data['blue_leader_count']}次 ({fail_rate:.1f}%), {scores['fail']:+.2f}σ")
            else:
                print("  从未担任蓝方队长")
            
            if data["red_leader_count"] > 0:
                success_rate = (data["red_leader_success"] / data["red_leader_count"] * 100)
                print(f"  作为红方队长任务成功: {data['red_leader_success']}/{data['red_leader_count']}次 ({success_rate:.1f}%), {scores['red_success']:+.2f}σ")
            
            print(f"  逻辑推理总分: {total_score:+.2f}σ")

            print(f"  \n  说服力/欺骗性表现：")
            scores = {}
            
            # 1. 平均被带入频率（显示原始偏差，越大表示被带入频率低）
            if data["blue_leader_rounds"] > 0 and data["selected_by_blue"] > 0:
                selection_rate = data["blue_leader_rounds"] / data["selected_by_blue"]
                raw_score = (selection_rate - persuasion_stats['selection_rates']['mean']) / persuasion_stats['selection_rates']['std']
                scores['selection'] = -raw_score  # 在计算总分时反转
                print(f"  平均每{selection_rate:.1f}轮被蓝方队长带入一次, {raw_score:+.2f}σ")
            
            # 2. 前两轮未参与被怀疑（显示原始偏差，越大表示被怀疑的人多）
            if data["red_early_inactive"] > 0:
                suspect_rate = data["red_early_suspects"] / data["red_early_inactive"]
                raw_score = (suspect_rate - persuasion_stats['suspect_counts']['mean']) / persuasion_stats['suspect_counts']['std']
                scores['suspect'] = -raw_score  # 在计算总分时反转
                print(f"  作为红方前两轮未参与时平均被{suspect_rate:.1f}个蓝方玩家怀疑, {raw_score:+.2f}σ")
            
            # 3. 红方参与失败任务（显示原始偏差，越大表示失败次数多）
            if data["red_games"] > 0:
                fail_rate = (data["red_failed_missions"] / data["red_games"]) + 1
                raw_score = (fail_rate - persuasion_stats['fail_rates']['mean']) / persuasion_stats['fail_rates']['std']
                scores['fail'] = raw_score  # 正向指标，不需要反转
                print(f"  作为红方参与失败任务: 总计{data['red_failed_missions']}次, 平均每局{fail_rate:.2f}次, {raw_score:+.2f}σ")
            
            # # 4. 未被识破人数（显示原始偏差，越大表示未被识破的人多）
            # if data["red_games"] > 0:
            #     undetected = data["red_undetected"] / data["red_games"]
            #     raw_score = (undetected - persuasion_stats['undetected_counts']['mean']) / persuasion_stats['undetected_counts']['std']
            #     scores['undetected'] = raw_score  # 正向指标，不需要反转
            #     print(f"  作为红方平均每局有{undetected:.1f}个蓝方玩家未识破身份, {raw_score:+.2f}σ")
            
            # 5. 首次被带入任务轮次（显示原始偏差，越大表示轮次晚）
            if data["first_selected_games"] > 0:
                first_selected = data["first_selected_total_rounds"] / data["first_selected_games"]
                raw_score = (first_selected - persuasion_stats['first_selected']['mean']) / persuasion_stats['first_selected']['std']
                scores['first_selected'] = -raw_score  # 在计算总分时反转
                print(f"  平均在第{first_selected:.1f}轮首次被蓝方队长带入任务, {raw_score:+.2f}σ")
            
            # 6. 被选为队长的轮次（显示原始偏差，越大表示轮次晚）
            if data["actual_leader_games"] > 0:
                leader_round = data["actual_leader_total_rounds"] / data["actual_leader_games"]
                raw_score = (leader_round - persuasion_stats['leader_rounds']['mean']) / persuasion_stats['leader_rounds']['std']
                scores['leader'] = -raw_score  # 在计算总分时反转
                print(f"  被选为队长时平均在第{leader_round:.1f}轮, {raw_score:+.2f}σ")
            
            # 计算总分（考虑指标方向）
            valid_scores = [score for score in scores.values() if score != 0]
            if valid_scores:
                total_score = sum(valid_scores) / len(valid_scores)
                print(f"  说服力/欺骗性总分: {total_score:+.2f}σ")
            
            # API稳定性表现
            print(f"  \n  API稳定性表现：")
            print(f"  输出不合规统计: 总计{data['total_fails']}次, 平均每局{avg_fails:.2f}次")

            # 计算API稳定性得分
            api_scores = {}
            if data["total_fails"] > 0:
                error_rate = data["total_api_calls"] / data["total_fails"]
                raw_score = (error_rate - api_stats['error_rates']['mean']) / api_stats['error_rates']['std']
                api_scores['error_rate'] = raw_score  # 正向指标，不需要反转
                print(f"  平均每{error_rate:.1f}次API调用出现一次不合规, {raw_score:+.2f}σ")
            else:
                print("  从未出现不合规输出")

            # 计算API稳定性总分
            if api_scores:
                api_total_score = sum(api_scores.values()) / len(api_scores.values())
                print(f"  API稳定性总分: {api_total_score:+.2f}σ")

            # 在所有计算和打印完成后，收集四个维度的总分
            dimension_scores = {}
            
            # 1. 胜率总分
            if winrate_scores:
                dimension_scores['winrate_total'] = winrate_total_score
            
            # 2. 逻辑推理总分
            dimension_scores['logic_total'] = total_score  # 来自 _calculate_logic_score
            
            # 3. 说服力总分
            if valid_scores:
                dimension_scores['persuasion_total'] = total_score  # 来自说服力计算
            
            # 4. API稳定性总分
            if api_scores:
                dimension_scores['api_total'] = api_total_score  # 来自API稳定性计算
            
            # 在所有打印完成后绘制雷达图
            self.plot_player_radar(pid, dimension_scores)

            # 在所有现有输出后添加红方队友统计
            print("\n  红方队友统计:")
            teammates = data["red_teammates"]
            morgan = data["morgan"]
            if teammates:
                # 按次数排序
                sorted_teammates = sorted(teammates.items(), key=lambda x: x[1], reverse=True)
                for teammate_id, count in sorted_teammates:
                    print(f"    • 与 {teammate_id} 同队 {count} 次")
            else:
                print("    从未作为红方")
            
            if morgan:
                print(f"    • 作为红方时，有 {morgan} 次随机到Morgan")

            # 在红方队友统计后添加非首轮队长统计
            if data["red_nonstart_games"] > 0:
                leader_count = data["red_nonstart_leader_count"]
                avg_round = data["red_nonstart_leader_rounds"] / leader_count if leader_count > 0 else 0
                leader_rate = (leader_count / data["red_nonstart_games"]) * 100
                print(f"    • 作为红方时，除去第一轮随机为队长的局，在剩下的 {data['red_nonstart_games']} 局中:")
                if leader_count > 0:
                    print(f"      - 成为队长 {leader_count} 次 ({leader_rate:.1f}%)")
                    print(f"      - 平均在第 {avg_round:.1f} 轮成为队长")
                    # 显示由蓝方交接的第二轮和第三轮队长统计
                    second_round_rate = (data["red_second_round_leader_from_blue"] / data["red_nonstart_games"]) * 100
                    third_round_rate = (data["red_third_round_leader_from_blue"] / data["red_nonstart_games"]) * 100
                    print(f"      - 由蓝方交接的第二轮队长 {data['red_second_round_leader_from_blue']} 次 ({second_round_rate:.1f}%)")
                    print(f"      - 由蓝方交接的第三轮队长 {data['red_third_round_leader_from_blue']} 次 ({third_round_rate:.1f}%)")
                else:
                    print("      - 从未成为队长")

            # 在红方队友统计前添加身份识别统计
            if data["red_games"] > 0:
                avg_mistaken = data["red_mistaken_as_blue"] / data["red_games"]
                avg_identified = data["red_identified_as_red"] / data["red_games"]
                red_leader_rate = (data["red_leader_count"] / data["red_games"]) * 100
                red_first_leader_rate = (data["red_first_round_leader"] / data["red_games"]) * 100
                
                # 计算除去第一轮随机到队长后的队长率
                remaining_red_games = data["red_games"] - data["red_first_round_leader"]
                remaining_red_leader = data["red_leader_count"] - data["red_first_round_leader"]
                red_remaining_rate = (remaining_red_leader / remaining_red_games * 100) if remaining_red_games > 0 else 0
                
                print(f"    • 作为红方时，平均每局被蓝方玩家误认为蓝方{avg_mistaken:.1f}人次 (总计 {data['red_mistaken_as_blue']} 人次)")
                 # 添加每个队友的分项统计
                for teammate, mistaken_count in data["red_mistaken_by_teammate"].items():
                    if teammates[teammate] > 0:  # 确保和这个队友一起玩过
                        avg = mistaken_count / teammates[teammate]
                        print(f"      - 与{teammate}为队友时: {avg:.1f}人次 (总计 {mistaken_count} 人次)")

                print(f"    • 作为红方时，平均每局被蓝方玩家准确识别为红方{avg_identified:.1f}人次 (总计 {data['red_identified_as_red']} 人次)")
                # 添加每个队友的分项统计
                for teammate, identified_count in data["red_identified_by_teammate"].items():
                    if teammates[teammate] > 0:  # 确保和这个队友一起玩过
                        avg = identified_count / teammates[teammate]
                        print(f"      - 与{teammate}为队友时: {avg:.1f}人次 (总计 {identified_count} 人次)")

                print(f"    • 作为红方时，成为队长 {data['red_leader_count']} 次，队长率 {red_leader_rate:.1f}%")
                print(f"    • 作为红方时，第一轮随机到队长 {data['red_first_round_leader']} 次，随机率 {red_first_leader_rate:.1f}%")
                print(f"    • 作为红方时，除去第一轮随机到队长的局，在剩余 {remaining_red_games} 局中成为队长 {remaining_red_leader} 次，队长率 {red_remaining_rate:.1f}%")
                
               
            
            if data["blue_games"] > 0:
                avg_identified = data["blue_correctly_identified"] / data["blue_games"]
                blue_leader_rate = (data["blue_leader_count"] / data["blue_games"]) * 100
                blue_first_leader_rate = (data["blue_first_round_leader"] / data["blue_games"]) * 100
                
                # 计算除去第一轮随机到队长后的队长率
                remaining_blue_games = data["blue_games"] - data["blue_first_round_leader"]
                remaining_blue_leader = data["blue_leader_count"] - data["blue_first_round_leader"]
                blue_remaining_rate = (remaining_blue_leader / remaining_blue_games * 100) if remaining_blue_games > 0 else 0
                
                print(f"    • 作为蓝方时，平均每局被其他蓝方玩家正确识别为蓝方{avg_identified:.1f}人次 "
                      f"(总计 {data['blue_correctly_identified']} 人次)")
                # 添加每个队友的分项统计
                for teammate, identified_count in data["blue_correctly_identified_by_teammate"].items():
                    #avg = identified_count / teammates[teammate]
                    print(f"      - 与{teammate}为队友时: 总计 {identified_count} 人次")
                print(f"    • 作为蓝方时，成为队长 {data['blue_leader_count']} 次，队长率 {blue_leader_rate:.1f}%")
                print(f"    • 作为蓝方时，第一轮随机到队长 {data['blue_first_round_leader']} 次，随机率 {blue_first_leader_rate:.1f}%")
                print(f"    • 作为蓝方时，除去第一轮随机到队长的局，在剩余 {remaining_blue_games} 局中成为队长 {remaining_blue_leader} 次，队长率 {blue_remaining_rate:.1f}%")

            # 添加红方3次任务胜利和翻盘统计
            if data["red_three_success"] > 0:
                flip_rate = (data["red_three_success_failed"] / data["red_three_success"] * 100)
                print(f"    • 作为红方时，赢得3次任务 {data['red_three_success']} 次，"
                      f"其中 {data['red_three_success_failed']} 次被蓝方识破翻盘 (翻盘率 {flip_rate:.1f}%)")

    def _get_level_description(self, score: float) -> str:
        """根据标准分获取水平描述"""
        if score > 10:
            return "显著高于平均水平"
        elif score > 5:
            return "中上水平"
        elif score > -5:
            return "接近平均水平"
        elif score > -10:
            return "略低于平均水平"
        else:
            return "明显低于平均水平"

    def plot_player_radar(self, pid: str, scores: dict):
        """为单个玩家绘制雷达图"""
        # 使用更基础的字体设置
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 只使用通用字体
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表，调整大小和布局
        fig = plt.figure(figsize=(6, 7))  # 增加高度留出标题空间
        ax = fig.add_subplot(111, polar=True)
        plt.subplots_adjust(top=0.85)  # 调整子图位置，给标题留出更多空间
        
        # 准备数据
        categories = ['Win Rate', 'Logic', 'Persuasion', 'API Stability']
        values = [
            scores.get('winrate_total', 0),    # 胜率总分
            scores.get('logic_total', 0),      # 逻辑推理总分
            scores.get('persuasion_total', 0), # 说服力总分
            scores.get('api_total', 0)         # API稳定性总分
        ]
        
        # 不再限制值的范围，让它可以超出圆圈
        # values = [max(min(v, 2), -2) for v in values]
        
        # 设置角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        # 绘制雷达图
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        
        # 设置刻度，改为±1.5σ
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(-1.5, 1.5)  # 修改为±1.5σ
        ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])  # 调整刻度标记
        
        # 添加标题
        plt.title(f'Player {pid} Performance', pad=20)
        
        # 保存图片
        plt.savefig(f'player_{pid}_radar.png')
        plt.close()

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
    parser.add_argument('--mode', choices=['basic', 'model'], default='basic',
                      help='分析模式: basic-基础分析, model-模型表现分析')
    args = parser.parse_args()
    
    app = create_app()
    with app.app_context():
        analyzer = GameAnalyzer()
        if args.mode == 'basic':
            results = analyzer.analyze_games(args.start_id, args.end_id)
            analyzer.print_analysis(results)
        else:
            results = analyzer.analyze_model_performance(args.start_id, args.end_id)
            analyzer.print_model_analysis(results) 