import os
import random
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage
from langchain_ollama import OllamaLLM
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import re
import json
from typing import List, Dict, Optional, Set
from langchain.memory import ConversationBufferMemory
from game_output import GameOutput, TerminalOutput
from flask_sqlalchemy import SQLAlchemy
from models import db, Game, GamePlayer, GameRound, Speech, AIThought, IdentityGuess, User
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask_migrate import Migrate
#from app import db  # 导入数据库实例

os.environ["OPENAI_API_KEY"] = "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"
CHARACTER = ['活泼', '激动', '沉稳', '粗鲁', '直白', '城府深', '卖弄', '单纯', '急躁']
STRATEGY = ['激进', '保守', '稳定']
_model_api = 'chatgpt'  # 私有变量

class OllamaAdapter:
    def __init__(self, model_name: str, temperature: float = 0.5):
        """
        初始化 Ollama 适配器
        :param model_name: Ollama 模型名称，如 "llama2"
        :param temperature: 温度参数，控制生成文本的随机性
        """
        self.llm = OllamaLLM(model=model_name, temperature=temperature)

    def __call__(self, messages):
        """
        模拟 OpenAI 的调用方式
        :param messages: 包含 HumanMessage 的列表
        :return: 返回一个包含生成内容的字典
        """
        # 将 HumanMessage 转换为 Ollama 的输入格式
        prompt = messages[0].content if messages else ""
        response = self.llm(prompt)
        return {"content": response}

class Player:
    def __init__(self, player_id: str, team_mates: List[str] = None, 
                 is_human: bool = False, output: Optional[GameOutput] = None):
        """
        初始化玩家
        Args:
            player_id: 玩家ID
            team_mates: 队友列表
            is_human: 是否为人类玩家
            output: 输出接口
        """
        self.output = output
        self.id = player_id
        self.role = None  # 初始化时不设置角色
        self.is_human = is_human
        self.character = random.choice(CHARACTER)
        self.strategy = random.choice(STRATEGY)
        
        # 记忆系统
        self.current_memory = ConversationBufferMemory()
        self.summary_memory = "First round, no summary generated yet"
        # 添加规则总结
        self.rule_summary = ""  # 存储规则的总结
        self.core_beliefs = {
            "confirmed_allies": [],
            "confirmed_enemies": []
        }
        
        self.team_mates = []  # 初始化为空列表，等待角色分配后设置
        
        # 初始化猜测表
        self.guess = {f"P{i}": "unknown" for i in range(1, 6) if f"P{i}" != self.id}
        
        self.character_role = None  # 特殊角色（如摩根勒菲）
        self.has_amulet = False  # 是否持有魔法指示物

    def get_guess(self):
        return self.guess

    def get_role(self):
        return self.role

    def add_event(self, event: str):
        """添加事件到当前轮次记忆"""
        self.current_memory.save_context(
            {"input": f"[事件] {event}"},
            {"output": ""}
        )

    def add_dialogue(self, speaker: str, content: str):
        """添加对话到当前轮次记忆"""
        self.current_memory.save_context(
            {"input": f"[对话] {speaker}说：{content}"},
            {"output": ""}
        )

    def get_history(self):
        """获取历史记忆"""
        return self.memory.load_memory_variables({})['history']

    def _get_current_memory(self) -> str:
        """获取当前轮次记忆（未总结的原始信息）"""
        return self.current_memory.load_memory_variables({})['history']

    def _get_summary_history(self) -> str:
        """获取历史摘要（最多保留最近3次）"""
        return self.summary_memory#"\n\n".join(self.summary_memory)

    def get_context_for_decision(self) -> str:
        """生成决策上下文：历史摘要 + 当前信息"""
        return f"""
        历史分析总结：
        {self._get_summary_history()}

        当前局信息：
        {self._get_current_memory()}
        """

    def generate_summary(self, game_state: Dict, llm) -> str:
        """生成本轮总结并存入记忆"""
        if not game_state['last_team']:  # 如果是第一轮
            return "First round, no summary needed."

        if self.role == 'red':
            prompt = f"""
                                你正在玩Quest桌游，你的ID是{self.id}, 身份是{self.role}阵营玩家。当前局势：
                                - 蓝方胜利次数：{game_state['blue_wins']}
                                - 红方胜利次数：{game_state['red_wins']}
                                - 最新任务结果：{game_state['last_result']}
                                - 最近这次任务的人员组成：{game_state['last_team']}
                                - 最近这次任务的队长：{game_state['last_leader_id']}
                                - 最近这次任务的失败票数：{game_state['last_fail_votes']}
                                - 你的游玩策略风格: {self.strategy}
                                提示：红方的目的是破坏任务，所以如果任务结果是"失败"，就意味着是红方赢下了一局

                                本轮关键信息，如果是第一轮，则玩家聊天记录为空：
                                {self._get_current_memory()}

                                历史摘要回顾：
                                {self._get_summary_history()}

                                游戏规则回顾：
                                {self.rule_summary}

                                请完成两个任务：
                                
                                1. 分析并总结当前局势：
                                如果你是红方玩家，请综合分析以下内容：
                                - 评估场上的胜负局势，和你跟你队友目前在其它玩家眼中的信任度，来思考是应该执行更保守的策略（比如隐藏身份，跟随大流），还是更激进的策略
                                - 对场上的蓝队玩家逐一整理一个名单，谁更值得拉拢，应该向谁泼脏水，让他被排除在任务之外。
                                - 对后续游戏的策略计划。
                                
                                2. 更新对其他玩家的身份猜测：
                                从除了你自己ID之外的其它玩家中猜测最有可能是红方的1-2个玩家，如果没有明确的怀疑对象，可以猜测"无"。
                                然后，针对剩下的玩家，判断他们是蓝方还是不能确定。
                                
                                请按以下格式输出：
                                Summary:
                                [在这里输出500字以内的局势分析]
                                
                                Guess:
                                // 对除自己外的所有玩家的猜测，值只能是"red"、"blue"或"unknown"， 示例：
                                {{"P1": "blue", "P2": "red", "P3": "unknown"}}
                                 """
        else:
            prompt = f"""
                    你正在玩Quest桌游，你的ID是{self.id}, 身份是{self.role}阵营玩家。当前局势：
                    - 蓝方胜利次数：{game_state['blue_wins']}
                    - 红方胜利次数：{game_state['red_wins']}
                    - 最新任务结果：{game_state['last_result']}
                    - 最近这次任务的人员组成：{game_state['last_team']}
                    - 最近这次任务的队长：{game_state['last_leader_id']}
                    - 最近这次任务的失败票数：{game_state['last_fail_votes']}
                    - 你的游玩策略风格: {self.strategy}

                    提示：如果任务失败票数为1，说明队伍中只有一个红方投了失败票；如果失败票数为2，说明可能有两个红方都投了失败票

                    本轮关键信息，如果是第一轮，则玩家聊天记录为空：
                    {self._get_current_memory()}

                    历史摘要回顾：
                    {self._get_summary_history()}

                    游戏规则回顾：
                    {self.rule_summary}

                    对其它玩家的身份推测：
                    {self.get_guess()}

                    如果你是蓝方玩家，请综合分析以下内容：
                    1. 当前局的任务表现和可疑玩家, 由于游戏中蓝方（忠臣）只能投成功票，只有红方（爪牙）才有可能投出失败票，所以如果任务出现失败，那其中一定有红方混入。并且即使任务成功，也有可能有红方为了隐藏身份而投了成功票，所以不能完全对队伍成员排除嫌疑。
                    2. 其他玩家的发言可信度分析。要结合对其它玩家的身份推测，来判断是否相信他们的发言，还是对他们的发言进行反向推理。也要尝试判断是不是有人在尝试混淆视听，尤其是质疑你是红方玩家的人，要重点关注。 
                    3. 你当前的核心怀疑对象及其依据。由于场上只有两个红方，所以你的核心怀疑对象不应超过两个，如果在没有确凿证据的情况下，也不应随便将玩家加入你的核心怀疑名单。
                    4. 你要尽可能尝试说服其它的蓝方玩家，让他们相信你是蓝方玩家，这样轮到他们做队长的时候才会把你加入他们的队伍。
                    4. 对后续游戏的策略计划。你的目标是要尽可能确定其它玩家的身份，来帮助你找到可以值得信任的蓝方队友，并且避开红方玩家，尽可能在后续的任务中确保只有蓝方玩家参与，保证任务的成功。

                    请完成两个任务：
                    
                    1. 分析并总结当前局势：
                    如果你是蓝方玩家，请综合分析以下内容：
                    1. 当前局的任务表现和可疑玩家, 由于游戏中蓝方（忠臣）只能投成功票，只有红方（爪牙）才有可能投出失败票，所以如果任务出现失败，那其中一定有红方混入。并且即使任务成功，也有可能有红方为了隐藏身份而投了成功票，所以不能完全对队伍成员排除嫌疑。
                    2. 其他玩家的发言可信度分析。要结合对其它玩家的身份推测，来判断是否相信他们的发言，还是对他们的发言进行反向推理。也要尝试判断是不是有人在尝试混淆视听，尤其是质疑你是红方玩家的人，要重点关注。 
                    3. 你当前的核心怀疑对象及其依据。由于场上只有两个红方，所以你的核心怀疑对象不应超过两个，如果在没有确凿证据的情况下，也不应随便将玩家加入你的核心怀疑名单。
                    4. 你要尽可能尝试说服其它的蓝方玩家，让他们相信你是蓝方玩家，这样轮到他们做队长的时候才会把你加入他们的队伍。
                    4. 对后续游戏的策略计划。你的目标是要尽可能确定其它玩家的身份，来帮助你找到可以值得信任的蓝方队友，并且避开红方玩家，尽可能在后续的任务中确保只有蓝方玩家参与，保证任务的成功。
                    
                    2. 更新对其他玩家的身份猜测：
                    从除了你自己ID之外的其它玩家中猜测最有可能是红方的1-2个玩家，如果没有明确的怀疑对象，可以猜测"无"。
                    然后，针对剩下的玩家，判断他们是蓝方还是不能确定。
                    
                    请按以下格式输出：
                    Summary:
                    [在这里输出500字以内的局势分析]
                    
                    Guess:
                    // 对除自己外的所有玩家的猜测，值只能是"red"、"blue"或"unknown"， 示例：
                    {{"P1": "blue", "P2": "red", "P3": "unknown"}}
                    """

        model_api = get_model_api()
        if model_api in ['ollama-32b', 'ollama-7b']:
            response = llm([HumanMessage(content=prompt)]).get("content", "")
        else:
            response = llm([HumanMessage(content=prompt)]).content

        # 解析响应
        try:
            # 清理和标准化响应文本
            response = response.strip()
            if "Summary:" not in response or "Guess:" not in response:
                print(f"Invalid response format: {response}")
                return response

            summary_part = response.split("Summary:")[1].split("Guess:")[0].strip()
            guess_part = response.split("Guess:")[1].strip()
            
            # 清理 JSON 字符串
            guess_part = guess_part.replace("'", '"')  # 替换单引号为双引号
            # 移除注释行
            guess_part = '\n'.join(line for line in guess_part.split('\n') if not line.strip().startswith('//'))
            # 移除 markdown 格式
            guess_part = re.sub(r'```json\s*|\s*```', '', guess_part)
            # 确保是有效的 JSON 字符串
            if not guess_part.startswith('{'):
                print(f"Invalid JSON format: {guess_part}")
                return response
            
            guess_json = json.loads(guess_part)
            
            # 更新猜测表
            red_count = sum(1 for p in guess_json if guess_json[p] == "red")
            if red_count > 2:
                red_players = [p for p in guess_json if guess_json[p] == "red"][:2]
                for p in guess_json:
                    if p not in red_players and guess_json[p] == "red":
                        guess_json[p] = "unknown"
            
            # 更新 self.guess
            for p in self.guess:
                if p in guess_json and guess_json[p] in ["red", "blue", "unknown"]:
                    self.guess[p] = guess_json[p]
            
            self.summary_memory = summary_part
        except Exception as e:
            print(f"解析LLM响应失败：{e}\n响应内容：{response}")
            # 保持原有的猜测不变
            self.summary_memory = response  # 至少保存一些信息

        # 清空当前轮次记忆
        self.current_memory.clear()
        return response

    def generate_speech(self, game_state: Dict, llm) -> str:
        if self.is_human:
            prompt = f"玩家 {self.id} 的回合：请发言（你是{'红方' if self.role == 'red' else '蓝方'}）"
            if self.role == "red":
                prompt += f"，你的队友是 {self.team_mates}"
            print(f"Waiting for human player {self.id} speech input")  # 调试信息
            input_text = self.output.get_player_input(prompt, self.id)
            print(f"Received speech: {input_text}")  # 调试信息
            return input_text
        else:
            """生成发言（使用整合后的记忆上下文）"""
            context = self.get_context_for_decision()

            prompt = f"""
                    {context}

                    你正在玩Quest桌游，身份是{"蓝" if self.role == "blue" else "红"}阵营的{"忠臣" if self.role == "blue" else "爪牙"}。
                    当前任务阶段：第{game_state['round'] + 1}轮
                    当前局势：蓝方{game_state['blue_wins']}胜 / 红方{game_state['red_wins']}胜
                    当前任务队长：
                    {game_state['leader_id']}

                    你的

                    你的怀疑清单：
                    {self.get_guess()}

                    请你分两步回答：
                    1. 在<think>...</think>中写详细推理：
                        <think>
                        1. 分析历史摘要中的关键线索
                        2. 结合当前信息评估其他玩家可信度
                        3. 根据你summary中的策略去制定这轮的发言策略
                        4. 不要怀疑自己（{self.id}）；
                        5. "我相信"和"我怀疑"列表不能有重叠；
                        6. 如果你是红方，请尽量隐藏自己和队友的身份，并尝试误导蓝方。
                        </think>

                    2. 用"Final:"开头给出一句不超过100个字的发言，如果你是蓝方，相信和怀疑的目标尽可能与你的怀疑清单一致，如果你是红方，可以根据策略自由发挥，混淆视听，不要暴露自己和队友的身份，甚至可以声称自己是另一方来迷惑对方。
                       如果这是第一轮，信息不足的情况下，你也可以不发表对任何人的相信和怀疑。
                       你也可以通过强调自己是蓝方，来获得（或者骗取）他人的信任
                       发言在允许的范围内个性化，增加游戏的趣味性,尤其是不能跟别人发一模一样的话

                       以下是一些发言的可参考模板，但是不要局限于这些模板：
                       "我相信X、Y， 原因：...， 我怀疑A、B， 原因：..."
                       "由于第一轮游戏，没有什么信息，我选择不发表评论"
                       "我是好人（或者蓝方，忠臣），请队长带上我做任务"

                       或者你可以选择去bluff：
                       "从直觉判断，我对X的身份有些怀疑，请向我证明你是一个好人（或者蓝方，忠臣）"

                    然后以Final:开头生成符合角色身份和你的人物个性{self.character}和你的游玩策略{self.strategy}的发言：
                    Final:"""
            model_api = get_model_api()
            if model_api in ['ollama-32b', 'ollama-7b']:
                response = llm([HumanMessage(content=prompt)]).get("content", "")
                message = response.split("Final:")[-1].strip()[:100]
            else:
                response = llm([HumanMessage(content=prompt)])
                message = response.content.split("Final:")[-1].strip()[:100]
            
            return message

    def propose_team(self, required_size, llm):
        """生成队伍提议，根据身份猜测表选择队友"""
        if self.is_human:
            prompt = f"=== 你的组队回合 ===\n"
            prompt += f"你需要选择 {required_size} 名玩家组成队伍（包括你自己）。\n"
            prompt += "可选玩家：P1, P2, P3, P4, P5\n"
            prompt += "请输入队伍成员（用空格分隔，例如：P1 P2）"
            if self.role == "red":
                prompt += f"\n你是红方玩家，你的队友是{[f'P{mate}' for mate in self.team_mates]}"
            else:
                prompt += "\n你是蓝方玩家，请根据你的分析做出选择"

            team = self.output.get_player_input(prompt, self.id).strip().split()
            # 统一格式化玩家ID
            team = [f"P{str(pid).upper().replace('P', '')}" for pid in team]
            return team[:required_size]
        else:
            team = [self.id]  # 首先包含自己
            
            # 获取当前轮次
            game_round = len(self.current_memory.load_memory_variables({})['history'].split("第")) - 1
            
            if self.role == "red":
                if game_round == 3:  # 第四轮
                    # 第四轮需要两个失败票，必须选择一个红队队友
                    team_mate = next(p for p in self.team_mates if f"P{p}" != self.id)
                    team.append(f"P{team_mate}")
                    # 剩余名额从非队友中随机选择
                    available_players = [f"P{i}" for i in range(1, 6) 
                                       if i not in self.team_mates 
                                       and f"P{i}" != self.id]
                    random.shuffle(available_players)
                    while len(team) < required_size:
                        if available_players:
                            team.append(available_players.pop())
                else:
                    # 其他轮次绝对不选择队友
                    available_players = [f"P{i}" for i in range(1, 6) 
                                       if i not in self.team_mates 
                                       and f"P{i}" != self.id]
                    random.shuffle(available_players)
                    while len(team) < required_size:
                        if available_players:
                            team.append(available_players.pop())
            else:
                # 蓝队优先从猜测为蓝方的玩家中选择
                blue_players = [pid for pid, role in self.guess.items() 
                              if role == "blue" and pid != self.id]
                unknown_players = [pid for pid, role in self.guess.items() 
                                 if role == "unknown" and pid != self.id]
                
                # 优先从蓝方玩家中选择
                random.shuffle(blue_players)
                while len(team) < required_size and blue_players:
                    team.append(blue_players.pop())
                
                # 如果还不够，从未知玩家中选择
                random.shuffle(unknown_players)
                while len(team) < required_size and unknown_players:
                    team.append(unknown_players.pop())
            
            # 确保没有重复的队员
            team = list(set(team))
            # 如果队伍人数不足，补充其他玩家
            while len(team) < required_size:
                available = [f"P{i}" for i in range(1, 6) if f"P{i}" not in team]
                if available:
                    team.append(random.choice(available))
            
            return team[:required_size]  # 确保不超过所需人数

    def choose_next_leader(self, current_players: List['Player']) -> str:
        """选择下一任队长"""
        if self.is_human:
            available_ids = [p.id for p in current_players]
            prompt = f"请选择下一任队长（可选玩家：{', '.join(available_ids)}）\n"
            if self.role == "red":
                prompt += f"提示：你是红方玩家，你的队友是{[f'P{mate}' for mate in self.team_mates]}\n"
            prompt += "请输入玩家ID（例如：P1）："
            while True:
                next_leader = self.output.get_player_input(prompt, self.id).strip()
                if next_leader in available_ids:
                    return next_leader
                self.output.send_message("无效的选择，请重新输入", 'action')
            
        # 排除自己和已经当过队长的人
        available_players = [p for p in current_players]
        
        if self.role == "blue":
            # 蓝方玩家从自己认为是蓝方的玩家中选择
            blue_candidates = [p for p in available_players if self.guess.get(p.id) == "blue"]
            if blue_candidates:
                return random.choice(blue_candidates).id
            
            # 如果没有确定的蓝方，从unknown中选择
            unknown_candidates = [p for p in available_players if self.guess.get(p.id) == "unknown"]
            if unknown_candidates:
                return random.choice(unknown_candidates).id
            
            # 如果都没有，随机选择一个
            return random.choice(available_players).id
        else:
            # 红方玩家有50%概率选择队友
            if random.random() < 0.5:
                # 从队友中选择（排除自己）
                team_mates = [p for p in available_players if int(p.id.replace('P', '')) in self.team_mates]
                if team_mates:
                    return random.choice(team_mates).id
            
            # 随机选择一个非队友玩家
            non_team_players = [p for p in available_players if int(p.id.replace('P', '')) not in self.team_mates]
            if non_team_players:
                return random.choice(non_team_players).id
            
            # 如果上述条件都不满足，随机选择一个
            return random.choice(available_players).id

    def get_red_identification_guesses(self) -> Set[str]:
        """
        从玩家的猜测表中提取红方猜测。
        如果红方猜测不足两个，从unknown中随机补充。
        如果还不足两个，从blue中随机补充。
        """
        # 收集所有猜测为红方的玩家ID
        red_guesses = {pid.strip().upper() for pid, role in self.guess.items() if role.lower() == "red"}
        
        # 如果红方猜测不足两个，从unknown中随机补充
        if len(red_guesses) < 2:
            unknown_players = {pid.strip().upper() for pid, role in self.guess.items() if role.lower() == "unknown"}
            needed = 2 - len(red_guesses)
            
            if unknown_players:
                # 从unknown中随机选择所需数量（或全部如果数量不足）
                to_add = random.sample(list(unknown_players), min(needed, len(unknown_players)))
                red_guesses.update(to_add)
            
            # 如果加入unknown后仍不足两个，从blue中随机补充
            if len(red_guesses) < 2:
                blue_players = {pid.strip().upper() for pid, role in self.guess.items() if role.lower() == "blue"}
                still_needed = 2 - len(red_guesses)
                
                if blue_players:
                    to_add = random.sample(list(blue_players), min(still_needed, len(blue_players)))
                    red_guesses.update(to_add)
        
        return red_guesses

class AvalonSimulator:
    def __init__(self, output: GameOutput, human_player_id: str = "P5", test_mode: bool = False, p5_is_morgan: bool = False):
        """
        初始化游戏模拟器
        Args:
            output: 输出接口
            human_player_id: 人类玩家ID
            test_mode: 是否为测试模式
            p5_is_morgan: 在测试模式下，是否将P5设置为摩根
        """
        self.output = output
        self.test_mode = test_mode
        self.p5_is_morgan = p5_is_morgan
        print(f"Initializing game in {'test' if test_mode else 'normal'} mode")
        
        # 初始化基本属性
        self.round = 0
        self.blue_wins = 0
        self.red_wins = 0
        self.final_winner = None
        self.task_sizes = [2, 2] if test_mode else [2, 3, 2, 3, 3]
        self.current_leader_index = 0
        self.leaders = []
        self.last_team = None
        self.last_result = None
        self.last_fail_votes = 0
        
        if test_mode:
            human_player_id = "P5"
            print("Test mode: Setting human_player_id to P5")
        
        # 初始化游戏数据库记录
        self.current_game = Game(
            start_time=datetime.now(timezone.utc)
        )
        db.session.add(self.current_game)
        db.session.commit()

        # 初始化玩家
        self.players = self._initialize_players(human_player_id)
        
        # 在测试模式下，确保 P5 是第一个队长
        if test_mode:
            print("Test mode: Setting P5 as first leader")
            self.current_leader_index = 4
            self.leaders = ["P5"]
            print(f"Current leader set to: {self.players[self.current_leader_index].id}")
        
        self.llm = self._initialize_llm()
        
        # 初始化非人类玩家的游戏规则总结记忆
        self.initialize_ai_memory()

    def _initialize_players(self, human_player_id: str):
        """初始化玩家列表"""
        players = []
        
        for i in range(1, 6):
            player_id = f"P{i}"
            is_human = (player_id == human_player_id)
            player = Player(player_id, is_human=is_human, output=self.output)
            players.append(player)
        
        return players

    def _initialize_llm(self):
        """初始化语言模型"""
        model_api = get_model_api()
        if model_api == 'ollama-32b':
            return OllamaAdapter(model_name="deepseek-r1:32b", temperature=0.8)
        elif model_api == 'ollama-7b':
            return OllamaAdapter(model_name="deepseek-r1:7b", temperature=0.8)
        elif model_api == 'glm-zero':
            return ChatOpenAI(
                model="glm-zero-preview",
                api_key="sk-DhBqcMHlBY21mmQTBE7VkwHhOrjjEdF5KsrOnR3rOwXVL9Il",
                base_url="https://www.dmxapi.com/v1/"
            )
        elif model_api == 'deepseek-reasoner':
            return ChatOpenAI(
                model="deepseek-reasoner",
                api_key="sk-DhBqcMHlBY21mmQTBE7VkwHhOrjjEdF5KsrOnR3rOwXVL9Il",
                base_url="https://www.dmxapi.com/v1/"
            )
        elif model_api == 'doubao-lite':
            return ChatOpenAI(
                model="doubao-lite-32k",
                api_key="sk-DhBqcMHlBY21mmQTBE7VkwHhOrjjEdF5KsrOnR3rOwXVL9Il",
                base_url="https://www.dmxapi.com/v1/"
            )
        elif model_api == 'siliconflow':
            return ChatOpenAI(
                model="deepseek-ai/DeepSeek-R1",
                api_key="sk-ailkxszopmpfvssuabvqsqccuhsigqfqmrybfjztezsmbhjh",
                base_url="https://api.siliconflow.cn/v1"
            )
        else:
            return ChatOpenAI(model='o1-mini')

    def reset_game(self):
        """重置游戏状态"""
        self.round = 0
        self.blue_wins = 0
        self.red_wins = 0
        self.current_leader_index = random.randint(0, len(self.players)-1)
        self.leaders = []
        self.last_team = None
        self.last_result = None
        self.last_fail_votes = 0
        
        # 重新初始化任务大小
        self.task_sizes = [2, 2] if self.test_mode else [2, 3, 2, 3, 3]
        
        # 重置所有玩家的状态
        for player in self.players:
            player.current_memory = ConversationBufferMemory()
            player.summary_memory = "First round, no summary generated yet"
            player.core_beliefs = {
                "confirmed_allies": [],
                "confirmed_enemies": []
            }
            # 重置猜测表
            player.guess = {f"P{i}": "unknown" for i in range(1, 6) if f"P{i}" != player.id}
        
        # 在测试模式下，确保 P5 是第一个队长
        if self.test_mode:
            self.current_leader_index = next((i for i, p in enumerate(self.players) if p.id == "P5"), None)
            if self.current_leader_index is None:
                # 这里可以打印错误日志或 raise 异常，提示未能找到 P5
                print("Error: 未在 players 列表中找到 P5")
        
        # 创建新的游戏记录
        self.current_game = Game()
        db.session.add(self.current_game)
        db.session.commit()

    def discussion_phase(self):
        """模拟讨论阶段"""
        self.output.send_message("讨论阶段开始", 'action')
        game_state = self.get_game_state()
        speeches = {}  # 只存储每个玩家的发言内容
        for player in self.players:
            speech = player.generate_speech(game_state, self.llm)
            speeches[player.id] = speech  # 存储发言内容
            self.output.send_message(f"{player.id} 说：{speech}", 'info')
            # 广播发言到所有玩家记忆（这部分保持不变，因为玩家需要记住所有人的发言）
            for listener in self.players:
                listener.add_event(f"{player.id}发言：{speech}")
        return speeches

    def run_round(self):
        """执行单轮游戏"""
        # 在测试模式下检查轮数
        if self.test_mode and self.round >= 2:
            return False

        # 保存当前的队长作为上一任队长
        self.last_leader = self.current_leader_index
        
        # 第一轮开始时选择首个队长
        if self.round == 0:
            if self.test_mode:
                leader = self.current_leader_index
                self.output.send_message(f"第{self.round + 1}轮 P5作为首个队长", 'action')
            else:
                # 从未当过队长的玩家中选择第一个队长
                available_players = [p for p in self.players if p.id not in self.leaders]
                next_leader = random.choice(available_players)
                self.current_leader_index = self.players.index(next_leader)
                self.leaders.append(next_leader.id)
                self.output.send_message(f"第{self.round + 1}轮 随机选择 {leader.id}作为队长", 'action')
   

        # 讨论阶段
        round_speeches = self.discussion_phase()

        # 队长选择队伍
        leader = self.players[self.current_leader_index]
        team = leader.propose_team(
            required_size=self.task_sizes[self.round],
            llm=self.llm
        )
        self.output.send_message(f"{leader.id}指定队伍：{team}", 'info')

        # 队长选择魔法指示物目标
        if leader.is_human:
            prompt = f"选择要施加魔法指示物的玩家（输入玩家ID）："
            target_id = self.output.get_player_input(prompt, leader.id)
            amulet_target = next((p for p in self.players if p.id == target_id), None)
        else:
            # AI队长随机选择非自己玩家：team 中存储的是玩家ID，此处转换为对应的 Player 对象
            candidates = [p for tid in team for p in self.players if p.id == tid and p != leader]
            amulet_target = random.choice(candidates) if candidates else None
        
        # 因为 amulet 是强制使用的，直接设置
        amulet_target.has_amulet = True
        self.last_amulet_player = amulet_target.id
        self.output.send_message(f"{leader.id} 对 {amulet_target.id} 使用了魔法指示物", 'action')

        # 执行任务
        success_votes = 0
        fail_votes = 0
        for member in team:
            # 将成员ID转换为Player对象
            player_obj = next((p for p in self.players if p.id == member), None)
            if player_obj:
                vote = self.get_mission_vote(player_obj)
            else:
                vote = False  # 默认处理无效成员
            if vote:
                success_votes += 1
            else:
                fail_votes += 1

        self.output.send_message(f"任务成功票数：{success_votes}", 'info')
        self.output.send_message(f"任务失败票数：{fail_votes}", 'info')
        self.last_fail_votes = fail_votes

        # 判断任务结果
        required_fails = 1  # 默认1票失败
        if self.round == 3:  # 第四轮
            required_fails = 1  # 根据新规则修改
        
        if fail_votes >= required_fails:
            self.red_wins += 1
        else:
            self.blue_wins += 1

        self.last_team = team
        self.last_result = "成功" if fail_votes == 0 else "失败"

        # 当前队长选择下一任队长：只允许未做过队长的玩家被选择
        available_players = [p for p in self.players if p.id not in self.leaders]
        if not available_players:
            # 当所有玩家都已经做过队长后，重置领队历史，但保留当前队长，避免连续重复
            self.leaders = [self.players[self.current_leader_index].id]
            available_players = [p for p in self.players if p.id != self.players[self.current_leader_index].id]
        
        # 选择下一任队长后更新历史
        next_leader = self.players[self.current_leader_index].choose_next_leader(available_players)
        if isinstance(next_leader, str):
            next_leader_obj = next((p for p in self.players if p.id == next_leader), None)
        else:
            next_leader_obj = next_leader
        self.current_leader_index = self.players.index(next_leader_obj)
        # 记录新队长的id（确保每位玩家在一个周期只做一次队长）
        if next_leader_obj.id not in self.leaders:
            self.leaders.append(next_leader_obj.id)
        self.output.send_message(f"{leader.id}选择了 {next_leader_obj.id}作为下一轮队长", 'action')
        
        # 将队长选择信息添加到所有玩家的记忆中
        for player in self.players:
            player.add_event(f"{leader.id}选择了{self.players[self.current_leader_index].id}作为下一任队长")
        
        # 第一部分：写入基本游戏信息和玩家发言
        try:
            current_round = GameRound(
                game=self.current_game,
                round_number=self.round,
                leader_id=leader.id,
                team_members=','.join(team),
                fail_votes=fail_votes,
                result='success' if success_votes > 0 else 'fail'
            )
            db.session.add(current_round)
            
            # 记录发言到数据库
            for player_id, speech_content in round_speeches.items():
                speech = Speech(
                    round=current_round,
                    player_id=player_id,
                    content=speech_content
                )
                db.session.add(speech)
            
            db.session.commit()
        except Exception as e:
            print(f"Error saving round data (part 1): {e}")
            db.session.rollback()
        
        self.round += 1
        
        # 检查游戏是否结束
        if self.blue_wins >= 3 or self.red_wins >= 3 or self.round >= len(self.task_sizes):
            return False
        
        # AI开始思考：并行调用生成总结，提高效率
        self.output.send_message("AI玩家思考中，过程可能会持续几分钟，请稍安勿躁...", 'action')
        ai_summaries = {}
        with ThreadPoolExecutor() as executor:
            # 提交所有非人类玩家的生成总结任务
            future_to_player = {
                executor.submit(player.generate_summary, game_state=self.get_game_state(), llm=self.llm): player.id
                for player in self.players if not player.is_human
            }
            # 收集各任务的返回结果
            for future in as_completed(future_to_player):
                player_id = future_to_player[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.output.send_message(f"AI总结生成失败 for {player_id}: {exc}", "error")
                    result = "总结生成失败"
                ai_summaries[player_id] = result
        
        # 第二部分：写入AI的思考和猜测
        try:
            # 获取刚才创建的回合记录
            current_round = GameRound.query.filter_by(
                game_id=self.current_game.id,
                round_number=self.round - 1
            ).first()
            
            # 记录AI的思考和猜测
            for player in self.players:
                if not player.is_human:
                    thought = AIThought(
                        round=current_round,
                        player_id=player.id,
                        summary=ai_summaries[player.id]
                    )
                    db.session.add(thought)
                    
                    for target_id, guess in player.guess.items():
                        guess_record = IdentityGuess(
                            round=current_round,
                            guesser_id=player.id,
                            target_id=target_id,
                            guessed_role=guess
                        )
                        db.session.add(guess_record)
            
            db.session.commit()
        except Exception as e:
            print(f"Error saving round data (part 2): {e}")
            db.session.rollback()
        
        # 清除魔法指示物
        for p in self.players:
            p.has_amulet = False
        
        return True

    def get_game_state(self):
        return {
            "blue_wins": self.blue_wins,
            "red_wins": self.red_wins,
            "last_team": self.last_team,
            "last_result": self.last_result,
            "last_fail_votes": getattr(self, 'last_fail_votes', 0),  # 添加失败票数
            "round": self.round,
            "leader_id":self.players[self.current_leader_index].id,
            "last_leader_id":self.players[self.last_leader].id
        }

    def get_mission_vote(self, player: Player) -> bool:
        """获取玩家的任务投票"""
        if player.is_human:
            if player.role == "red":
                # 如果不是摩根且被施加魔法指示物，则只能投成功票
                if player.character_role != "morgan" and player.has_amulet:
                    self.output.send_message("你被魔法指示物强制，必须投成功票", 'action')
                    return True
                elif player.character_role == "morgan" and player.has_amulet:
                    self.output.send_message("作为摩根勒菲，你无视了魔法指示物的限制", 'action')
                prompt = "你是红方玩家，请输入数字选择你的投票：\n1 = 投成功票（任务成功）\n2 = 投失败票（任务失败）\n请输入1或2："
                vote = self.output.get_player_input(prompt, player.id)
                return vote.strip() == "1"
            else:
                self.output.send_message("你是蓝方玩家，只能投成功票", 'action')
                return True
        else:
            if player.role == "red":
                # 摩根勒菲无视魔法指示物
                if player.character_role == "morgan":
                    return random.random() > 0.9  # 90%概率投失败
                
                # 普通红队玩家如果被施加魔法指示物必须投成功
                if player.has_amulet:
                    return True
                
                return random.random() > 0.9  # 90%概率投失败
            else:
                return True

    def run_game(self, test_mode=False, p5_is_morgan=False):
        """
        运行游戏主循环
        :param test_mode: 是否为测试模式
        :param p5_is_morgan: 在测试模式下，是否将P5设置为摩根
        """
        #from app import db  # 导入数据库实例
        
        self.test_mode = test_mode
        self.p5_is_morgan = p5_is_morgan
        print(f"Game started with test_mode={test_mode}, p5_is_morgan={p5_is_morgan}")  # 调试日志

        # 重新获取当前游戏记录
        self.current_game = db.session.merge(self.current_game)
        
        # 分配角色
        self.assign_roles()
        
        while self.blue_wins < 3 and self.red_wins < 3:
            if not self.run_round():
                break
                
        # 最终指认阶段
        if self.red_wins >= 3:
            self.final_identification()
        else:
            # 如果红方未达到3胜，则蓝方获胜
            self.final_winner = "blue"
            
        # 统一的游戏结果公布
        self.output.send_message("=== 游戏结束 ===", 'result')
        self.output.send_message(f"最终比分：蓝方 {self.blue_wins} 胜 | 红方 {self.red_wins} 胜", 'result')
        self.output.send_message(f"最终胜利方：{'蓝方' if self.final_winner == 'blue' else '红方'}", 'result')
        self.output.send_message("玩家身份揭晓：", 'result')
        for p in self.players:
            self.output.send_message(f"{p.id}: {p.role}", 'result')
        
        # 更新游戏结束时间和最终胜负标记
        self.current_game.end_time = datetime.now(timezone.utc)
        self.current_game.winner = self.final_winner

        # 更新数据库中与 user 关联的 GamePlayer 战绩
        for gp in self.current_game.players:
            if gp.user_id is not None:
                user = gp.user
                # 判断该 GamePlayer 是否获胜（即其角色与最终胜利方一致）
                if gp.role == self.final_winner:
                    if gp.role == "blue":
                        user.blue_wins += 1
                    elif gp.role == "red":
                        user.red_wins += 1
                else:
                    if gp.role == "blue":
                        user.blue_losses += 1
                    elif gp.role == "red":
                        user.red_losses += 1
        db.session.commit()

    def final_identification(self):
        """最终指认阶段"""
        self.output.send_message("=== 最终指认阶段 ===", 'action')
        self.output.send_message("好人阵营请指认所有红方成员", 'info')
        
        # 获取实际红方玩家集合（统一大写）
        actual_reds = {p.id.upper() for p in self.players if p.role == "red"}
        
        blue_guesses = {}
        for player in self.players:
            if player.role == "blue":
                if player.is_human:
                    prompt = "请指认你认为是红方的两位玩家（例如：P1 P2）："
                    answer = self.output.get_player_input(prompt, player.id)
                    # 将输入解析为集合，统一大写
                    guess_set = {x.strip().upper() for x in answer.split()}
                else:
                    # 对 AI 蓝队玩家，调用新方法提取红方猜测
                    guess_set = player.get_red_identification_guesses()
                    
                # 强制要求每名蓝队玩家必须指认2个红方玩家
                if len(guess_set) != 2:
                    raise ValueError(f"调试错误: 玩家 {player.id} 指认的红方玩家数量不等于2，实际为: {guess_set}")
                blue_guesses[player.id] = guess_set
        
        # 条件1：所有蓝队玩家的猜测必须正确，即各自猜测集合应为实际红队的子集
        for pid, guess_set in blue_guesses.items():
            if not guess_set.issubset(actual_reds):
                self.output.send_message(f"玩家 {pid} 的指认有误：{guess_set} 不全在正确红方 {actual_reds} 中", "result")
                self.final_winner = "red"
                return
        
        # 条件2：所有蓝队玩家猜测的并集必须覆盖所有实际红方
        union_of_guesses = set()
        for guess_set in blue_guesses.values():
            union_of_guesses.update(guess_set)
        if union_of_guesses == actual_reds:
            self.final_winner = "blue"
            self.output.send_message("指认阶段结束：蓝队的反败为胜成立。", "action")
        else:
            self.final_winner = "red"
            self.output.send_message("指认阶段结束：红队成功防守。", "action")

    def assign_roles(self):
        # 在测试模式下，如果指定P5为摩根，需要先确保P5为红方
        if self.test_mode and self.p5_is_morgan:
            p5 = next((p for p in self.players if p.id == "P5"), None)
            if p5:
                # 先将P5分配为红方
                other_players = [p for p in self.players if p.id != "P5"]
                blue_players = random.sample(other_players, 3)
                red_players = [p for p in other_players if p not in blue_players]
                red_players.append(p5)  # 将P5添加到红方
            else:
                self.output.send_message("错误：找不到P5玩家", "error")
                # 如果找不到P5，使用默认分配
                blue_players = random.sample(self.players, 3)
                red_players = [p for p in self.players if p not in blue_players]
        else:
            # 正常分配阵营
            blue_players = random.sample(self.players, 3)
            red_players = [p for p in self.players if p not in blue_players]
        
        # 设置所有玩家的角色和队友
        for p in self.players:
            if p in blue_players:
                p.role = "blue"
            else:
                p.role = "red"
                p.team_mates = [int(other.id.replace('P', '')) for other in red_players if other != p]
            
            # 如果是红方玩家，更新其猜测表
            if p.role == "red":
                for mate in p.team_mates:
                    p.guess[f"P{mate}"] = "red"
        
        # 分配摩根角色
        if self.test_mode and self.p5_is_morgan:
            p5 = next((p for p in self.players if p.id == "P5"), None)
            if p5:
                morgan = p5
            else:
                self.output.send_message("警告：找不到P5玩家", "error")
                morgan = random.choice(red_players)
        else:
            morgan = random.choice(red_players)
        
        morgan.character_role = "morgan"
        print(f"{morgan.id} 是摩根勒菲")
        
        # 更新数据库中对应的 GamePlayer 记录
        gp = GamePlayer(
            game_id=self.current_game.id,
            player_id=morgan.id,
            role=morgan.role,
            is_ai=not morgan.is_human,
            character=morgan.character,
            strategy=morgan.strategy,
            morgan=True
        )
        db.session.add(gp)
        
        # 为其他玩家创建记录
        for p in [p for p in self.players if p != morgan]:
            player_record = GamePlayer(
                game_id=self.current_game.id,
                player_id=p.id,
                role=p.role,
                is_ai=not p.is_human,
                character=p.character,
                strategy=p.strategy,
                morgan=False
            )
            db.session.add(player_record)
        
        db.session.commit()
        
        if morgan.is_human:
            self.output.send_message("你是摩根勒菲，你可以无视魔法指示物的限制", "action")

    def initialize_ai_memory(self):
        """
        对所有非人类玩家，通过LLM读取游戏规则生成初始化的总结记忆，
        帮助AI玩家熟悉游戏的基本规则
        """
        try:
            with open("game_rules.md", encoding="utf-8") as f:
                rules_text = f.read()
        except Exception as e:
            rules_text = "无法读取游戏规则。"
            self.output.send_message(f"读取游戏规则失败: {e}", "error")
        
        def init_single_ai(player):
            """单个AI玩家的初始化函数"""
            if not player.is_human:
                prompt = (
                    f"请仔细阅读以下游戏规则，并生成一段简短的总结，帮助你熟悉该游戏的基本规则：\n\n"
                    f"{rules_text}\n\n总结："
                )
                try:
                    response = self.llm([HumanMessage(content=prompt)])
                    if isinstance(response, dict) and "content" in response:
                        summary = response["content"]
                    else:
                        summary = response.content if hasattr(response, "content") else str(response)
                except Exception as ex:
                    summary = "初始化总结失败"
                    self.output.send_message(f"LLM调用失败: {ex}", "error")
                
                # 直接存储规则总结
                player.rule_summary = summary
                return f"AI玩家 {player.id} 初始化总结记忆生成成功"
            return None
        
        # 使用线程池并行处理所有AI玩家的初始化
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(init_single_ai, player) for player in self.players]
            
            # 等待所有初始化完成并输出结果
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.output.send_message(result, "info")

def set_model_api(model: str):
    """设置全局模型 API"""
    global _model_api
    _model_api = model

def get_model_api():
    """获取当前模型 API"""
    return _model_api

if __name__ == "__main__":
    terminal_output = TerminalOutput()
    game = AvalonSimulator(output=terminal_output)
    game.run_game()
