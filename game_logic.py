import os
import random
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage
from langchain_ollama import OllamaLLM
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_fireworks import ChatFireworks
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
from config import Config
from game_messages import GAME_MESSAGES

os.environ["OPENAI_API_KEY"] = "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"
CHARACTER = ['沉稳']#['活泼', '激动', '沉稳', '粗鲁', '直白', '城府深', '卖弄', '单纯', '急躁']
STRATEGY = ['稳定']#['激进', '保守', '稳定']
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
                 is_human: bool = False, output: Optional[GameOutput] = None,
                 model_api: str = None, game_lang: str = 'zh'):
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
        self.model_api = model_api or get_model_api()  # 使用指定的模型或默认模型
        self.game_lang = game_lang  # 游戏语言，影响 LLM 提示语
        self.llm = self._initialize_llm()
        
        # 记忆系统
        self.current_memory = ConversationBufferMemory()
        self.summary_memory = "First round, no summary generated yet"
        self.rules_text = ""  # 存储原始规则文本
        self.core_beliefs = {
            "confirmed_allies": [],
            "confirmed_enemies": []
        }
        
        self.team_mates = []  # 初始化为空列表，等待角色分配后设置
        
        # 初始化猜测表
        self.guess = {f"P{i}": "unknown" for i in range(1, 6) if f"P{i}" != self.id}
        
        self.character_role = None  # 特殊角色（如摩根勒菲）
        self.has_amulet = False  # 是否持有魔法指示物
        self.next_speech = ""  # 存储下一轮的发言
        self.selected_team = []
        self.magic_target = ""

    def _initialize_llm(self):
        """初始化该玩家专属的语言模型"""
        try:
            if self.model_api == 'fireworks':
                return ChatFireworks(
                    model_name="accounts/fireworks/models/deepseek-r1",
                    max_tokens=40960,
                    fireworks_api_key=Config.FIREWORKS_API_KEY,
                    temperature=0.3,
                )
            elif self.model_api == 'gemini':
                return ChatOpenAI(
                    model="gemini-2.0-flash",
                    api_key=Config.GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif self.model_api == 'ollama-32b':
                return OllamaAdapter(model_name="deepseek-r1:32b", temperature=0.8)
            elif self.model_api == 'ollama-7b':
                return OllamaAdapter(model_name="deepseek-r1:7b", temperature=0.8)
            elif self.model_api == 'glm-zero':
                return ChatOpenAI(
                    model="glm-zero-preview",
                    api_key="sk-DhBqcMHlBY21mmQTBE7VkwHhOrjjEdF5KsrOnR3rOwXVL9Il",
                    base_url="https://www.dmxapi.com/v1/"
                )
            elif self.model_api == 'deepseek-reasoner':
                return ChatOpenAI(
                    model="deepseek-ai/DeepSeek-R1",
                    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ5dng1MjExQHBzdS5lZHUiLCJpYXQiOjE3Mzk1OTkwNTR9.2CCkv7SpWhcAK6pOvkb2rXK9uPYum4FUFUKunES16yM",
                    base_url="https://api.hyperbolic.xyz/v1",
                    max_tokens=20480,
                    temperature=0.5,
                )
            elif self.model_api == 'doubao-lite':
                return ChatOpenAI(
                    model="doubao-lite-32k",
                    api_key="sk-DhBqcMHlBY21mmQTBE7VkwHhOrjjEdF5KsrOnR3rOwXVL9Il",
                    base_url="https://www.dmxapi.com/v1/"
                )
            elif self.model_api == 'siliconflow':
                llm = ChatOpenAI(
                    model="Pro/deepseek-ai/DeepSeek-R1",
                    max_tokens=20480,
                    api_key="sk-ailkxszopmpfvssuabvqsqccuhsigqfqmrybfjztezsmbhjh",
                    base_url="https://api.siliconflow.cn/v1",
                )
                # 测试 API 连接
                return llm
            else:
                return ChatOpenAI(model='o3-mini', reasoning_effort="high")
        except Exception as e:
            print(f"LLM 初始化失败 ({self.model_api}): {str(e)}")
            print("使用备用模型 o1-mini")
            return ChatOpenAI(model='o1-mini')

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

    def generate_summary(self, game_state: Dict, llm, team_size: int) -> str:
        """生成本轮总结并存入记忆"""
        game_history = game_state['game_history']
        is_leader = game_state['leader_id'] == self.id
        required_team_size = team_size
        leader_info = ""
        leader_task = ""
        other_players = [f"P{i}" for i in range(1, 6) if f"P{i}" != self.id]

        def validate_leader_response(response: str) -> bool:
            if not ("TeamSelection:" in response and "MagicTarget:" in response):
                return False
            
            try:
                # 验证队伍大小
                team_part = response.split("TeamSelection:")[1].split("MagicTarget:")[0].strip()
                team_members = [p.strip() for p in team_part.split() if p.strip()]

                for i in range(len(team_members)):
                    team_members[i] = clean_player_id(team_members[i])
                
                team_members = [p for p in team_members if p is not None][:required_team_size-1]

                print(f"队长选择的队伍: {team_members}")

                if len(team_members) != required_team_size - 1:
                    print(f"队长选择的队伍大小不正确: 需要 {required_team_size-1} 人，实际选择了 {len(team_members)} 人")
                    return False
                
                # 验证是否有重复队员
                if len(set(team_members)) != len(team_members):
                    print(f"队长选择的队伍中有重复队员: {team_members}")
                    return False
                
                # 验证队长是否把自己也加入了队伍
                if self.id in team_members:
                    print(f"队长 {self.id} 不应该在 TeamSelection 中包含自己")
                    return False
                
                # 验证魔法目标
                magic_part = response.split("MagicTarget:")[1].strip().split()[0]
                magic_part = clean_player_id(magic_part)
                if not magic_part:
                    print("队长没有选择魔法目标")
                    return False
                
                # 验证魔法目标是否在队伍中（包括队长自己）
                valid_targets = team_members + [self.id]
                if magic_part not in valid_targets:
                    print(f"魔法目标 {magic_part} 不在队伍中")
                    return False
                
                # 验证通过后，直接设置 selected_team
                
                self.selected_team = valid_targets
                print(f"解析的队员: {team_part}")
                print(f"最终队伍: {self.selected_team}")

                self.magic_target = clean_player_id(magic_part)
                print(f"队长 {self.id} 选择的魔法目标: {self.magic_target}")
                
                return True
            except Exception as e:
                print(f"验证队长响应时出错: {e}")
                return False

        if is_leader:
            if self.game_lang == 'en':
                leader_info = f"""
                    Leader's Additional Tasks: As the current round leader, you need to complete these extra tasks, you must strictly follow the format in completing these tasks:
                    1. Start with "TeamSelection:", list players other than yourself, you want to include in this mission (you will be automatically added to the team later).
                       - You must select exactly {required_team_size-1} players from {', '.join(other_players)}
                       - Format example: If you need to select 2 players, you may output in the following format: "TeamSelection: P3 P5"

                    2. Start with "MagicTarget:" to specify a target for the magic token
                       - The target must be one of your selected team members in TeamSelection or yourself {self.id}
                       - Format example: "MagicTarget: P1"
                    
                    Note:
                    - You choice must be based on your analysis and thinking.
                    - If you are Blue team, you should choose a player who you think is reliable (other blue team members), and be consistent with your speech as much as possible.
                    - If you are Red team, you can consider a strategy to confuse the blue team, but you must strictly follow the format and number of choices.
                    """
                
                leader_task = """
                    TeamSelection:
                    [List exactly {required_team_size-1} other player IDs here, space-separated, e.g., P1 P5 ]
                    
                    MagicTarget:
                    [Specify one player ID from your TeamSelection or your own ID]
                    """
            else:
                leader_info = f"""
                    队长附加任务：作为本轮队长，你需要额外完成以下任务：
                    1. 在"请按以下格式输出的"提示之后，以"TeamSelection:"开头列出你要选择加入此次任务的队员（不包括你自己）。
                      - 你必须选择恰好 {required_team_size-1} 名其他玩家 {', '.join(other_players)}，不能多也不能少（因为你自己会自动加入队伍）
                      - 示例格式：如果需要选择2名队员，可输入 "TeamSelection: P1 P5"， 用空格分隔

                    2. 在"请按以下格式输出的"提示之后，以"MagicTarget:"开头指定一名魔法指示物目标
                      - 目标必须是你选择的队员之一或你自己 {self.id}
                      - 示例格式："MagicTarget: P1"
                    
                    请记住：
                    - 你的选择应该基于你的分析和策略
                    - 如果你是蓝方，应该选择你认为可信的玩家，并且要尽可能与你发言中说自己要带的人一致，否则会降低你的信任度
                    - 如果你是红方，可以考虑混淆视听的策略，但必须严格遵守数量和格式的规范。
                    """
                
                leader_task = """
                    TeamSelection:
                    [在这里列出恰好 {required_team_size-1} 名其他玩家的ID，用空格分隔， 如 P1 P5 ]
                    
                    MagicTarget:
                    [在这里从队伍成员（包括你自己）中选择1名作为魔法目标, 用ID（如P1）表示]
                """

        if self.role == 'red':
            if self.game_lang == 'en':
                prompt_info = f"""
                    You are playing the Quest board game. Your ID is {self.id}, you are on the {self.role} team. Current round: {game_state['round']+1}. Game history:
                    
                    {game_history}

                    Your ID: {self.id}

                    Your teammates' IDs: {','.join(self.team_mates)}

                    Current round leader ID: {game_state['leader_id']}

                    All players: P1 P2 P3 P4 P5
                    
                    Recent information and chat history:
                    {self._get_current_memory()}

                    Personal analysis history review:
                    {self._get_summary_history()}

                    Game rules review:
                    {self.rules_text}

                    Your playing strategy style: {self.strategy}

                    Please complete the following tasks:

                    Note: Please limit your thinking depth to no more than 5 layers and keep it under 500 words, put your analysis in the Summary section.
                    
                    1. As a red team player, please analyze the following content and output your analysis starting with "Summary:":
                    - Evaluate the current game situation, team formations, and your team's trust level in other blue players' views
                    - Consider whether to hide your identity to build trust or create confusion to reduce blue team players' trust
                    - Plan your strategy for the upcoming rounds
                    Please note these rules when analyzing:
                    - Morgan Le Fay can ignore magic token restrictions and can still play fail cards even when targeted by magic
                    - A regular red player must play success cards when targeted by magic
                    - Players who have been leader cannot be chosen as leader again

                    2. Generate your next speech:
                    Start with "NextSpeech:" and give a 100-200 word speech. You can be strategic and misleading, never reveal your or your teammate's identity.
                    Make your speech unique and entertaining within reasonable bounds, avoid copying others' speeches, and do not include any thinking process in the speech
                    """
                
                prompt_task = f"""
                    Please output in the following format:
                    Summary:
                    [Output your situation analysis here, within 500 words]

                    NextSpeech:
                    [Output your next round speech here, 100-200 words. Base it on your analysis and match your strategy style {self.strategy} and character trait {self.character}]
                    """
            else:
                prompt_info = f"""
                        你正在玩Quest桌游，你的ID是{self.id}, 身份是{self.role}阵营玩家。当前游戏进行到了第{game_state['round']+1}轮，历史局势：
                        
                        {game_history}

                        你自己的ID：{self.id}

                        你队友的ID：{','.join(self.team_mates)}

                        当前轮队长ID：{game_state['leader_id']}

                        场上所有玩家：P1 P2 P3 P4 P5

                        最近信息与对话记录，如果是第一轮，则玩家聊天记录为空：
                        {self._get_current_memory()}

                        个人历史分析回顾：
                        {self._get_summary_history()}

                        游戏规则回顾：
                        {self.rules_text}

                        你的游玩策略风格: {self.strategy}

                        请完成以下任务：

                        提醒：请限制你思考的深度，不要超过5层，不要超过500字。
                        
                        一、 分析并总结当前局势：
                        如果你是蓝方玩家，请综合分析以下内容，并以"Summary："开头输出你的分析：
                        1. 根据所有以往任务表现、组队选择、队长转移、魔法指示物使用等去推理可疑玩家。
                        2. 判断摩根勒菲是否被使用魔法，摩根勒菲可以无视魔法指示物的限制，正常出失败牌，而另一名普通红方玩家在被使用魔法后，只能出成功牌。
                        3. 已经当过队长的人，不可以再重复被选为队长，在分析历史队长转移信息，请注意这一点。
                        4. 要结合对其它玩家的身份推测，来判断是否相信他们的发言，还是对他们的发言进行反向推理。
                        5. 你当前的怀疑对象及其依据。由于场上只有两个红方，所以你的核心怀疑对象不应超过两个。
                        6. 你要尽可能尝试说服其它的蓝方玩家，让他们相信你是蓝方玩家，并带你做任务。
                        7. 对后续游戏的策略计划。
                        
                        二、 更新对其他玩家的身份猜测，并以"Guess："开头输出你的猜测：
                        从除了你自己ID之外的其它玩家中猜测最有可能是红方的0-2个玩家。
                        然后，针对剩下的玩家，判断他们是蓝方还是不能确定。
                        如果当前是第一轮，信息不足的情况下，你可以不认为任何玩家是红方。
                        
                        三、 生成下一轮发言：
                        用"NextSpeech:"开头给出一段100-200个字的发言，相信和怀疑的目标尽可能与你的怀疑清单一致。
                        如果这是第一轮，信息不足的情况下，你也可以不发表对任何人的相信和怀疑。
                        发言在允许的范围内个性化，增加游戏的趣味性,尤其是不能跟别人发一模一样的话
                        """
                
                prompt_task = f"""
                        请按以下格式输出：
                        Summary:
                        [在这里输出500字以内的局势和策略分析]
                        
                        Guess:
                        // 对除自己外的所有玩家的猜测，值只能是"red"、"blue"或"unknown"， 示例：
                        {{"P1": "blue", "P2": "red", "P3": "unknown"}}

                        NextSpeech:
                        [在这里输出下一轮你的发言内容，字数100-200字。发言要基于你的分析，你的身份猜测，符合你的策略风格{self.strategy}和性格特点{self.character}]
                        """
            
            if is_leader:
                prompt = "\n".join([prompt_info, leader_info, prompt_task, leader_task])
            else:
                prompt = "\n".join([prompt_info, prompt_task])
                
        else:
            if self.game_lang == 'en':
                prompt_info = f"""
                    You are playing the Quest board game. Your ID is {self.id}, you are on the {self.role} team. Current round: {game_state['round']+1}. Game history:
                    
                    {game_history}

                    Your ID: {self.id}

                    Current round leader ID: {game_state['leader_id']}

                    All players: P1 P2 P3 P4 P5

                    Recent information and chat history (empty if this is the first round):
                    {self._get_current_memory()}

                    Personal analysis history review:
                    {self._get_summary_history()}

                    Game rules review:
                    {self.rules_text}

                    Your playing strategy style: {self.strategy}

                    Please complete the following tasks:

                    Note: Please limit your thinking depth to no more than 5 layers and keep it under 500 words, put your analysis in the Summary section.
                    
                    1. Analyze and summarize the current situation:
                    As a blue team player, please analyze the following content and output your analysis starting with "Summary:":
                    1. Analyze suspicious players based on all past task performances, team selections, leader transfers, and magic token usage.
                    2. Determine if Morgana has been targeted by magic - Morgana can ignore magic token restrictions and still play fail cards, while other red players must play success cards when targeted.
                    3. Note that players who have been leader cannot be chosen as leader again when analyzing leader transfer history.
                    4. Consider whether to trust other players' speeches or analyze them in reverse based on your identity guesses.
                    5. List your current suspects and reasoning. Since there are only two red players, you should not have more than two main suspects.
                    6. Try to convince other blue players that you are on the blue team and get them to include you in missions.
                    7. Plan your strategy for upcoming rounds.
                    
                    2. Update your guesses about other players' identities, starting with "Guess:":
                    Guess 0-2 players most likely to be red from all players except yourself.
                    Then determine if the remaining players are blue or uncertain.
                    If this is the first round with insufficient information, you may not suspect anyone of being red.
                    
                    3. Generate your next speech:
                    Start with "NextSpeech:" and give a 100-200 word speech. Your trust and suspicions should align with your guess list.
                    If this is the first round with insufficient information, you may not express trust or suspicion of anyone.
                    Make your speech unique and entertaining within reasonable bounds, avoid copying others' speeches, and do not include any thinking process in the speech
                    """

                prompt_task = f"""
                    Please output in the following format:
                    Summary:
                    [Output your situation analysis here, within 500 words]
                    
                    Guess:
                    // Your guesses for all players except yourself, values can only be "red", "blue" or "unknown", example:
                    {{"P1": "blue", "P2": "red", "P3": "unknown"}}

                    NextSpeech:
                    [Output your next round speech here, 100-200 words. Base it on your analysis, your identity guesses, and match your strategy style {self.strategy} and character trait {self.character}]
                    """
            else:
                prompt_info = f"""
                    你正在玩Quest桌游，你的ID是{self.id}, 身份是{self.role}阵营玩家。当前游戏进行到了第{game_state['round']+1}轮，历史局势：
                    
                    {game_history}

                    你自己的ID：{self.id}

                    当前轮队长ID：{game_state['leader_id']}

                    场上所有玩家：P1 P2 P3 P4 P5

                    最近信息与对话记录，如果是第一轮，则玩家聊天记录为空：
                    {self._get_current_memory()}

                    个人历史分析回顾：
                    {self._get_summary_history()}

                    游戏规则回顾：
                    {self.rules_text}

                    你的游玩策略风格: {self.strategy}

                    请完成以下任务：

                    提醒：请限制你思考的深度，不要超过5层，不要超过500字。
                    
                    一、 分析并总结当前局势：
                    如果你是蓝方玩家，请综合分析以下内容，并以"Summary："开头输出你的分析：
                    1. 根据所有以往任务表现、组队选择、队长转移、魔法指示物使用等去推理可疑玩家。
                    2. 判断摩根勒菲是否被使用魔法，摩根勒菲可以无视魔法指示物的限制，正常出失败牌，而另一名普通红方玩家在被使用魔法后，只能出成功牌。
                    3. 已经当过队长的人，不可以再重复被选为队长，在分析历史队长转移信息，请注意这一点。
                    4. 要结合对其它玩家的身份推测，来判断是否相信他们的发言，还是对他们的发言进行反向推理。
                    5. 你当前的怀疑对象及其依据。由于场上只有两个红方，所以你的核心怀疑对象不应超过两个。
                    6. 你要尽可能尝试说服其它的蓝方玩家，让他们相信你是蓝方玩家，并带你做任务。
                    7. 对后续游戏的策略计划。
                    
                    二、 更新对其他玩家的身份猜测，并以"Guess："开头输出你的猜测：
                    从除了你自己ID之外的其它玩家中猜测最有可能是红方的0-2个玩家。
                    然后，针对剩下的玩家，判断他们是蓝方还是不能确定。
                    如果当前是第一轮，信息不足的情况下，你可以不认为任何玩家是红方。
                    
                    三、 生成下一轮发言：
                    用"NextSpeech:"开头给出一段100-200个字的发言，相信和怀疑的目标尽可能与你的怀疑清单一致。
                    如果这是第一轮，信息不足的情况下，你也可以不发表对任何人的相信和怀疑。
                    发言在允许的范围内个性化，增加游戏的趣味性,尤其是不能跟别人发一模一样的话
                    """
            
                prompt_task = f"""
                        请按以下格式输出：
                        Summary:
                        [在这里输出500字以内的局势和策略分析]
                        
                        Guess:
                        // 对除自己外的所有玩家的猜测，值只能是"red"、"blue"或"unknown"， 示例：
                        {{"P1": "blue", "P2": "red", "P3": "unknown"}}

                        NextSpeech:
                        [在这里输出下一轮你的发言内容，字数100-200字。发言要基于你的分析，你的身份猜测，符合你的策略风格{self.strategy}和性格特点{self.character}]
                        """
            
            if is_leader:
                prompt = "\n".join([prompt_info, leader_info, prompt_task, leader_task])
            else:
                prompt = "\n".join([prompt_info, prompt_task])

        # 生成响应，如果是队长且响应不符合要求则重试
  
        max_retries = 5
        retry_count = 0
        while True:
            if self.model_api.startswith('ollama'):
                response = llm([HumanMessage(content=prompt)]).get("content", "")
            else:
                response = llm([HumanMessage(content=prompt)]).content

            print(f"队长 {self.id} 的响应: {response}")
            
            if not is_leader or validate_leader_response(response):
                break
            
            retry_count += 1
            if retry_count >= max_retries:
                print(f"警告：队长 {self.id} 的响应在 {max_retries} 次尝试后仍不符合要求")
                break
            
            print(f"队长 {self.id} 的响应不包含必要元素，正在重试 ({retry_count}/{max_retries})")

        # 解析响应
        try:
            # 清理和标准化响应文本
            response = response.strip()

            if "Summary:" not in response:
                print(f"Invalid response format: {response}")
                return response

            # 提取总结部分
            parts = response.split("Summary:")[1]
            summary_part = parts.split("Guess:" if "Guess:" in parts else "NextSpeech:")[0].strip()
            
            # 提取下一轮发言
            if "NextSpeech:" in response:
                next_speech = response.split("NextSpeech:")[1]

                if 'TeamSelection:' in next_speech:
                    next_speech = next_speech.split("TeamSelection:")[0].strip()
                    self.next_speech = next_speech
                else:
                    self.next_speech = next_speech.strip()

            if self.role == 'blue':
                # 只有蓝方需要猜测
                guess_part = response.split("Guess:")[1].split("NextSpeech:")[0].strip()
                
                # 清理 JSON 字符串
                guess_part = guess_part.replace("'", '"')
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

            return response
            
        except Exception as e:
            print(f"解析LLM响应失败：{e}\n响应内容：{response}")
            self.summary_memory = response
            return response


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
            return self.selected_team

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
                team_mates = [p for p in available_players if p.id in self.team_mates]
                if team_mates:
                    return random.choice(team_mates).id
            
            # 随机选择一个非队友玩家
            non_team_players = [p for p in available_players if p.id not in self.team_mates]
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
    _current_lang = 'zh'  # 静态语言设置
      
    def __init__(self, output: GameOutput, human_player_id: str = "P5", 
                 test_mode: bool = False, p5_is_morgan: bool = False,
                 player_models: Dict[str, str] = None,
                 random_team: bool = True,
                 player_teams: Dict[str, str] = None,
                 lang: str = None):
        self.output = output
        self.test_mode = test_mode
        self.p5_is_morgan = p5_is_morgan
        self.human_player_id = None if test_mode and not human_player_id else human_player_id
        self.player_models = player_models or {}
        self.random_team = random_team
        self.player_teams = player_teams or {}
        # 使用传入的语言设置，如果没有则从输出对象获取
        self.lang = lang or getattr(output, 'lang', 'zh')
        
        # 添加游戏历史记录表头的中英文版本
        self.game_history_headers = {
            'zh': "| 轮次 | 队长 | 任务队员 | 魔法目标 | 任务结果 | 失败票数 |\n|------|------|----------|-----------|----------|----------|",
            'en': "| Round | Leader | Team | Magic Target | Result | Fails |\n|--------|---------|------|--------------|---------|--------|"
        }
        self.game_history_header = self.game_history_headers.get(self.lang, self.game_history_headers['zh'])
        
        print(f"Initializing game in {'test' if test_mode else 'normal'} mode")
        print(f"Human player ID: {self.human_player_id}")
        
        # 初始化基本属性
        self.round = 0
        self.blue_wins = 0
        self.red_wins = 0
        self.final_winner = None
        self.task_sizes = [2, 2] if test_mode else [2, 3, 2, 3, 3]
        print(f"初始化任务大小: {self.task_sizes}")
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
        
        # 读取游戏规则
        try:
            rules_file = "game_rules_en.md" if self.lang == 'en' else "game_rules.md"
            with open(rules_file, encoding="utf-8") as f:
                self.rules_text = f.read()
        except Exception as e:
            error_msg = "Unable to read game rules." if self.lang == 'en' else "无法读取游戏规则。"
            self.rules_text = error_msg
            self.output.send_message(f"读取游戏规则失败: {e}", "error")

        # 添加游戏历史记录
        self.game_history = []

    def _initialize_players(self, human_player_id: str):
        """初始化玩家列表"""
        players = []
        
        for i in range(1, 6):
            player_id = f"P{i}"
            is_human = (player_id == human_player_id)
            player = Player(player_id, is_human=is_human, output=self.output, model_api=self.player_models.get(f"P{i}"), game_lang=self.lang)
            players.append(player)
        
        return players

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
            player.rules_text = ""  # 存储原始规则文本
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
        
        # 保持当前语言设置
        current_lang = self.lang
        self.lang = current_lang

    def discussion_phase(self):
        """讨论阶段"""
        self.output.send_message(self.get_message('discussion_phase'), 'action')
        
        # 获取所有玩家的发言
        speeches = {}  # 只存储每个玩家的发言内容
        for player in self.players:
            if player.is_human:
                prompt = f"玩家 {player.id} 的回合：请发言（你是{'红方' if player.role == 'red' else '蓝方'}）"
                if player.role == "red":
                    prompt += f"，你的队友是 {player.team_mates}"
                print(f"Waiting for human player {player.id} speech input")  # 调试信息
                speech = self.output.get_player_input(prompt, player.id)
                print(f"Received speech: {speech}")  # 调试信息
            else:
                speech = player.next_speech#generate_speech(game_state, self.llm)
            speeches[player.id] = speech  # 存储发言内容
            self.output.send_message(f"{player.id} 说：{speech}", 'info')
            # 广播发言到所有玩家记忆（这部分保持不变，因为玩家需要记住所有人的发言）
            for listener in self.players:
                listener.add_event(f"{player.id}发言：{speech}")
        return speeches
    

    def run_round(self):
        """执行单轮游戏"""
        print(f"第 {self.round + 1} 轮的任务大小数组: {self.task_sizes}")
        print(f"当前轮次索引 {self.round}, 需要的队员数: {self.task_sizes[self.round]}")
        
        # 在测试模式下检查轮数
        if self.test_mode and self.round >= 2:
            return False

        # 保存当前的队长作为上一任队长
        #self.last_leader = self.current_leader_index
        """运行一轮游戏"""
        # 第一轮开始时选择首个队长
        if self.round == 0:
            if self.test_mode:
                # 测试模式下，P5作为第一个队长
                self.current_leader_index = 4  # P5的索引
                self.leaders.append("P5")
                self.output.send_message(self.get_message('random_leader', self.round + 1, self.players[self.current_leader_index].id), 'action')
            else:
                # 随机选择第一个队长
                first_leader = random.choice(self.players)
                self.current_leader_index = self.players.index(first_leader)
                self.leaders.append(first_leader.id)
                self.output.send_message(self.get_message('random_leader', self.round + 1, first_leader.id), 'action')

        #Generate initial summaries and speeches for all players
        if self.round == 0:
            ai_summaries = self.run_ai_thinking(self.get_game_state(), self.task_sizes[self.round])
        
        # 讨论阶段
        round_speeches = self.discussion_phase()

        #Generate thinking right after the first speech
        if self.round == 0:
            ai_summaries = self.run_ai_thinking(self.get_game_state(), self.task_sizes[self.round])
        
        # 调试日志：检查队长的 selected_team
        print(f"队长 {self.players[self.current_leader_index].id} 在讨论阶段后的 selected_team: {self.players[self.current_leader_index].selected_team}")
        
        # 队长选择队伍
        leader = self.players[self.current_leader_index]
        team = leader.propose_team(
            required_size=self.task_sizes[self.round],
            llm=leader.llm
        )
        print(f"propose_team 返回的队伍: {team}")
        self.output.send_message(self.get_message('team_selected', leader.id, ', '.join(team)), 'info')

        # 队长选择魔法指示物目标
        if leader.is_human:
            prompt = f"选择要施加魔法指示物的玩家（输入玩家ID）："
            target_id = self.output.get_player_input(prompt, leader.id)
            amulet_target = next((p for p in self.players if p.id == target_id), None)
        else:
            # AI队长随机选择非自己玩家：team 中存储的是玩家ID，此处转换为对应的 Player 对象
            # candidates = [p for tid in team for p in self.players if p.id == tid and p != leader]
            amulet_target = next((p for p in self.players if p.id == leader.magic_target), None)
            
            #self.players[leader.magic_target] #random.choice(candidates) if candidates else None
        
        # 因为 amulet 是强制使用的，直接设置
        amulet_target.has_amulet = True
        self.last_amulet_player = amulet_target.id
        self.output.send_message(self.get_message('magic_used', leader.id, amulet_target.id), 'action')

        # 执行任务
        success_votes = 0
        fail_votes = 0
        for member in team:
            # 将成员ID转换为Player对象
            player_obj = next((p for p in self.players if p.id == member), None)
            if player_obj:
                vote = self.get_mission_vote(player_obj)
            else:
                raise ValueError(f"无效的成员ID: {member}")  # 默认处理无效成员
            if vote:
                success_votes += 1
            else:
                fail_votes += 1

        self.output.send_message(self.get_message('success_votes', success_votes), 'info')
        self.output.send_message(self.get_message('fail_votes', fail_votes), 'info')
        self.last_fail_votes = fail_votes

        # 判断任务结果
        required_fails = 1  # 默认1票失败
        if self.round == 3:  # 第四轮
            required_fails = 1  # 根据新规则修改
        
        success = fail_votes < required_fails

        # 更新胜负次数
        if success:
            self.blue_wins += 1
            self.output.send_message(self.get_message('mission_success', self.blue_wins, self.red_wins), "result")
        else:
            self.red_wins += 1
            self.output.send_message(self.get_message('mission_fail', self.blue_wins, self.red_wins), "result")

        self.last_team = team
        self.last_result = "成功" if success else "失败"

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
        
        self.last_leader = self.current_leader_index
        self.current_leader_index = self.players.index(next_leader_obj)
        # 记录新队长的id（确保每位玩家在一个周期只做一次队长）
        if next_leader_obj.id not in self.leaders:
            self.leaders.append(next_leader_obj.id)
        self.output.send_message(self.get_message('next_leader', leader.id, next_leader_obj.id), 'info')
        
        # 将队长选择信息添加到所有玩家的记忆中
        for player in self.players:
            player.add_event(f"{leader.id}选择了{self.players[self.current_leader_index].id}作为下一任队长")
        
        # 第一部分：写入基本游戏信息和玩家发言
        try:
            current_round = GameRound(
                game=self.current_game,
                round_number=self.round,
                leader_id=leader.id,
                team_members='_'.join([','.join(team), amulet_target.id]),
                fail_votes=fail_votes,
                result='success' if success else 'fail'
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
        
        # 确保 team 列表中的元素格式正确

        self.update_game_history(
            leader_id=self.players[self.last_leader].id, 
            team=team,
            magic_target=amulet_target.id, 
            result='success' if success else 'fail', 
            fail_votes=fail_votes
        )
        
        # AI开始思考：并行调用生成总结，提高效率
        if self.red_wins == 3:
            self.output.send_message("红方已获得三次胜利，蓝方玩家进行最后分析...", "action")
        elif self.blue_wins == 3:
            self.output.send_message(self.get_message('blue_victory'), "action")
            return False
        else:
            self.output.send_message(self.get_message('ai_thinking'), 'info')
        
        ai_summaries = self.run_ai_thinking(self.get_game_state(), self.task_sizes[min(self.round+1, 4)])
        
        # 第二部分：写入AI的思考和猜测
        try:
            # 获取刚才创建的回合记录
            current_round = GameRound.query.filter_by(
                game_id=self.current_game.id,
                round_number=self.round
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
        
        self.round += 1
        
        # 清除魔法指示物
        for p in self.players:
            p.current_memory.clear()
            p.has_amulet = False
        
        if self.blue_wins >= 3 or self.red_wins >= 3 or self.round >= len(self.task_sizes):
            return False
        
        return True

    def get_game_state(self) -> Dict:
        return {
            "blue_wins": self.blue_wins,
            "red_wins": self.red_wins,
            "round": self.round,
            "leader_id": self.players[self.current_leader_index].id,
            "game_history": self.get_formatted_history(),
            "last_team": self.last_team,
            "last_result": self.last_result,
            "last_fail_votes": getattr(self, 'last_fail_votes', 0),  # 添加失败票数
            "last_leader_id":self.players[self.current_leader_index].id,
            "required_team_size": self.task_sizes[self.round]
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
                    return False #random.random() > 1.0  # 90%概率投失败
                
                # 普通红队玩家如果被施加魔法指示物必须投成功
                if player.has_amulet:
                    return True
                
                return False #random.random() > 0.9  # 90%概率投失败
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

        self.initialize_ai_memory()
        
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
        self.output.send_message(self.get_message('game_over'), "result")
        self.output.send_message(self.get_message('final_score', self.blue_wins, self.red_wins), "result")
        self.output.send_message(self.get_message('final_winner', self.final_winner), "result")
        self.output.send_message(self.get_message('reveal_roles'), "info")
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
        self.output.send_message(self.get_message('final_identification_phase'), 'action')
        self.output.send_message(self.get_message('blue_team_identify'), 'info')
        
        # 获取实际红方玩家集合（统一大写）
        actual_reds = {p.id.upper() for p in self.players if p.role == "red"}
        
        blue_guesses = {}
        for player in self.players:
            if player.role == "blue":
                if player.is_human:
                    prompt = self.get_message('identify_prompt')
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
                self.output.send_message(self.get_message('wrong_identification', pid, guess_set, actual_reds), "result")
                self.final_winner = "red"
                return
        
        # 条件2：所有蓝队玩家猜测的并集必须覆盖所有实际红方
        union_of_guesses = set()
        for guess_set in blue_guesses.values():
            union_of_guesses.update(guess_set)
        if union_of_guesses == actual_reds:
            self.final_winner = "blue"
            self.output.send_message(self.get_message('blue_identification_success'), "action")
        else:
            self.final_winner = "red"
            self.output.send_message(self.get_message('red_identification_success'), "action")

    def assign_roles(self):
        print("Assigning roles with:", {
            'random_team': self.random_team,
            'player_teams': self.player_teams
        })
        
        # 检查是否使用手动分配的队伍
        if self.player_teams and not self.random_team:
            blue_players = [p for p in self.players if self.player_teams.get(p.id) == 'blue']
            red_players = [p for p in self.players if self.player_teams.get(p.id) == 'red']

            print("Manual team assignment:", {
                'blue_players': [p.id for p in blue_players],
                'red_players': [p.id for p in red_players]
            })
            
            # 验证分配是否合法（3蓝2红）
            if len(blue_players) != 3 or len(red_players) != 2:
                self.output.send_message("错误：必须分配3个蓝方和2个红方玩家", "error")
                # 回退到随机分配
                blue_players = random.sample(self.players, 3)
                red_players = [p for p in self.players if p not in blue_players]
        else:
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
                # 正常随机分配阵营
                blue_players = random.sample(self.players, 3)
                red_players = [p for p in self.players if p not in blue_players]
        
        # 设置所有玩家的角色和队友
        for p in self.players:
            if p in blue_players:
                p.role = "blue"
                p.team_mates = []  # 蓝方没有队友
            else:
                p.role = "red"
                # 确保队友列表只包含其他红方玩家的ID
                p.team_mates = [other.id for other in red_players if other.id != p.id]
            
            # 如果是红方玩家，更新其猜测表
            if p.role == "red":
                for mate in p.team_mates:
                    p.guess[mate] = "red"
                for mate in blue_players:
                    p.guess[mate.id] = "blue"
            print(f"玩家 {p.id} 的角色是 {p.role}，队友是 {p.team_mates}")
        
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
            team_mates=','.join(morgan.team_mates),
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
                team_mates=','.join(p.team_mates),
                morgan=False
            )
            db.session.add(player_record)
        
        db.session.commit()
        
        if morgan.is_human:
            self.output.send_message("你是摩根勒菲，你可以无视魔法指示物的限制", "action")

    def initialize_ai_memory(self):
        """初始化所有 AI 玩家的规则文本"""
        for player in self.players:
            if not player.is_human:
                player.rules_text = self.rules_text

    def update_game_history(self, leader_id: str, team: List[str], magic_target: Optional[str], result: str, fail_votes: int):
        """
        更新游戏历史记录
        """
        round_num = len(self.game_history) + 1
        team_str = ', '.join(sorted(team))  # 排序以保持一致性
        magic_str = magic_target if magic_target else '-'
        if self.lang == 'en':
            result_str = "Success(Blue wins)" if result == "success" else "Fail(Red wins)"
        else:
            result_str = "成功(蓝方胜)" if result == "success" else "失败(红方胜)"
        
        history_entry = f"| {round_num} | {leader_id} | {team_str} | {magic_str} | {result_str} | {fail_votes} |"
        self.game_history.append(history_entry)

    def get_formatted_history(self) -> str:
        """
        获取格式化的游戏历史记录
        """
        if not self.game_history:
            return "No game history yet." if self.lang == 'en' else "游戏刚刚开始，还没有历史记录。"
            
        return self.game_history_header + '\n' + '\n'.join(self.game_history)

    def run_ai_thinking(self, game_state: Dict, team_size: int) -> Dict[str, str]:
        """
        运行所有 AI 玩家的思考过程
        Args:
            game_state: 当前游戏状态
        Returns:
            Dict[str, str]: AI 玩家 ID 到其思考结果的映射
        """
        ai_summaries = {}
        with ThreadPoolExecutor() as executor:
            # 提交所有非人类玩家的生成总结任务
            future_to_player = {
                executor.submit(player.generate_summary, game_state=game_state, llm=player.llm, team_size=team_size): player.id
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
        return ai_summaries

    def get_message(self, key, *args):
        """获取当前语言的消息"""
        message_template = GAME_MESSAGES.get(self.lang, GAME_MESSAGES['zh'])[key]
        return message_template.format(*args) if args else message_template

def set_model_api(model: str):
    """设置全局模型 API"""
    global _model_api
    _model_api = model

def get_model_api():
    """获取当前模型 API"""
    return _model_api

def clean_player_id(player_id: str) -> Optional[str]:
    """清理玩家ID前后的特殊符号，如果不是有效的玩家ID则返回None
    例如: 
    - 'P1.' -> 'P1'
    - ' P2,' -> 'P2'
    - 'P3。' -> 'P3'
    - 'llm' -> None
    - 'player' -> None
    """
    match = re.match(r'^[^P\d]*P?(\d)[^0-9]*$', player_id.strip())
    if match and 1 <= int(match.group(1)) <= 5:  # 确保数字在1-5范围内
        return f"P{match.group(1)}"
    return None

def parse_team_selection(self, response: str) -> List[str]:
    """解析队伍选择响应"""
    try:
        team_match = re.search(r'TeamSelection:\s*(.*?)(?:\n|$)', response)
        if not team_match:
            return []
        
        # 分割、清理玩家ID并过滤掉无效ID
        selected_players = [pid for pid in 
            (clean_player_id(p) for p in team_match.group(1).split())
            if pid is not None]
        return selected_players
    except Exception as e:
        print(f"解析队伍选择失败: {str(e)}")
        return []
        
def parse_magic_target(self, response: str) -> str:
    """解析魔法指示物目标"""
    try:
        target_match = re.search(r'MagicTarget:\s*(.*?)(?:\n|$)', response)
        if not target_match:
            return ""
        return clean_player_id(target_match.group(1))
    except Exception as e:
        print(f"解析魔法目标失败: {str(e)}")
        return ""

if __name__ == "__main__":
    terminal_output = TerminalOutput()
    game = AvalonSimulator(output=terminal_output)
    game.run_game()
