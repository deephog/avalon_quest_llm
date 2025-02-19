import json
from typing import Dict, Any

class PromptSystem:
    """统一管理游戏提示的系统"""
    
    def __init__(self, lang='zh', version='v2.1'):
        self.version = version
        self.template_versions = {
            'v2.1': self._load_v2_1_templates,
            'v2.0': self._load_v2_0_templates
        }
        self.templates = self.template_versions[version]()
    
    def _load_v2_1_templates(self):
        """加载多语言模板"""
        return {
            'zh': {
                'base': self._zh_base_template(),
                'red': self._zh_red_template(),
                'blue': self._zh_blue_template()
            },
            'en': {
                'base': self._en_base_template(),
                'red': self._en_red_template(),
                'blue': self._en_blue_template()
            }
        }
    
    def _load_v2_0_templates(self):
        """加载旧版本模板（兼容用）"""
        return {
            'zh': {
                'base': self._zh_base_template(),
                'red': self._zh_red_template(),
                'blue': self._zh_blue_template()
            },
            'en': {
                'base': self._en_base_template(),
                'red': self._en_red_template(),
                'blue': self._en_blue_template()
            }
        }
    
    # 中文模板部分
    def _zh_base_template(self) -> Dict[str, str]:
        return {
            'role_info': "玩家ID：{player_id} | 阵营：{team}",
            'history_header': "## 游戏历史\n{history}",
            'rules_header': "## 核心规则\n{rules}",
            'output_format': "## 输出格式要求\n{format}",
            'leader_section': "## 队长专属任务\n{tasks}"
        }
    
    def _zh_red_template(self) -> Dict[str, Any]:
        return {
            'common': {
                'task': (
                    "## 主要任务\n"
                    "1. 隐藏身份并破坏任务\n"
                    "2. 误导蓝方玩家\n"
                    "3. 保护红方队友不被发现"
                ),
                'output': {
                    'summary': ("## 局势分析\n"
                               "分析当前游戏局势，重点指出蓝方玩家的可疑行为"),
                    'suspicions': ("## 怀疑列表\n"
                                  "对其他玩家的身份猜测（红方/蓝方/未知）"),
                    'speech': ("## 发言内容\n"
                              "100-200字的自然发言，符合{character}性格")
                }
            },
            'leader': {
                'task': (
                    "## 队长专属任务\n"
                    "1. 选择{required}名队员（不包括自己）\n"
                    "2. 指定魔法指示物目标\n"
                    "3. 制造合理的选择理由"
                ),
                'output': {
                    'team_selection': ["P1", "P3"],
                    'magic_target': "P1",
                    'rationale': "选择理由说明..."
                }
            }
        }
    
    def _zh_blue_template(self) -> Dict[str, Any]:
        # Implementation needed
        pass
    
    def _en_base_template(self) -> Dict[str, str]:
        # Implementation needed
        pass
    
    def _en_red_template(self) -> Dict[str, Any]:
        return {
            'common': {
                'task': (
                    "## Main Tasks\n"
                    "1. Hide identity and sabotage missions\n"
                    "2. Mislead Blue team players\n"
                    "3. Protect Red teammates from exposure"
                ),
                'output': {
                    'summary': "局势分析...",
                    'suspicions': "怀疑对象...",
                    'speech': "发言内容..."
                }
            },
            'leader': {
                'task': "选择{required}名队员...",
                'output': {
                    'team_selection': ["P1", "P3"],
                    'magic_target': "P1"
                }
            }
        }
    
    # 其他模板方法类似，限于篇幅省略...
    
    def build_prompt(self, role: str, is_leader: bool, **kwargs) -> str:
        """构建完整提示"""
        template = self.templates[self.lang]
        parts = [
            self._build_section('base.role_info', player_id=kwargs['player_id'], team=kwargs['team']),
            self._build_section('base.history_header', history=kwargs['history']),
            self._build_section('base.rules_header', rules=kwargs['rules'])
        ]
        
        role_data = template[role.lower()]
        if is_leader:
            parts.append(self._build_leader_section(role_data))
        else:
            parts.append(self._build_common_section(role_data))
            
        return '\n\n'.join(parts)
    
    def _build_section(self, key: str, **kwargs) -> str:
        """构建模板段落"""
        keys = key.split('.')
        section = self.templates[self.lang]
        for k in keys:
            section = section[k]
        return section.format(**kwargs) 