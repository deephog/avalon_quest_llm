from jsonschema import validate
from jsonschema.exceptions import ValidationError
import re

SCHEMA = {
    "common": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "minLength": 50,  # 至少50字分析
                "maxLength": 5000  # 最多500字
            },
            "next_speech": {
                "type": "string",
                "minLength": 50,
                "maxLength": 5000
            },
            "guess": {
                "type": "object",
                "patternProperties": {
                    "^P[1-5]$": {
                        "enum": ["red", "blue", "unknown"],
                        "description": "必须为red/blue/unknown"
                    }
                },
                "additionalProperties": False,  # 禁止额外属性
                "minProperties": 4  # 必须猜测至少4个玩家
            }
        },
        "required": ["summary", "next_speech", "guess"],
        "additionalProperties": False  # 禁止响应中出现未定义字段
    },
    "leader": {
        "type": "object",
        "properties": {
            "team_selection": {
                "type": "array",
                "items": {"pattern": "^P[1-5]$"},
                "minItems": 1,
                "maxItems": 2
            },
            "magic_target": {"pattern": "^P[1-5]$"}
        },
        "required": ["team_selection", "magic_target"]
    }
}

class ResponseValidator:
    @staticmethod
    def validate(response_data: dict, schema_type: str) -> bool:
        """验证响应数据结构
        Args:
            response_data: 解析后的响应字典
            schema_type: 验证模式 (common/leader)
        Returns:
            bool: 是否通过验证
        """
        try:
            validate(
                instance=response_data,
                schema=SCHEMA[schema_type]
            )
            return True
        except ValidationError as e:
            print(f"格式验证失败: {e.message} (路径: {e.json_path})")
            return False
        except KeyError:
            print(f"未知的验证模式: {schema_type}")
            return False

    def validate_common_response(self, data: dict) -> bool:
        """自定义普通响应验证逻辑"""
        # 基础模式验证
        if not self.validate(data, "common"):
            return False
        
        return True 