from jsonschema import validate
from jsonschema.exceptions import ValidationError

SCHEMA = {
    "common": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "next_speech": {"type": "string"},
            "guess": {
                "type": "object",
                "patternProperties": {
                    "^P[1-5]$": {"enum": ["red", "blue", "unknown"]}
                }
            }
        },
        "required": ["summary", "next_speech"]
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