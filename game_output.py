from abc import ABC, abstractmethod
from typing import Any, Optional

class GameOutput(ABC):
    @abstractmethod
    def send_message(self, message: str, msg_type: str = 'info') -> None:
        pass
    
    @abstractmethod
    def get_player_input(self, prompt: str, player_id: str) -> str:
        pass

class TerminalOutput(GameOutput):
    def send_message(self, message: str, msg_type: str = 'info') -> None:
        # 根据消息类型添加不同的格式
        if msg_type == 'result':
            print(f"\n=== {message} ===")
        elif msg_type == 'action':
            print(f"\n> {message}")
        else:
            print(message)
    
    def get_player_input(self, prompt: str, player_id: str) -> str:
        print(f"\n{prompt}")
        return input("> ").strip()

class WebSocketOutput(GameOutput):
    def __init__(self, socketio, game_manager):
        self.socketio = socketio
        self.game_manager = game_manager
        print("WebSocketOutput initialized")  # 调试信息
    
    def send_message(self, message: str, msg_type: str = 'info') -> None:
        print(f"Sending message: {message} (type: {msg_type})")  # 调试信息
        self.socketio.emit('game_update', {
            'message': message,
            'type': msg_type
        })
    
    def get_player_input(self, prompt: str, player_id: str) -> str:
        print(f"Waiting for input from player {player_id}")  # 调试信息
        self.send_message(prompt, 'input_prompt')
        input_text = self.game_manager.wait_for_input()
        print(f"Received input: {input_text}")  # 调试信息
        return input_text 