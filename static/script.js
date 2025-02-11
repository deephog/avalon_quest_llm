const socket = io();

// 处理游戏更新
socket.on('game_update', (data) => {
    const log = document.getElementById('game-log');
    const message = document.createElement('div');
    message.className = `log-message ${data.type}`;
    message.textContent = data.message;
    log.appendChild(message);
    log.scrollTop = log.scrollHeight;
});

// 处理输入请求
socket.on('input_request', () => {
    const inputSection = document.getElementById('input-section');
    if (inputSection) {
        inputSection.style.display = 'flex'; // 显示输入框
        const inputField = document.getElementById('player-input');
        if (inputField) {
            inputField.focus(); // 聚焦到输入框
        }
    } else {
        console.error("Input section not found");
    }
});

// 发送玩家输入
//document.getElementById('send-button').addEventListener('click', () => {
//    const input = document.getElementById('player-input');
//    if (input && input.value.trim()) {
//        socket.emit('player_input', { input: input.value.trim() });
//        input.value = '';
//        document.getElementById('input-section').style.display = 'none';
//    }
//});
document.getElementById('send-button').addEventListener('click', () => {
    const input = document.getElementById('player-input');
    if (input && input.value.trim()) {
        socket.emit('player_input', { input: input.value.trim() });
        // 不再隐藏输入框
        // 不再清空输入内容：input.value = '';
    }
});


document.getElementById('start-game').addEventListener('click', () => {
    console.log("Start Game button clicked"); // 调试信息
    socket.emit('start_game');
    document.getElementById('start-game').disabled = true;
});

socket.on('input_active', (data) => {
    const inputSection = document.getElementById('input-section');
    inputSection.style.border = data.active ? '2px solid #4CAF50' : 'none';
    document.getElementById('player-input').placeholder =
        data.active ? `第${data.round}轮请输入...` : '等待游戏阶段...';
});