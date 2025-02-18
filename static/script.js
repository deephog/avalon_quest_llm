let socket = io();
let currentLang = localStorage.getItem('gameLang') || 'zh'; // 保存在本地存储中

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

function startGame() {
    try {
        console.log('Starting game...');
        const includeHuman = document.getElementById('include-human-checkbox').checked;
        const randomTeam = document.getElementById('random-team-checkbox').checked;
        let simulationCount = parseInt(document.getElementById('simulation-count').value);
        
        // 验证模拟次数
        if (isNaN(simulationCount) || simulationCount < 1 || simulationCount > 100) {
            console.warn('Invalid simulation count, using default value 1');
            simulationCount = 1;
            document.getElementById('simulation-count').value = '1';
        }
        
        const playerModels = {};
        const playerTeams = {};
        
        for (let i = 1; i <= 5; i++) {
            playerModels[`P${i}`] = document.getElementById(`p${i}-model`).value;
            playerTeams[`P${i}`] = document.getElementById(`p${i}-team`).value;
        }
        
        socket.emit('start_game', {
            include_human: includeHuman,
            player_models: playerModels,
            random_team: randomTeam,
            player_teams: playerTeams,
            simulation_count: simulationCount,
            lang: currentLang  // 添加当前语言设置
        });
    } catch (error) {
        console.error('Error starting game:', error);
    }
}

function switchLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('gameLang', lang);
    socket.emit('switch_language', {lang: lang});
}

socket.on('input_active', (data) => {
    const inputSection = document.getElementById('input-section');
    inputSection.style.border = data.active ? '2px solid #4CAF50' : 'none';
    document.getElementById('player-input').placeholder =
        data.active ? `第${data.round}轮请输入...` : '等待游戏阶段...';
});