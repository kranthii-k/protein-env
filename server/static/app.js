document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Server Health Check
    async function checkServerHealth() {
        const indicator = document.getElementById('server-status');
        const statusText = indicator.querySelector('span');
        const versionBox = document.getElementById('version-box').querySelector('.stat-value');

        try {
            const response = await fetch('/health');
            if (response.ok) {
                const data = await response.json();
                indicator.classList.add('connected');
                statusText.textContent = 'Server Connected';
                versionBox.textContent = `v${data.version || '0.1'}`;
            } else {
                throw new Error('Server returned non-200');
            }
        } catch (error) {
            indicator.classList.remove('connected');
            statusText.textContent = 'Server Disconnected';
            versionBox.textContent = '--';
        }
    }

    // Check health immediately and every 10 seconds
    checkServerHealth();
    setInterval(checkServerHealth, 10000);
});

// Simulation button animation
window.runSimulation = function(taskId) {
    const pane = document.getElementById(taskId);
    const terminal = pane.querySelector('.mock-terminal');
    const originalContent = terminal.innerHTML;
    
    // Clear and simulate typing
    terminal.innerHTML = '<span class="cmd">Simulating OpenEnv agent run...</span><br>';
    
    setTimeout(() => {
        terminal.innerHTML = originalContent;
    }, 1500);
};
