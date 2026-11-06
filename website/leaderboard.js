(() => {
  const root = document.getElementById('leaderboard');
  if (!root) return;
  const buttons = [...root.querySelectorAll('[data-leaderboard-task]')];
  const panels = [...root.querySelectorAll('[data-leaderboard-panel]')];
  buttons.forEach(button => button.addEventListener('click', () => {
    buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    panels.forEach(panel => { panel.hidden = panel.dataset.leaderboardPanel !== button.dataset.leaderboardTask; });
  }));
})();
