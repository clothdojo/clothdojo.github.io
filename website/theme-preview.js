// Explicit design comparison only. The ordinary homepage keeps its light theme.
(() => {
  const requested = new URL(location.href).searchParams.get('theme');
  if (!['light', 'dark'].includes(requested)) return;
  const root = document.documentElement;
  root.dataset.theme = requested;
  root.classList.add('theme-study');
  document.addEventListener('DOMContentLoaded', () => {
    const bar = document.createElement('aside');
    bar.className = 'theme-study-controls';
    bar.setAttribute('aria-label', 'Homepage color study');
    const label = document.createElement('span');
    label.textContent = 'Color study';
    bar.append(label);
    const buttons = ['light', 'dark'].map(theme => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = theme === 'light' ? 'Light' : 'Dark';
      button.dataset.themeChoice = theme;
      button.addEventListener('click', () => select(theme));
      bar.append(button);
      return button;
    });
    function select(theme) {
      root.dataset.theme = theme;
      buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.themeChoice === theme)));
      const url = new URL(location.href);
      url.searchParams.set('theme', theme);
      history.replaceState(null, '', url);
      document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#151a18' : '#fafbf9';
    }
    document.body.append(bar);
    select(requested);
  });
})();
