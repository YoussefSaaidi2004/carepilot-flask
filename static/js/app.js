(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];

  // Mobile navigation: close with Escape and when a menu item is chosen.
  const sidebar = $('#sidebar');
  const mobileMenu = $('#mobile-menu');
  if (sidebar && mobileMenu) {
    mobileMenu.addEventListener('click', () => {
      const open = sidebar.classList.toggle('open');
      mobileMenu.setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        sidebar.classList.remove('open');
        mobileMenu.setAttribute('aria-expanded', 'false');
      }
    });
    $$('.nav-link', sidebar).forEach(link => link.addEventListener('click', () => sidebar.classList.remove('open')));
    document.addEventListener('click', (event) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(event.target) && !mobileMenu.contains(event.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  // Radio cards and segmented controls mirror the selected state visibly.
  $$('input[type="radio"]').forEach(input => {
    const update = () => {
      const group = input.closest('[role="radiogroup"]');
      if (!group) return;
      $$('label', group).forEach(label => {
        const radio = $('input[type="radio"]', label);
        label.classList.toggle('selected', Boolean(radio && radio.checked));
      });
    };
    input.addEventListener('change', update);
    update();
  });

  // Number steppers enforce integer bounds in the UI; server validation is still authoritative.
  $$('[data-number-step]').forEach(button => {
    button.addEventListener('click', () => {
      const input = button.parentElement.querySelector('[data-number-input]');
      if (!input || input.disabled) return;
      const min = Number(input.min || 0);
      const max = Number(input.max || 999999);
      const current = input.value.trim() === '' ? min : Number(input.value);
      const next = Math.min(max, Math.max(min, current + Number(button.dataset.numberStep)));
      input.value = String(next);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      input.focus();
    });
  });
  $$('[data-number-input]').forEach(input => {
    const inspect = () => {
      const value = input.value.trim();
      const n = Number(value);
      const valid = value !== '' && Number.isInteger(n) && n >= Number(input.min) && n <= Number(input.max);
      input.setAttribute('aria-invalid', String(!valid && value !== ''));
      input.closest('.number-control')?.classList.toggle('is-invalid', !valid && value !== '');
    };
    input.addEventListener('input', inspect);
    input.addEventListener('blur', inspect);
  });

  // Explicitly mark numeric values as undocumented without confusing them with zero.
  $$('[data-toggle-unknown]').forEach(toggle => {
    const fieldName = toggle.dataset.toggleUnknown;
    const input = $('#' + CSS.escape(fieldName));
    const hidden = $(`[data-unknown-hidden="${CSS.escape(fieldName)}"]`);
    const steppers = input?.closest('.number-control') ? $$('[data-number-step]', input.closest('.number-control')) : [];
    if (!input || !hidden) return;
    const sync = () => {
      const unknown = toggle.checked;
      input.disabled = unknown;
      input.required = !unknown;
      hidden.disabled = !unknown;
      input.closest('.number-control')?.classList.toggle('is-unknown', unknown);
      steppers.forEach(btn => { btn.disabled = unknown; });
      if (unknown) input.value = '';
      input.setAttribute('aria-label', unknown ? `${fieldName}, valeur non documentée` : fieldName);
    };
    toggle.addEventListener('change', sync);
    sync();
  });

  // Code lookups are strict: an arbitrary string must not pass as a valid category.
  $$('[data-strict-list]').forEach(input => {
    const inspect = () => {
      const list = $('#' + CSS.escape(input.dataset.strictList));
      if (!list) return;
      const values = new Set($$('option', list).map(option => option.value));
      const value = input.value.trim();
      const valid = value !== '' && values.has(value);
      input.setCustomValidity(valid || value === '' ? '' : 'Sélectionnez un code présent dans la liste proposée.');
      input.setAttribute('aria-invalid', String(value !== '' && !valid));
    };
    input.addEventListener('input', inspect);
    input.addEventListener('change', inspect);
    input.addEventListener('blur', inspect);
    inspect();
  });

  // Friendly inline validation; backend validation remains mandatory and repeats every check.
  $$('form[data-validate-form]').forEach(form => {
    form.addEventListener('submit', event => {
      let firstInvalid = null;
      $$('input, select', form).forEach(input => {
        if (input.type === 'hidden' || input.disabled) return;
        const valid = input.checkValidity();
        input.setAttribute('aria-invalid', String(!valid));
        input.closest('[data-field-block]')?.classList.toggle('field-invalid', !valid);
        if (!valid && !firstInvalid) firstInvalid = input;
      });
      const ref = $('#patient_ref', form);
      if (ref && ref.value.trim() && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$/.test(ref.value.trim())) {
        ref.setCustomValidity('Référence invalide : 1 à 32 caractères alphanumériques, point, tiret ou underscore.');
        if (!firstInvalid) firstInvalid = ref;
      } else if (ref) ref.setCustomValidity('');
      if (firstInvalid) {
        event.preventDefault();
        firstInvalid.reportValidity();
        firstInvalid.focus({ preventScroll: false });
      }
    });
  });
  $$('input, select').forEach(input => input.addEventListener('input', () => {
    if (input.validity.valid) {
      input.setAttribute('aria-invalid', 'false');
      input.closest('[data-field-block]')?.classList.remove('field-invalid');
    }
  }));

  // Password reveal.
  $$('[data-toggle-password]').forEach(button => {
    button.addEventListener('click', () => {
      const input = $(button.dataset.togglePassword);
      if (!input) return;
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      button.textContent = show ? 'Masquer' : 'Voir';
      button.setAttribute('aria-label', show ? 'Masquer le mot de passe' : 'Afficher le mot de passe');
    });
  });

  // Enter only lightweight reveal animations when supported; reduced-motion is respected.
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    $$('.reveal-on-scroll').forEach(el => observer.observe(el));
  }

  // Dismiss stale status banners accessibly.
  $$('.flash-close').forEach(btn => btn.addEventListener('click', () => btn.closest('.flash')?.remove()));
})();
