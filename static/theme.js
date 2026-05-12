/**
 * SIREKPRODI Premium Theme Interactions
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- 1. Sliding Navigation Indicator ---
  const navShell = document.querySelector('.nav-shell');
  const navLinksWrap = navShell?.querySelector('.nav-links');
  const navLinks = navShell?.querySelectorAll('[data-nav-link]') || [];
  const navIndicator = navShell?.querySelector('#nav-indicator');

  if (navShell && navLinksWrap && navIndicator && navLinks.length > 0) {
    let navActive = navShell.querySelector('.nav-link.is-active') || navLinks[0];

    function moveIndicator(link) {
      if (!link) return;
      const wrapRect = navLinksWrap.getBoundingClientRect();
      const linkRect = link.getBoundingClientRect();
      navIndicator.style.width = `${linkRect.width}px`;
      navIndicator.style.transform = `translateX(${linkRect.left - wrapRect.left}px)`;
    }

    function setActive(link) {
      navActive = link;
      navLinks.forEach(l => l.classList.toggle('is-active', l === link));
      moveIndicator(link);
    }

    navLinks.forEach(link => {
      link.addEventListener('mouseenter', () => moveIndicator(link));
      link.addEventListener('mouseleave', () => moveIndicator(navActive));
      link.addEventListener('click', () => setActive(link));
    });

    window.addEventListener('resize', () => moveIndicator(navActive));
    // Initial position
    setTimeout(() => moveIndicator(navActive), 100); // Wait for fonts/layout to settle
  }

  // --- 2. Toast Notifications ---
  const toastContainer = document.createElement('div');
  toastContainer.className = 'toast-container';
  document.body.appendChild(toastContainer);

  window.showToast = function (type, message) {
    const el = document.createElement('div');
    const typeClass = type === 'error' ? 'toast-error' : type === 'warning' ? 'toast-warning' : type === 'info' ? 'toast-info' : 'toast-success';
    el.className = `toast ${typeClass}`;

    // Icon mapping
    let iconSvg = '';
    if (type === 'success') {
      iconSvg = '<svg class="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
    } else if (type === 'error') {
      iconSvg = '<svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>';
    } else if (type === 'warning') {
      iconSvg = '<svg class="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>';
    } else {
      iconSvg = '<svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
    }

    el.innerHTML = `${iconSvg} <span>${message}</span>`;
    toastContainer.appendChild(el);

    // Auto remove after 3.5 seconds
    setTimeout(() => {
      el.classList.add('hiding');
      el.addEventListener('animationend', () => el.remove());
    }, 3500);
  };

  // Check for global flash messages injected by Flask
  if (window.__FLASH_MESSAGES__ && Array.isArray(window.__FLASH_MESSAGES__)) {
    window.__FLASH_MESSAGES__.forEach(([cat, msg], index) => {
      // Stagger toasts slightly
      setTimeout(() => window.showToast(cat, msg), index * 300);
    });
  }

  // --- 3. Generic Confirm Modal ---
  const confirmModal = document.getElementById('confirm-modal');
  if (confirmModal) {
    const confirmTitle = document.getElementById('confirm-title');
    const confirmMessage = document.getElementById('confirm-message');
    const confirmAction = document.getElementById('confirm-action');
    const confirmCancel = document.getElementById('confirm-cancel');
    const confirmIcon = document.getElementById('confirm-icon');

    const accentMap = {
      red: { bg: 'bg-red-50', text: 'text-red-600', btn: 'bg-red-600 hover:bg-red-700' },
      amber: { bg: 'bg-amber-50', text: 'text-amber-600', btn: 'bg-amber-500 hover:bg-amber-600' },
      green: { bg: 'bg-emerald-50', text: 'text-emerald-600', btn: 'bg-emerald-600 hover:bg-emerald-700' },
      blue: { bg: 'bg-blue-50', text: 'text-blue-600', btn: 'bg-blue-600 hover:bg-blue-700' }
    };

    window.showConfirm = function (url, title, message, accent = 'red') {
      const tone = accentMap[accent] || accentMap.red;
      if (confirmTitle) confirmTitle.textContent = title;
      if (confirmMessage) confirmMessage.textContent = message;
      if (confirmAction) {
        confirmAction.href = url;
        confirmAction.className = `px-5 py-2.5 rounded-xl text-white shadow-sm font-semibold transition-all ${tone.btn}`;
      }
      if (confirmIcon) {
        confirmIcon.className = `h-12 w-12 rounded-2xl flex items-center justify-center text-2xl ${tone.bg} ${tone.text}`;
      }

      confirmModal.classList.remove('hidden');
      confirmModal.classList.add('flex');
      // Subtle entrance animation
      const modalContent = confirmModal.querySelector('div');
      if (modalContent) {
        modalContent.style.opacity = '0';
        modalContent.style.transform = 'scale(0.95) translateY(10px)';
        requestAnimationFrame(() => {
          modalContent.style.transition = 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
          modalContent.style.opacity = '1';
          modalContent.style.transform = 'scale(1) translateY(0)';
        });
      }
    };

    window.hideConfirm = function () {
      const modalContent = confirmModal.querySelector('div');
      if (modalContent) {
        modalContent.style.transform = 'scale(0.95) translateY(10px)';
        modalContent.style.opacity = '0';
        setTimeout(() => {
          confirmModal.classList.add('hidden');
          confirmModal.classList.remove('flex');
        }, 200);
      } else {
        confirmModal.classList.add('hidden');
        confirmModal.classList.remove('flex');
      }
    };

    if (confirmCancel) confirmCancel.addEventListener('click', window.hideConfirm);
    confirmModal.addEventListener('click', (e) => {
      if (e.target === confirmModal) window.hideConfirm();
    });

    document.querySelectorAll('[data-confirm]').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        window.showConfirm(
          link.href,
          link.dataset.confirmTitle || 'Konfirmasi',
          link.dataset.confirmMessage || 'Apakah Anda yakin ingin melanjutkan?',
          link.dataset.confirmAccent || 'red'
        );
      });
    });
  }

  // --- 4. Chart.js Global Polish ---
  if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = "'Plus Jakarta Sans', system-ui, sans-serif";
    Chart.defaults.color = '#64748b'; // slate-500

    if (Chart.defaults.plugins.tooltip) {
      Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.95)'; // slate-900 with opacity
      Chart.defaults.plugins.tooltip.padding = 12;
      Chart.defaults.plugins.tooltip.cornerRadius = 8;
      Chart.defaults.plugins.tooltip.titleFont = { size: 13, weight: 'bold', family: "'Plus Jakarta Sans', system-ui, sans-serif" };
      Chart.defaults.plugins.tooltip.bodyFont = { size: 12, family: "'Plus Jakarta Sans', system-ui, sans-serif" };
      Chart.defaults.plugins.tooltip.boxPadding = 6;
      Chart.defaults.plugins.tooltip.usePointStyle = true;
    }

    // Default scales if they exist
    if (Chart.defaults.scale) {
      if (Chart.defaults.scale.grid) {
        Chart.defaults.scale.grid.color = 'rgba(15, 23, 42, 0.04)'; // very subtle slate border
        Chart.defaults.scale.grid.borderColor = 'transparent';
      }
    }
  }

  // --- 5. Profile Dropdown Logic ---
  const profileBtn = document.getElementById('profileDropdownBtn');
  const profileMenu = document.getElementById('profileDropdownMenu');

  if (profileBtn && profileMenu) {
    profileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      profileMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', () => {
      profileMenu.classList.add('hidden');
    });

    profileMenu.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  }

  // --- 6. Number Counter Animation for KPIs ---
  const kpiValues = document.querySelectorAll('.kpi-card .value');
  
  if (kpiValues.length > 0) {
    const animateValue = (el, start, end, duration) => {
      let startTimestamp = null;
      const isPercent = el.textContent.includes('%');
      
      const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        // Easing function (easeOutExpo)
        const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        
        const current = Math.floor(ease * (end - start) + start);
        
        // Keep % sign if it was there
        if (isPercent) {
           el.textContent = (ease * (end - start) + start).toFixed(1) + '%';
        } else {
           el.textContent = current.toLocaleString('id-ID');
        }
        
        if (progress < 1) {
          window.requestAnimationFrame(step);
        } else {
           if (isPercent) {
             el.textContent = end.toFixed(1) + '%';
           } else {
             el.textContent = end.toLocaleString('id-ID');
           }
        }
      };
      window.requestAnimationFrame(step);
    };

    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px"
    };

    const observer = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target;
          let textValue = el.textContent.replace(/[^0-9.]/g, '');
          if (textValue) {
            const endValue = parseFloat(textValue);
            if (!isNaN(endValue)) {
              el.style.opacity = '1';
              animateValue(el, 0, endValue, 1500);
            }
          }
          observer.unobserve(el);
        }
      });
    }, observerOptions);

    kpiValues.forEach(el => {
      observer.observe(el);
    });
  }
});
