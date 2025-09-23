// Main JavaScript for Start Your Quant

document.addEventListener('DOMContentLoaded', function() {
  // Scroll to Top Button
  const scrollToTopBtn = document.getElementById('scrollToTop');

  if (scrollToTopBtn) {
    window.addEventListener('scroll', function() {
      if (window.pageYOffset > 300) {
        scrollToTopBtn.classList.add('visible');
      } else {
        scrollToTopBtn.classList.remove('visible');
      }
    });

    scrollToTopBtn.addEventListener('click', function() {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add active class to current navigation item
  const currentLocation = location.pathname;
  const menuItems = document.querySelectorAll('.nav-item');

  menuItems.forEach(item => {
    if (item.getAttribute('href') === currentLocation) {
      item.classList.add('active');
    }
  });

  // Progress indicator for long articles
  const progressBar = document.createElement('div');
  progressBar.className = 'reading-progress';
  progressBar.style.cssText = `
    position: fixed;
    top: 60px;
    left: 0;
    width: 0%;
    height: 3px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    z-index: 999;
    transition: width 0.2s ease;
  `;
  document.body.appendChild(progressBar);

  window.addEventListener('scroll', function() {
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrolled = (window.scrollY / docHeight) * 100;
    progressBar.style.width = scrolled + '%';
  });

  // Copy code button for code blocks
  const codeBlocks = document.querySelectorAll('pre');

  codeBlocks.forEach(block => {
    const button = document.createElement('button');
    button.className = 'copy-code-btn';
    button.textContent = '📋 Copiar';
    button.style.cssText = `
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.2);
      padding: 0.25rem 0.75rem;
      border-radius: 0.25rem;
      cursor: pointer;
      font-size: 0.875rem;
      transition: all 0.3s ease;
    `;

    block.style.position = 'relative';
    block.appendChild(button);

    button.addEventListener('click', function() {
      const code = block.querySelector('code') || block;
      const text = code.textContent;

      navigator.clipboard.writeText(text).then(() => {
        button.textContent = '✅ Copiado!';
        button.style.background = 'rgba(40, 167, 69, 0.2)';
        button.style.borderColor = '#28a745';

        setTimeout(() => {
          button.textContent = '📋 Copiar';
          button.style.background = 'rgba(255, 255, 255, 0.1)';
          button.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        }, 2000);
      });
    });

    button.addEventListener('mouseenter', function() {
      button.style.background = 'rgba(255, 255, 255, 0.2)';
    });

    button.addEventListener('mouseleave', function() {
      button.style.background = 'rgba(255, 255, 255, 0.1)';
    });
  });

  // Add external link indicators
  const externalLinks = document.querySelectorAll('a[href^="http"]:not([href*="jefrnc.github.io"])');

  externalLinks.forEach(link => {
    link.classList.add('external-link');
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');

    const icon = document.createElement('span');
    icon.textContent = ' ↗';
    icon.style.fontSize = '0.75em';
    link.appendChild(icon);
  });

  // Lazy loading for images
  const images = document.querySelectorAll('img');

  const imageOptions = {
    threshold: 0,
    rootMargin: '0px 0px 50px 0px'
  };

  const imageObserver = new IntersectionObserver(function(entries, observer) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        if (img.dataset.src) {
          img.src = img.dataset.src;
          img.classList.add('fade-in');
          observer.unobserve(img);
        }
      }
    });
  }, imageOptions);

  images.forEach(img => {
    imageObserver.observe(img);
  });

  // Add keyboard navigation
  document.addEventListener('keydown', function(e) {
    // Press '/' to focus search (when implemented)
    if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
      const searchInput = document.querySelector('.search-input');
      if (searchInput && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
      }
    }

    // Press 'g' then 'h' to go home
    if (e.key === 'g') {
      window.addEventListener('keydown', function goHome(e2) {
        if (e2.key === 'h') {
          window.location.href = '/start-your-quant/';
        }
        window.removeEventListener('keydown', goHome);
      });
    }
  });

  // Print friendly version
  window.addEventListener('beforeprint', function() {
    document.body.classList.add('print-mode');
  });

  window.addEventListener('afterprint', function() {
    document.body.classList.remove('print-mode');
  });
});

// Performance monitoring
if ('performance' in window && 'PerformanceObserver' in window) {
  const perfObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.entryType === 'largest-contentful-paint') {
        console.log('LCP:', entry.startTime);
      }
    }
  });

  perfObserver.observe({ entryTypes: ['largest-contentful-paint'] });
}