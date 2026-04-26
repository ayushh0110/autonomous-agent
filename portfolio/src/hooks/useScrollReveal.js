import { useEffect, useRef } from 'react';

export function useScrollReveal() {
  const ref = useRef(null);

  useEffect(() => {
    const triggerVisible = (container) => {
      const els = container.querySelectorAll('.visible');
      els.forEach((el, i) => {
        const delay = el.style.animationDelay || `${i * 0.08}s`;
        setTimeout(() => {
          el.classList.add('in-view');
        }, parseFloat(delay) * 1000);
      });
    };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            triggerVisible(entry.target);
          }
        });
      },
      { threshold: 0.05, rootMargin: '0px 0px -40px 0px' }
    );

    if (ref.current) {
      observer.observe(ref.current);
      // Trigger immediately if already in viewport
      const rect = ref.current.getBoundingClientRect();
      if (rect.top < window.innerHeight) {
        triggerVisible(ref.current);
      }
    }
    return () => observer.disconnect();
  }, []);

  return ref;
}
