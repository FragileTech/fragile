/**
 * Expert Mode Toggle for Fragile Docs
 *
 * This script provides a toggle switch that allows readers to switch between:
 * - Full Mode: Shows all content including Feynman-style explanatory prose
 * - Expert Mode: Hides explanatory prose, showing only formal mathematical content
 *
 * The preference is persisted in localStorage across sessions.
 */
(function() {
    'use strict';

    const STORAGE_KEY = 'fragile-expert-mode';

    /**
     * Initialize expert mode from saved preference (runs immediately)
     */
    function initExpertMode() {
        const isExpert = localStorage.getItem(STORAGE_KEY) === 'true';
        if (isExpert) {
            document.documentElement.classList.add('expert-mode');
        }
        return isExpert;
    }

    /**
     * Create the toggle switch component
     */
    function createToggleSwitch(isExpert) {
        // Container
        const container = document.createElement('div');
        container.className = 'expert-mode-container';

        // Label "Full"
        const labelFull = document.createElement('span');
        labelFull.className = 'expert-mode-label expert-mode-label-full';
        labelFull.textContent = 'Full';

        // Switch wrapper
        const switchLabel = document.createElement('label');
        switchLabel.className = 'expert-mode-switch';
        switchLabel.setAttribute('title', 'Toggle between Full Mode (with explanations) and Expert Mode (formal content only)');

        // Hidden checkbox
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = isExpert;
        checkbox.setAttribute('aria-label', 'Toggle expert mode');

        // Slider
        const slider = document.createElement('span');
        slider.className = 'expert-mode-slider';

        switchLabel.appendChild(checkbox);
        switchLabel.appendChild(slider);

        // Label "Expert"
        const labelExpert = document.createElement('span');
        labelExpert.className = 'expert-mode-label expert-mode-label-expert';
        labelExpert.textContent = 'Expert';

        // Assemble
        container.appendChild(labelFull);
        container.appendChild(switchLabel);
        container.appendChild(labelExpert);

        // Update active label styling
        function updateLabels(expert) {
            if (expert) {
                labelFull.classList.remove('active');
                labelExpert.classList.add('active');
            } else {
                labelFull.classList.add('active');
                labelExpert.classList.remove('active');
            }
        }
        updateLabels(isExpert);

        // Event handler
        checkbox.addEventListener('change', function() {
            const nowExpert = checkbox.checked;
            document.documentElement.classList.toggle('expert-mode', nowExpert);
            localStorage.setItem(STORAGE_KEY, nowExpert);
            updateLabels(nowExpert);
            announceChange(nowExpert);
        });

        return container;
    }

    /**
     * Announce mode change for accessibility
     */
    function announceChange(isExpert) {
        const announcement = isExpert
            ? 'Expert mode enabled. Explanatory text is now hidden.'
            : 'Full mode enabled. All content is now visible.';

        let liveRegion = document.getElementById('expert-mode-announce');
        if (!liveRegion) {
            liveRegion = document.createElement('div');
            liveRegion.id = 'expert-mode-announce';
            liveRegion.setAttribute('aria-live', 'polite');
            liveRegion.setAttribute('aria-atomic', 'true');
            liveRegion.style.cssText = 'position:absolute;left:-9999px;';
            document.body.appendChild(liveRegion);
        }
        liveRegion.textContent = announcement;
    }

    /**
     * Insert the toggle switch into the page
     */
    function insertToggle(toggle) {
        // Try to find the primary sidebar toggle button
        const primaryToggle = document.querySelector('.sidebar-toggle.primary-toggle');
        if (primaryToggle) {
            const wrapper = primaryToggle.closest('.header-article-item');
            if (wrapper && wrapper.parentNode) {
                // Insert after the sidebar toggle
                wrapper.parentNode.insertBefore(toggle, wrapper.nextSibling);
                return;
            }
        }

        // Fallback: try header article items area
        const headerItems = document.querySelector('.header-article-items');
        if (headerItems) {
            headerItems.appendChild(toggle);
            return;
        }

        // Fallback: fixed position at top-right
        toggle.classList.add('expert-mode-container-fixed');
        document.body.appendChild(toggle);
    }

    // Initialize immediately (before DOM ready) to prevent flash
    const isExpert = initExpertMode();

    // Add toggle when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            insertToggle(createToggleSwitch(isExpert));
        });
    } else {
        insertToggle(createToggleSwitch(isExpert));
    }
})();
