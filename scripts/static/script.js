document.addEventListener('DOMContentLoaded', () => {
    const apiBase = window.location.origin;
    const statusBadge = document.getElementById('status-badge');
    const statusText = statusBadge.querySelector('.status-text');
    const generateBtn = document.getElementById('generate-btn');
    
    // UI elements for states
    const welcomeState = document.getElementById('welcome-state');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    
    // Report text elements
    const findingsText = document.getElementById('findings-text');
    const impressionText = document.getElementById('impression-text');
    const displayId = document.getElementById('display-id');
    
    // File inputs and previews
    const zones = {
        ap: { input: document.getElementById('input-ap'), zone: document.getElementById('zone-ap'), preview: document.getElementById('preview-ap'), file: null },
        pa: { input: document.getElementById('input-pa'), zone: document.getElementById('zone-pa'), preview: document.getElementById('preview-pa'), file: null },
        lateral: { input: document.getElementById('input-lateral'), zone: document.getElementById('zone-lateral'), preview: document.getElementById('preview-lateral'), file: null }
    };

    /**
     * System Initialization: Check API status
     */
    async function checkStatus() {
        try {
            const response = await fetch(`${apiBase}/api/status`);
            const data = await response.json();
            
            if (data.status === 'ready') {
                statusBadge.className = 'status-badge ready';
                statusText.textContent = 'System Ready';
                updateGenerateButtonState();
            } else if (data.status === 'model_not_found') {
                statusBadge.className = 'status-badge error';
                statusText.textContent = 'Checkpoint Missing';
                showError("Model weights (best_model.pth) not found. Please train the model first.");
            }
        } catch (e) {
            statusBadge.className = 'status-badge error';
            statusText.textContent = 'Server Offline';
            showError("Cannot connect to backend server.");
        }
    }

    /**
     * Handle File Upload Zones
     */
    Object.keys(zones).forEach(key => {
        const { input, zone, preview } = zones[key];
        
        // Trigger file input on click
        zone.addEventListener('click', () => input.click());

        // Handle file selection
        input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const file = e.target.files[0];
                zones[key].file = file;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (re) => {
                    preview.innerHTML = `<img src="${re.target.result}" alt="Preview">`;
                    preview.classList.add('visible');
                    zone.classList.add('active');
                };
                reader.readAsDataURL(file);
                updateGenerateButtonState();
            }
        });

        // Drag and Drop
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('hover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('hover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('hover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                const file = e.dataTransfer.files[0];
                input.files = e.dataTransfer.files;
                // Dispatch event manually to trigger preview
                input.dispatchEvent(new Event('change'));
            }
        });
    });

    /**
     * Prediction Trigger
     */
    generateBtn.addEventListener('click', async () => {
        if (generateBtn.classList.contains('disabled')) return;

        // UI State Switch: Loading
        showState(loadingState);
        generateBtn.classList.add('loading');
        
        const formData = new FormData();
        if (zones.ap.file) formData.append('ap_view', zones.ap.file);
        if (zones.pa.file) formData.append('pa_view', zones.pa.file);
        if (zones.lateral.file) formData.append('lateral_view', zones.lateral.file);

        try {
            const response = await fetch(`${apiBase}/api/predict`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayId.textContent = `REF-${Math.floor(Math.random() * 90000) + 10000}`;
                showState(resultState);
                typeWriterEffect(findingsText, data.report.split('Impression:')[0].replace('Findings: ', '').trim());
                impressionText.textContent = data.report.split('Impression:')[1]?.trim() || "No acute intrathoracic abnormality.";
            } else {
                showError(data.detail || "Error generating report.");
            }
        } catch (e) {
            showError("Network error. Please check your connection.");
        } finally {
            generateBtn.classList.remove('loading');
        }
    });

    /**
     * Helpers
     */
    function updateGenerateButtonState() {
        const hasFile = zones.ap.file || zones.pa.file || zones.lateral.file;
        const isReady = statusBadge.classList.contains('ready');
        
        if (hasFile && isReady) {
            generateBtn.classList.remove('disabled');
            generateBtn.disabled = false;
        } else {
            generateBtn.classList.add('disabled');
            generateBtn.disabled = true;
        }
    }

    function showState(stateElement) {
        [welcomeState, loadingState, resultState, errorState].forEach(st => st.classList.add('hidden'));
        stateElement.classList.remove('hidden');
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        showState(errorState);
    }

    function typeWriterEffect(element, text) {
        element.textContent = '';
        let i = 0;
        const speed = 15; // ms per char
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    // Copy to clipboard
    document.getElementById('copy-btn').addEventListener('click', () => {
        const report = `Findings: ${findingsText.textContent}\nImpression: ${impressionText.textContent}`;
        navigator.clipboard.writeText(report).then(() => {
            const icon = document.querySelector('#copy-btn i');
            icon.className = 'fas fa-check';
            setTimeout(() => icon.className = 'far fa-copy', 2000);
        });
    });

    // Run status check
    checkStatus();
});
