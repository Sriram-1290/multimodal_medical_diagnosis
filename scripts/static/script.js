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
    
    // Folder Upload elements
    const folderInput = document.getElementById('input-folder');
    const folderBtn = document.getElementById('folder-upload-btn');
    
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

    checkStatus();

    /**
     * Folder Upload Handling
     */
    if (folderBtn && folderInput) {
        // Log setup
        console.log("Folder upload system initialization");

        folderInput.addEventListener('change', (e) => {
            try {
                const files = Array.from(e.target.files);
                if (files.length === 0) return;

                console.log("Total files in folder:", files.length);

                // Filter by extension (more reliable than MIME type on some Windows setups)
                const allowedExts = ['.jpg', '.jpeg', '.png', '.webp'];
                const imageFiles = files.filter(file => {
                    const name = file.name.toLowerCase();
                    return allowedExts.some(ext => name.endsWith(ext));
                }).sort((a, b) => a.name.localeCompare(b.name));

                // Reset state
                Object.keys(zones).forEach(k => {
                    zones[k].file = null;
                    zones[k].preview.innerHTML = '';
                    zones[k].preview.classList.remove('visible');
                    zones[k].zone.classList.remove('active');
                });

                if (imageFiles.length === 0) {
                    statusText.textContent = "Error: No images found";
                    return;
                }

                // Distribution Logic
                let filledCount = 0;
                imageFiles.forEach(file => {
                    const name = file.name.toLowerCase();
                    let targetKey = null;

                    // Prioritize exact view names first
                    if (name.includes('ap.jpg') || name.includes('view_1')) targetKey = 'ap';
                    else if (name.includes('pa.jpg') || name.includes('view_2')) targetKey = 'pa';
                    else if (name.includes('lateral.jpg') || name.includes('view_3')) targetKey = 'lateral';

                    if (targetKey && !zones[targetKey].file) {
                        zones[targetKey].file = file;
                        renderPreview(targetKey, file);
                        filledCount++;
                    }
                });

                // Fill remaining empty slots sequentially
                const keys = ['ap', 'pa', 'lateral'];
                imageFiles.forEach(file => {
                    const emptyKey = keys.find(k => !zones[k].file);
                    const alreadyUsed = Object.values(zones).some(z => z.file === file);
                    if (emptyKey && !alreadyUsed) {
                        zones[emptyKey].file = file;
                        renderPreview(emptyKey, file);
                        filledCount++;
                    }
                });

                function renderPreview(key, file) {
                    const reader = new FileReader();
                    reader.onload = (re) => {
                        zones[key].preview.innerHTML = `<img src="${re.target.result}" alt="Preview">`;
                        zones[key].preview.classList.add('visible');
                        zones[key].zone.classList.add('active');
                    };
                    reader.readAsDataURL(file);
                }

                statusText.textContent = `Success: ${filledCount} views loaded`;
                setTimeout(updateGenerateButtonState, 100);
            } catch (err) {
                console.error("Folder upload failed:", err);
                statusText.textContent = "Upload Error";
            }
        });
    }

    /**
     * Handle File Upload Zones
     */
    Object.keys(zones).forEach(key => {
        const { input, zone, preview } = zones[key];
        
        // Trigger file input on click
        zone.addEventListener('click', () => input.click());

        // Handle file selection (Now supports multi-selection/distribution)
        input.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            if (files.length === 0) return;

            if (files.length === 1) {
                // Standard single file behavior
                const file = files[0];
                zones[key].file = file;
                const reader = new FileReader();
                reader.onload = (re) => {
                    preview.innerHTML = `<img src="${re.target.result}" alt="Preview">`;
                    preview.classList.add('visible');
                    zone.classList.add('active');
                };
                reader.readAsDataURL(file);
            } else {
                // Multi-selection behavior: Distribute files to empty slots
                console.log(`Distributing ${files.length} files from ${key} slot`);
                const slotKeys = ['ap', 'pa', 'lateral'];
                let fileIdx = 0;
                
                slotKeys.forEach(sKey => {
                    if (fileIdx < files.length) {
                        const file = files[fileIdx++];
                        zones[sKey].file = file;
                        const reader = new FileReader();
                        reader.onload = (re) => {
                            zones[sKey].preview.innerHTML = `<img src="${re.target.result}" alt="Preview">`;
                            zones[sKey].preview.classList.add('visible');
                            zones[sKey].zone.classList.add('active');
                        };
                        reader.readAsDataURL(file);
                    }
                });
            }
            updateGenerateButtonState();
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
                
                // Robust parsing for Findings/Impression (case-insensitive)
                const report = data.report;
                const impIndex = report.toLowerCase().indexOf('impression:');
                
                if (impIndex !== -1) {
                    const findings = report.substring(0, impIndex).replace(/findings:/i, '').trim();
                    const impression = report.substring(impIndex + 11).trim();
                    typeWriterEffect(findingsText, findings || "No significant findings documented.");
                    impressionText.textContent = impression || "Unremarkable.";
                } else {
                    typeWriterEffect(findingsText, report.replace(/findings:/i, '').trim());
                    impressionText.textContent = "See findings above.";
                }
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
