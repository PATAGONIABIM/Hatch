(() => {
    'use strict';

    /* ============ i18n ============ */
    const I18N = {
        es: {
            'hero.eyebrow': 'PATAGONIABIM / HERRAMIENTAS',
            'hero.subtitle': 'Patrones de relleno para Revit, generados desde imágenes y DXF.',
            'mode.image': 'IMAGEN',
            'mode.dxf': 'DXF',
            'drop.title': 'Arrastra tu archivo aquí',
            'drop.sub': 'PNG, JPG o DXF — máx. 12 MB',
            'drop.select': 'SELECCIONAR ARCHIVO',
            'ctl.detection': 'DETECCIÓN',
            'ctl.preprocess': 'PRE-PROCESAMIENTO',
            'ctl.scale': 'ESCALA Y GEOMETRÍA',
            'ctl.method': 'ALGORITMO',
            'ctl.minlen': 'LONGITUD MÍN. DE LÍNEA',
            'ctl.gap': 'UNIR HUECOS (GAP)',
            'ctl.eps': 'SUAVIZADO DE CURVAS',
            'ctl.clahe': 'CLAHE — mejorar contraste',
            'ctl.claheclip': 'INTENSIDAD CLAHE',
            'ctl.skeleton': 'Skeleton — adelgazar trazos',
            'ctl.merge': 'Unir segmentos colineales',
            'ctl.blackhat': 'Black-hat — trazos oscuros finos',
            'ctl.blackhatk': 'TAMAÑO KERNEL',
            'ctl.adaptive': 'Adaptive Threshold — en vez de Canny',
            'ctl.adapblock': 'BLOQUE ADAPTIVO',
            'ctl.adapc': 'CONSTANTE C',
            'ctl.autocanny': 'Auto-Canny — umbrales automáticos',
            'ctl.cannylow': 'CANNY LOW',
            'ctl.cannyhigh': 'CANNY HIGH',
            'ctl.blur': 'DESENFOQUE — LIMPIAR RUIDO',
            'ctl.filter': 'FILTRO DE RUIDO',
            'ctl.tilemm': 'ESCALA REAL DEL TILE (MM)',
            'ctl.maxres': 'RESOLUCIÓN DE TRABAJO (PX)',
            'ctl.offx': 'OFFSET X',
            'ctl.offy': 'OFFSET Y',
            'method.hough': 'Líneas rectas — Hough',
            'method.lsd': 'Líneas precisas — LSD',
            'method.fld': 'FastLineDetector',
            'method.contour': 'Formas orgánicas — Contornos',
            'filter.gaussian': 'Gaussiano',
            'filter.bilateral': 'Bilateral — preserva bordes',
            'filter.none': 'Ninguno',
            'btn.convert': 'GENERAR PATRÓN',
            'btn.converting': 'PROCESANDO...',
            'btn.updating': 'ACTUALIZANDO...',
            'res.live': 'EN VIVO',
            'res.preview': 'SIMULACIÓN REVIT',
            'res.tile3': '3x3 TILES',
            'res.scale': 'ESCALA',
            'res.compare': 'COMPARATIVA',
            'res.detected': 'Líneas detectadas',
            'res.code': 'CÓDIGO .PAT',
            'res.copy': 'COPIAR',
            'res.copied': 'COPIADO',
            'res.download': 'DESCARGAR .PAT',
            'footer.rights': 'Todos los derechos reservados.',
            'footer.design': 'Diseño WEB',
            'footer.revit': 'REVIT: MANAGE → ADDITIONAL SETTINGS → FILL PATTERNS → IMPORT',
            'err.upload': 'Sube un archivo para continuar',
            'err.network': 'No se pudo conectar con el servidor. Intenta de nuevo.',
        },
        en: {
            'hero.eyebrow': 'PATAGONIABIM / TOOLS',
            'hero.subtitle': 'Revit fill patterns, generated from images and DXF.',
            'mode.image': 'IMAGE',
            'mode.dxf': 'DXF',
            'drop.title': 'Drop your file here',
            'drop.sub': 'PNG, JPG or DXF — max 12 MB',
            'drop.select': 'SELECT FILE',
            'ctl.detection': 'DETECTION',
            'ctl.preprocess': 'PRE-PROCESSING',
            'ctl.scale': 'SCALE & GEOMETRY',
            'ctl.method': 'ALGORITHM',
            'ctl.minlen': 'MIN. LINE LENGTH',
            'ctl.gap': 'JOIN GAPS',
            'ctl.eps': 'CURVE SMOOTHING',
            'ctl.clahe': 'CLAHE — enhance contrast',
            'ctl.claheclip': 'CLAHE INTENSITY',
            'ctl.skeleton': 'Skeleton — thin strokes',
            'ctl.merge': 'Merge collinear segments',
            'ctl.blackhat': 'Black-hat — thin dark strokes',
            'ctl.blackhatk': 'KERNEL SIZE',
            'ctl.adaptive': 'Adaptive Threshold — instead of Canny',
            'ctl.adapblock': 'ADAPTIVE BLOCK',
            'ctl.adapc': 'CONSTANT C',
            'ctl.autocanny': 'Auto-Canny — automatic thresholds',
            'ctl.cannylow': 'CANNY LOW',
            'ctl.cannyhigh': 'CANNY HIGH',
            'ctl.blur': 'BLUR — REDUCE NOISE',
            'ctl.filter': 'NOISE FILTER',
            'ctl.tilemm': 'REAL TILE SIZE (MM)',
            'ctl.maxres': 'WORKING RESOLUTION (PX)',
            'ctl.offx': 'OFFSET X',
            'ctl.offy': 'OFFSET Y',
            'method.hough': 'Straight lines — Hough',
            'method.lsd': 'Precise lines — LSD',
            'method.fld': 'FastLineDetector',
            'method.contour': 'Organic shapes — Contours',
            'filter.gaussian': 'Gaussian',
            'filter.bilateral': 'Bilateral — preserves edges',
            'filter.none': 'None',
            'btn.convert': 'GENERATE PATTERN',
            'btn.converting': 'PROCESSING...',
            'btn.updating': 'UPDATING...',
            'res.live': 'LIVE',
            'res.preview': 'REVIT SIMULATION',
            'res.tile3': '3x3 TILES',
            'res.scale': 'SCALE',
            'res.compare': 'COMPARISON',
            'res.detected': 'Detected lines',
            'res.code': '.PAT CODE',
            'res.copy': 'COPY',
            'res.copied': 'COPIED',
            'res.download': 'DOWNLOAD .PAT',
            'footer.rights': 'All rights reserved.',
            'footer.design': 'Web Design',
            'footer.revit': 'REVIT: MANAGE → ADDITIONAL SETTINGS → FILL PATTERNS → IMPORT',
            'err.upload': 'Upload a file to continue',
            'err.network': 'Could not reach the server. Try again.',
        },
    };

    const LANG_KEY = 'lang';
    let lang = localStorage.getItem(LANG_KEY) || 'es';
    if (!I18N[lang]) lang = 'es';

    function t(key) {
        return (I18N[lang] && I18N[lang][key]) || key;
    }

    function applyI18n() {
        document.documentElement.lang = lang;
        document.querySelectorAll('[data-i18n]').forEach((el) => {
            el.textContent = t(el.dataset.i18n);
        });
    }

    /* ============ Helpers ============ */
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function setFill(input) {
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        const val = parseFloat(input.value);
        input.style.setProperty('--fill', ((val - min) / (max - min)) * 100 + '%');
    }

    function formatSliderOutput(id, value) {
        if (id === 'ctl-offx' || id === 'ctl-offy') {
            return (parseFloat(value) / 100).toFixed(2);
        }
        if (id === 'ctl-p2cont') {
            return (parseFloat(value) / 1000).toFixed(3);
        }
        if (id === 'ctl-claheclip') {
            return parseFloat(value).toFixed(1);
        }
        if (id === 'scale-slider') {
            return (parseFloat(value) / 10).toFixed(1) + 'x';
        }
        return value;
    }

    $$('input[type="range"]').forEach((input) => {
        const out = document.querySelector(`output[for="${input.id}"]`);
        setFill(input);
        if (out) out.value = formatSliderOutput(input.id, input.value);
        input.addEventListener('input', () => {
            setFill(input);
            if (out) out.value = formatSliderOutput(input.id, input.value);
        });
    });

    /* ============ Smooth scroll (Lenis, igual que patagoniabim.cl) ============ */
    if (window.Lenis) {
        const lenis = new Lenis({
            duration: 1.0,
            easing: (t) => 1 - Math.pow(1 - t, 3),
            smoothWheel: true,
            touchMultiplier: 1.5,
        });
        function raf(time) {
            lenis.raf(time);
            requestAnimationFrame(raf);
        }
        requestAnimationFrame(raf);
    }

    /* ============ Page init ============ */
    document.body.classList.add('is-loaded');

    /* ============ Grid canvas background ============ */
    const canvas = $('#grid-canvas');
    const ctx = canvas.getContext('2d');
    let gridCells = 96;
    let mouse = { x: 0, y: 0 };

    function sizeCanvas() {
        const dpr = Math.min(devicePixelRatio || 1, 2);
        canvas.width = innerWidth * dpr;
        canvas.height = innerHeight * dpr;
        canvas.style.width = innerWidth + 'px';
        canvas.style.height = innerHeight + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        gridCells = innerWidth < 640 ? 72 : 96;
    }

    function drawGrid() {
        const w = innerWidth, h = innerHeight;
        const off = reducedMotion ? 0 : (mouse.x - w / 2) * 0.02;
        ctx.clearRect(0, 0, w, h);

        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(255,255,255,0.045)';
        ctx.beginPath();
        for (let x = gridCells; x < w; x += gridCells) {
            ctx.moveTo(x + off, 0);
            ctx.lineTo(x + off, h);
        }
        for (let y = gridCells; y < h; y += gridCells) {
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
        }
        ctx.stroke();

        ctx.fillStyle = 'rgba(255,215,0,0.35)';
        for (let x = gridCells; x < w; x += gridCells) {
            for (let y = gridCells; y < h; y += gridCells) {
                ctx.beginPath();
                ctx.arc(x + off, y, 1.2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        if (!reducedMotion) {
            ctx.strokeStyle = 'rgba(255,215,0,0.05)';
            ctx.beginPath();
            ctx.moveTo(w / 2 + off, 0);
            ctx.lineTo(w / 2 + off, h);
            ctx.stroke();
        }
    }

    sizeCanvas();
    drawGrid();
    addEventListener('resize', () => { sizeCanvas(); drawGrid(); });
    if (!reducedMotion) {
        let gridFrame = null;
        document.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
            if (!gridFrame) {
                gridFrame = requestAnimationFrame(() => {
                    drawGrid();
                    gridFrame = null;
                });
            }
        });
    }

    /* ============ Lang switch ============ */
    $$('.lang-btn').forEach((btn) => {
        btn.classList.toggle('is-active', btn.dataset.lang === lang);
        btn.addEventListener('click', () => {
            lang = btn.dataset.lang;
            localStorage.setItem(LANG_KEY, lang);
            $$('.lang-btn').forEach((b) => b.classList.toggle('is-active', b.dataset.lang === lang));
            applyI18n();
        });
    });
    applyI18n();

    /* ============ Mode switch ============ */
    let mode = 'image';
    const controls = $('#controls');
    const dropTitle = $('.dropzone-title');
    const dropSub = $('.dropzone-sub');

    $$('.mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            mode = btn.dataset.mode;
            $$('.mode-btn').forEach((b) => b.classList.toggle('is-active', b === btn));
            controls.classList.toggle('hidden', mode === 'dxf');
            dropSub.textContent = mode === 'dxf'
                ? (lang === 'es' ? 'Archivo DXF — máx. 12 MB' : 'DXF file — max 12 MB')
                : t('drop.sub');
            fileInput.accept = mode === 'dxf' ? '.dxf' : '.png,.jpg,.jpeg';
            clearFile();
            hideResults();
        });
    });

    /* ============ Dropzone ============ */
    const dz = $('#dropzone');
    const fileInput = $('#file-input');
    let currentFile = null;

    fileInput.accept = '.png,.jpg,.jpeg';

    dz.addEventListener('click', () => fileInput.click());
    dz.addEventListener('dragover', (e) => {
        e.preventDefault();
        dz.classList.add('is-dragover');
    });
    dz.addEventListener('dragleave', () => dz.classList.remove('is-dragover'));
    dz.addEventListener('drop', (e) => {
        e.preventDefault();
        dz.classList.remove('is-dragover');
        if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) setFile(fileInput.files[0]);
    });
    $('.file-clear').addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    function setFile(file) {
        const okTypes = mode === 'dxf'
            ? ['.dxf']
            : ['.png', '.jpg', '.jpeg'];
        const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
        if (!okTypes.includes(ext)) {
            showError(lang === 'es'
                ? 'Formato no válido. Usa ' + okTypes.join(', ')
                : 'Invalid format. Use ' + okTypes.join(', '));
            return;
        }
        if (file.size > 12 * 1024 * 1024) {
            showError(lang === 'es' ? 'El archivo supera 12 MB' : 'File exceeds 12 MB');
            return;
        }
        currentFile = file;
        $('.dropzone-file').classList.remove('hidden');
        $('.file-name').textContent = file.name;
        $('.file-size').textContent = formatBytes(file.size);
        const thumb = $('.file-thumb');
        if (mode === 'image' && file.type.startsWith('image/')) {
            thumb.classList.remove('hidden');
            thumb.src = URL.createObjectURL(file);
        } else {
            thumb.classList.add('hidden');
            thumb.removeAttribute('src');
        }
        dz.classList.add('has-file');
        // Auto-generación inmediata al cargar archivo
        convert({ initial: true, shouldScroll: true });
    }

    function clearFile() {
        if (debounceTimer) {
            clearTimeout(debounceTimer);
            debounceTimer = null;
        }
        if (activeAbortController) {
            activeAbortController.abort();
            activeAbortController = null;
        }
        currentFile = null;
        fileInput.value = '';
        dz.classList.remove('has-file');
        $('.dropzone-file').classList.add('hidden');
        hideResults();
        hideAlert();
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    /* ============ Collapsible groups ============ */
    $$('.group-header').forEach((header) => {
        header.addEventListener('click', () => {
            const open = header.getAttribute('aria-expanded') === 'true';
            header.setAttribute('aria-expanded', String(!open));
        });
    });

    /* ============ Toggle dependencies ============ */
    function syncDependencies() {
        $$('[data-depends]').forEach((el) => {
            const dep = $('#' + el.dataset.depends);
            el.classList.toggle('hidden', !(dep && dep.checked));
        });
        $$('[data-depends-not]').forEach((el) => {
            const dep = $('#' + el.dataset.dependsNot);
            el.classList.toggle('hidden', !!(dep && dep.checked));
        });
    }
    $$('input[type="checkbox"]').forEach((cb) => {
        cb.addEventListener('change', syncDependencies);
    });
    syncDependencies();

    /* ============ Method-dependent params ============ */
    const methodSel = $('#ctl-method');
    const p1Label = $('[data-for-method="all"] .field-label');
    function syncMethod() {
        const m = methodSel.value;
        $$('[data-for-method]').forEach((el) => {
            el.classList.toggle('hidden', el.dataset.forMethod !== 'all' && el.dataset.forMethod !== m);
        });
        const defaults = { hough: 20, lsd: 15, fld: 20, contour: 20 };
        const p1 = $('#ctl-p1');
        p1.value = defaults[m];
        const outP1 = $('output[for="ctl-p1"]');
        if (outP1) outP1.value = formatSliderOutput('ctl-p1', p1.value);
        setFill(p1);
        p1Label.textContent = m === 'contour'
            ? (lang === 'es' ? 'LONGITUD MÍN. CONTORNO' : 'MIN. CONTOUR LENGTH')
            : t('ctl.minlen');
        if (currentFile && mode === 'image') {
            scheduleConvert(0);
        }
    }
    methodSel.addEventListener('change', syncMethod);
    syncMethod();

    /* ============ Live reactive updates & Convert ============ */
    const alertBox = $('#alert-box');
    const alertMsg = $('.alert-msg');
    const results = $('#results');
    let debounceTimer = null;
    let activeAbortController = null;
    let latestRequestId = 0;

    function scheduleConvert(delay = 140) {
        if (debounceTimer) {
            clearTimeout(debounceTimer);
            debounceTimer = null;
        }
        if (!currentFile || mode !== 'image') return;
        if (delay === 0) {
            convert({ isLive: true, shouldScroll: false });
        } else {
            debounceTimer = setTimeout(() => {
                convert({ isLive: true, shouldScroll: false });
            }, delay);
        }
    }

    // Listeners reactivos para todos los controles de detección, pre-procesamiento y escala
    controls.addEventListener('input', (e) => {
        if (!currentFile || mode !== 'image') return;
        if (e.target.matches('input[type="range"]')) {
            scheduleConvert(140);
        }
    });

    controls.addEventListener('change', (e) => {
        if (!currentFile || mode !== 'image') return;
        // Checkboxes, selects y fin de arrastre de sliders
        scheduleConvert(0);
    });

    function showError(msg, { keepResults = false } = {}) {
        alertMsg.textContent = msg;
        alertBox.classList.remove('hidden');
        if (!keepResults) {
            hideResults();
        }
    }

    function hideAlert() {
        alertBox.classList.add('hidden');
    }

    function hideResults() {
        results.classList.add('hidden');
    }

    async function convert(opts = {}) {
        const initial = !!opts.initial;
        const isLive = !!opts.isLive;
        const shouldScroll = !!opts.shouldScroll;
        const isManual = !!opts.isManual;

        if (!currentFile) {
            showError(t('err.upload'));
            return;
        }

        if (debounceTimer) {
            clearTimeout(debounceTimer);
            debounceTimer = null;
        }

        hideAlert();

        if (initial) {
            hideResults();
        } else {
            results.classList.add('is-updating');
        }

        const fd = new FormData();
        fd.append('file', currentFile);

        if (mode === 'dxf') {
            fd.append('pattern_name', 'DXF_Pattern');
        } else {
            const num = (id) => parseFloat($('#' + id).value);
            const bool = (id) => $('#' + id).checked;
            fd.append('method', methodSel.value);
            fd.append('canny_low', $('#ctl-cannylow').value);
            fd.append('canny_high', $('#ctl-cannyhigh').value);
            fd.append('blur_size', $('#ctl-blur').value);
            fd.append('param1', $('#ctl-p1').value);
            fd.append('param2', methodSel.value === 'hough' ? $('#ctl-p2hough').value
                : methodSel.value === 'contour' ? (num('ctl-p2cont') / 1000).toString()
                : '0');
            fd.append('use_clahe', bool('ctl-clahe'));
            fd.append('clahe_clip', $('#ctl-claheclip').value);
            fd.append('use_adaptive', bool('ctl-adaptive'));
            fd.append('adaptive_block', $('#ctl-adapblock').value);
            fd.append('adaptive_c', $('#ctl-adapc').value);
            fd.append('use_auto_canny', bool('ctl-autocanny'));
            fd.append('filter_mode', $('#ctl-filter').value);
            fd.append('use_blackhat', bool('ctl-blackhat'));
            fd.append('blackhat_ksize', $('#ctl-blackhatk').value);
            fd.append('use_skeleton', bool('ctl-skeleton'));
            fd.append('merge_segments', bool('ctl-merge'));
            fd.append('merge_angle_tol', '5.0');
            fd.append('merge_gap_tol', '10.0');
            fd.append('dedup_auto', 'true');
            fd.append('dedup_k', '2.5');
            fd.append('dedup_threshold', '8.0');
            fd.append('offset_x', (num('ctl-offx') / 100).toFixed(2));
            fd.append('offset_y', (num('ctl-offy') / 100).toFixed(2));
            fd.append('max_resolution', $('#ctl-maxres').value);
            fd.append('tile_mm', $('#ctl-tilemm').value);
            fd.append('min_dash_mm', '0.3');
            fd.append('min_gap_mm', '0.3');
        }

        const endpoint = mode === 'dxf' ? '/api/convert/dxf' : '/api/convert/image';

        if (activeAbortController) {
            activeAbortController.abort();
        }
        activeAbortController = new AbortController();
        const reqId = ++latestRequestId;

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                body: fd,
                signal: activeAbortController.signal,
            });
            if (reqId !== latestRequestId) return;
            const json = await res.json();
            if (!res.ok) {
                throw new Error(json.error || t('err.network'));
            }
            renderResults(json, { initial, shouldScroll: shouldScroll || isManual });
        } catch (err) {
            if (err.name === 'AbortError') {
                return;
            }
            if (reqId !== latestRequestId) return;
            showError(err.message || t('err.network'), { keepResults: !initial && !!lastPat });
        } finally {
            if (reqId === latestRequestId) {
                results.classList.remove('is-updating');
            }
        }
    }

    /* ============ Results ============ */
    let lastPat = null;

    function renderResults(json, { initial = false, shouldScroll = false } = {}) {
        lastPat = json.pat_content;

        $('#stats').textContent = json.stats || '';
        const warnBox = $('#warnings');
        warnBox.innerHTML = '';
        (json.warnings || []).forEach((w) => {
            const div = document.createElement('div');
            div.className = 'warning-item';
            div.textContent = w;
            warnBox.appendChild(div);
        });

        $('#preview-img').src = json.preview || '';
        $('#debug-img').src = json.debug || '';

        const scaleSlider = $('#scale-slider');
        if (initial || results.classList.contains('hidden')) {
            scaleSlider.value = 10;
            $('output[for="scale-slider"]').value = '1.0x';
            applyPreviewScale(1.0);
        } else {
            const curScale = parseFloat(scaleSlider.value) / 10;
            applyPreviewScale(curScale);
        }

        $('#pat-code').value = json.pat_content;

        const wasHidden = results.classList.contains('hidden');
        results.classList.remove('hidden');

        if (shouldScroll || (wasHidden && !initial)) {
            results.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'nearest' });
        }
    }

    function applyPreviewScale(scale) {
        $('#preview-img').style.transform = `scale(${scale})`;
    }

    $('#scale-slider').addEventListener('input', (e) => {
        const scale = parseFloat(e.target.value) / 10;
        $('output[for="scale-slider"]').value = scale.toFixed(1) + 'x';
        applyPreviewScale(scale);
    });

    $('#copy-btn').addEventListener('click', async () => {
        if (!lastPat) return;
        try {
            await navigator.clipboard.writeText(lastPat);
        } catch (_) {
            const ta = $('#pat-code');
            ta.select();
            document.execCommand('copy');
        }
        const label = $('#copy-btn span');
        label.textContent = t('res.copied');
        setTimeout(() => { label.textContent = t('res.copy'); }, 1600);
    });

    $('#download-btn').addEventListener('click', () => {
        if (!lastPat) return;
        const blob = new Blob([lastPat], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'HATCH.it.pat';
        a.click();
        URL.revokeObjectURL(url);
    });

    /* ============ Reveals ============ */
    const io = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                const el = entry.target;
                if (document.body.classList.contains('is-loaded')) {
                    el.classList.add('is-visible');
                } else {
                    pendingReveals.push(el);
                }
                io.unobserve(el);
            }
        });
    }, { threshold: 0.12 });
    $$('.reveal').forEach((el) => io.observe(el));

    /* ============ Health check ============ */
    fetch('/api/health').catch(() => {});
})();