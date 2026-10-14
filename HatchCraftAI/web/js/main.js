(() => {
    'use strict';

    /* ============ i18n ============ */
    const I18N = {
        es: {
            'hero.eyebrow': 'PATAGONIABIM / HERRAMIENTAS',
            'hero.subtitle': 'Patrones de relleno para Revit, generados desde imágenes y DXF.',
            'mode.image': 'IMAGEN',
            'mode.dxf': 'DXF',
            'mode.draw': 'DIBUJAR',
            'draw.snapgrid': 'Snap — rejilla',
            'draw.snaportho': 'Snap — ortogonal / 45°',
            'draw.snappoints': 'Snap — puntos',
            'draw.undo': 'DESHACER',
            'draw.clear': 'LIMPIAR',
            'draw.tilemm': 'TILE REAL (MM)',
            'draw.hint': 'Click para fijar el punto inicial; mueve y haz click de nuevo para trazar la línea.',
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
            'ctl.patterntype': '00 PATRÓN DE RELLENO',
            'ctl.patterntype_title': 'PATRÓN DE RELLENO',
            'pat.model_short': 'MODELO',
            'pat.drafting_short': 'DISEÑO',
            'pat.model': 'Modelo — usa escala real',
            'pat.drafting': 'Diseño/Detalle — escala fija de símbolo',
            'ctl.maxres': 'RESOLUCIÓN DE TRABAJO (PX)',
            'ctl.offx': 'OFFSET X',
            'ctl.offy': 'OFFSET Y',
            'ctl.tilemm': 'ESCALA REAL DEL TILE (MM)',
            'unit.cm': 'CM',
            'unit.in': 'PULGADAS',
            'unit.title': 'Unidades de exportación del patrón',
            'cad.rotate': 'Rotar 90° horario',
            'method.hough': 'Líneas rectas — Hough',
            'method.lsd': 'Líneas precisas — LSD',
            'method.contour': 'Formas orgánicas — Contornos',
            'method.precision': 'Precisión — vectorial exacto',
            'ctl.precision': 'PRECISIÓN',
            'ctl.precmode': 'MODO',
            'prec.auto': 'Auto',
            'prec.centerline': 'Dibujo',
            'prec.region': 'Foto',
            'prec.strict': 'Estricto',
            'ctl.sagitta': 'NIVEL DE DETALLE',
            'ctl.corner': 'NITIDEZ DE ESQUINAS (ÁNGULO MÁX. °)',
            'ctl.snap': 'RIGIDEZ GEOMÉTRICA (SNAP °)',
            'btn.auto': 'AUTO — AJUSTAR AUTOMÁTICO',
            'view.base': 'BASE',
            'view.detect': 'RED',
            'view.types': 'TIPOS',
            'view.heat': 'HEATMAP',
            'filter.gaussian': 'Gaussiano',
            'filter.bilateral': 'Bilateral — preserva bordes',
            'filter.none': 'Ninguno',
            'btn.convert': 'GENERAR PATRÓN',
            'btn.converting': 'PROCESANDO...',
            'btn.updating': 'ACTUALIZANDO...',
            'res.live': 'EN VIVO',
            'res.preview': 'SIMULACIÓN',
            'res.tile3': '3x3 TILES',
            'res.scale': 'ESCALA',
            'res.detected': 'LÍNEAS DETECTADAS',
            'res.code': 'CÓDIGO .PAT',
            'res.copy': 'COPIAR',
            'res.copied': 'COPIADO',
            'res.download': 'DESCARGAR .PAT',
            'res.downloaddxf': 'DESCARGAR .DXF',
            'cad.trim': 'EX/TR',
            'cad.draw': 'LINEA',
            'cad.delete': 'DEL',
            'cad.undo': 'DESHACER',
            'cad.reset': 'REINICIAR',
            'cad.hint.trim': 'Límite infinito: clic en una línea para fijar el LÍMITE. Clic en otra para cortar (Trim) o alargar (Extend). Clic en vacío para deseleccionar.',
            'cad.hint.draw': 'Clic para fijar inicio; clic para fijar fin. Snaps automáticos a extremos y rejilla.',
            'cad.hint.delete': 'Clic sobre cualquier línea para eliminarla.',
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
            'mode.draw': 'DRAW',
            'draw.snapgrid': 'Snap — grid',
            'draw.snaportho': 'Snap — orthogonal / 45°',
            'draw.snappoints': 'Snap — points',
            'draw.undo': 'UNDO',
            'draw.clear': 'CLEAR',
            'draw.tilemm': 'REAL TILE (MM)',
            'draw.hint': 'Click to set the start point; move and click again to draw the line.',
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
            'ctl.patterntype': '00 FILL PATTERN',
            'ctl.patterntype_title': 'FILL PATTERN',
            'pat.model_short': 'MODEL',
            'pat.drafting_short': 'DRAFTING',
            'pat.model': 'Model — real scale',
            'pat.drafting': 'Drafting/Detail — fixed symbol scale',
            'ctl.maxres': 'WORKING RESOLUTION (PX)',
            'ctl.offx': 'OFFSET X',
            'ctl.offy': 'OFFSET Y',
            'unit.cm': 'CM',
            'unit.in': 'INCHES',
            'unit.title': 'Pattern export measurement units',
            'cad.rotate': 'Rotate 90° clockwise',
            'method.hough': 'Straight lines — Hough',
            'method.lsd': 'Precise lines — LSD',
            'method.contour': 'Organic shapes — Contours',
            'method.precision': 'Precision — exact vector',
            'ctl.precision': 'PRECISION',
            'ctl.precmode': 'MODE',
            'prec.auto': 'Auto',
            'prec.centerline': 'Drawing',
            'prec.region': 'Photo',
            'prec.strict': 'Strict',
            'ctl.sagitta': 'DETAIL LEVEL',
            'ctl.corner': 'CORNER SHARPNESS (MAX ANGLE °)',
            'ctl.snap': 'GEOMETRIC RIGIDITY (SNAP °)',
            'btn.auto': 'AUTO — TUNE AUTOMATICALLY',
            'view.base': 'BASE',
            'view.detect': 'NET',
            'view.types': 'TYPES',
            'view.heat': 'HEATMAP',
            'filter.gaussian': 'Gaussian',
            'filter.bilateral': 'Bilateral — preserves edges',
            'filter.none': 'None',
            'btn.convert': 'GENERATE PATTERN',
            'btn.converting': 'PROCESSING...',
            'btn.updating': 'UPDATING...',
            'res.live': 'LIVE',
            'res.preview': 'SIMULATION',
            'res.tile3': '3x3 TILES',
            'res.scale': 'SCALE',
            'res.detected': 'DETECTED LINES',
            'res.code': '.PAT CODE',
            'res.copy': 'COPY',
            'res.copied': 'COPIED',
            'res.download': 'DOWNLOAD .PAT',
            'res.downloaddxf': 'DOWNLOAD .DXF',
            'cad.trim': 'EX/TR',
            'cad.draw': 'LINEA',
            'cad.delete': 'DEL',
            'cad.undo': 'UNDO',
            'cad.reset': 'RESET',
            'cad.hint.trim': 'Infinite boundary: click a line to set BOUNDARY. Click another line to Trim or Extend. Click empty space to deselect.',
            'cad.hint.draw': 'Click to set start; click to set end. Automatic snaps to endpoints and grid.',
            'cad.hint.delete': 'Click on any line to delete it.',
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
    const pendingReveals = [];

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
        if (id === 'ctl-sagitta') {
            const SAG = { 1: '0.02mm', 2: '0.05mm', 3: '0.10mm', 4: '0.20mm', 5: '0.40mm' };
            return SAG[Math.round(parseFloat(value))] || value;
        }
        if (id === 'ctl-snap') {
            return parseFloat(value).toFixed(1) + '°';
        }
        if (id === 'ctl-corner') {
            return Math.round(parseFloat(value)) + '°';
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

    /* ============ Loader ============ */
    const loader = $('#loader');
    const loaderBar = $('#loader-bar-fill') || $('.loader-bar-fill');
    if (loader && loaderBar) {
        const MIN_LOAD = 1400;
        const FADE_MS = 1200;
        const start = performance.now();
        let faded = false;

        function easeOutCubic(t) {
            return 1 - Math.pow(1 - t, 3);
        }

        function easeInOutQuad(p) {
            return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
        }

        function frame(now) {
            const p = Math.min((now - start) / MIN_LOAD, 1);
            loaderBar.style.transform = 'scaleX(' + easeOutCubic(p).toFixed(4) + ')';
            if (p < 1) {
                requestAnimationFrame(frame);
            } else {
                beginFade();
            }
        }

        function beginFade() {
            if (faded) return;
            faded = true;
            document.body.classList.add('is-loaded');
            pendingReveals.forEach((el, i) => {
                el.style.transitionDelay = Math.min(i * 140, 560) + 'ms';
                el.classList.add('is-visible');
            });
            pendingReveals.length = 0;
            const fstart = performance.now();
            (function fadeStep(now) {
                const p = Math.min((now - fstart) / FADE_MS, 1);
                loader.style.opacity = (1 - easeInOutQuad(p)).toFixed(4);
                if (p < 1) {
                    requestAnimationFrame(fadeStep);
                } else {
                    loader.style.opacity = '0';
                    loader.style.display = 'none';
                    if (window.parent && window.parent !== window) {
                        window.parent.postMessage({ type: 'app-ready' }, '*');
                    }
                }
            })(fstart);
        }

        const loadDone = new Promise((resolve) => {
            if (document.readyState === 'complete') {
                resolve();
            } else {
                window.addEventListener('load', resolve, { once: true });
            }
        });

        Promise.all([
            new Promise((r) => setTimeout(r, MIN_LOAD)),
            loadDone,
        ]).then(beginFade);

        requestAnimationFrame(frame);
    } else {
        document.body.classList.add('is-loaded');
        if (window.parent && window.parent !== window) {
            window.parent.postMessage({ type: 'app-ready' }, '*');
        }
    }

    /* ============ Close / Return to PatagoniaBIM ============ */
    const siteX = $('#site-x');
    if (siteX) {
        if (window.self !== window.top) {
            siteX.style.display = 'none';
        }
    }

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
    const drawPanel = $('#draw-panel');

    function syncControlsMode() {
        // En modo DXF solo aplica el selector de unidades (grupo 03);
        // los grupos de imagen se ocultan. En modo DIBUJAR no aplica nada.
        const isDraw = mode === 'draw';
        const isDxf = mode === 'dxf';
        controls.classList.toggle('hidden', isDraw);
        $$('#controls [data-image-only]').forEach((el) => {
            el.classList.toggle('hidden', isDxf);
        });
        const unitSel = $('.group-unit');
        if (unitSel) unitSel.classList.toggle('hidden', !isDxf);
    }

    $$('.mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            mode = btn.dataset.mode;
            const isDraw = mode === 'draw';
            $$('.mode-btn').forEach((b) => b.classList.toggle('is-active', b === btn));
            syncControlsMode();
            if (!isDraw) {
                dropSub.textContent = mode === 'dxf'
                    ? (lang === 'es' ? 'Archivo DXF — máx. 12 MB' : 'DXF file — max 12 MB')
                    : t('drop.sub');
                fileInput.accept = mode === 'dxf' ? '.dxf' : '.png,.jpg,.jpeg';
            }
            clearFile();
            hideResults();
            if (drawPanel) drawPanel.classList.toggle('hidden', !isDraw);
            if (dz) dz.classList.toggle('hidden', isDraw);
            if (isDraw && typeof drawCanvasResize === 'function') {
                drawCanvasResize();
                redrawDrawCanvas();
            }
        });
    });

    /* ============ Dropzone ============ */
    const dz = $('#dropzone');
    const fileInput = $('#file-input');
    let currentFile = null;
    let pendingAutoTune = false;

    fileInput.accept = '.png,.jpg,.jpeg,.dxf';

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
        const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
        const isDxf = ext === '.dxf';
        const isImg = ['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(ext);

        if (!isDxf && !isImg) {
            showError(lang === 'es'
                ? 'Formato no válido. Usa PNG, JPG o DXF'
                : 'Invalid format. Use PNG, JPG or DXF');
            return;
        }
        if (file.size > 12 * 1024 * 1024) {
            showError(lang === 'es' ? 'El archivo supera 12 MB' : 'File exceeds 12 MB');
            return;
        }

        // Auto-reconocimiento del modo (DXF o Imagen)
        mode = isDxf ? 'dxf' : 'image';
        syncControlsMode();

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
        mode = 'image';
        syncControlsMode();
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
        const defaults = { hough: 20, lsd: 15, contour: 20, precision: 20 };
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

    // Botón AUTO (precisión): pide auto-tuning en la próxima conversión
    const autoBtn = $('#btn-auto');
    if (autoBtn) {
        autoBtn.addEventListener('click', () => {
            if (!currentFile || mode !== 'image') {
                showError(lang === 'es'
                    ? 'Carga una imagen primero'
                    : 'Load an image first');
                return;
            }
            if (methodSel.value !== 'precision') {
                methodSel.value = 'precision';
                syncMethod();
            }
            pendingAutoTune = true;
            convert({ isLive: true, shouldScroll: true });
        });
    }

    /* ============ Draw canvas (modo DIBUJAR) ============ */
    const drawCanvas = $('#draw-canvas');
    const dctx = drawCanvas ? drawCanvas.getContext('2d') : null;
    const DRAW_PAD = 10;
    let drawSegments = [];
    let drawStart = null;
    let drawHover = null;
    let drawDebounce = null;
    let drawCssSize = 280;

    function drawTileMm() {
        const el = $('#draw-tilemm');
        const v = el ? parseFloat(el.value) : 100;
        return (isFinite(v) && v > 0) ? v : 100;
    }

    function drawCanvasResize() {
        if (!drawCanvas || !dctx) return;
        const w = Math.max(200, Math.round(drawCanvas.clientWidth || 320));
        drawCssSize = w;
        const dpr = Math.min(devicePixelRatio || 1, 2);
        drawCanvas.width = w * dpr;
        drawCanvas.height = w * dpr;
        drawCanvas.style.height = w + 'px';
        dctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function drawSnapPoints() {
        const pts = [];
        drawSegments.forEach((s) => {
            pts.push({ x: s.x1, y: s.y1 });
            pts.push({ x: s.x2, y: s.y2 });
            pts.push({ x: (s.x1 + s.x2) / 2, y: (s.y1 + s.y2) / 2 });
        });
        return pts;
    }

    function applyDrawSnap(p) {
        if (!p) return { x: 0, y: 0, snapped: false };
        const inner = drawCssSize - 2 * DRAW_PAD;
        let { x, y } = p;
        let snapped = false;
        if ($('#draw-snap-points') && $('#draw-snap-points').checked) {
            let best = null;
            let bestD = 8;
            drawSnapPoints().forEach((q) => {
                const d = Math.hypot(q.x - x, q.y - y);
                if (d < bestD) { bestD = d; best = q; }
            });
            if (best) return { x: best.x, y: best.y, snapped: true };
        }
        if (drawStart && $('#draw-snap-ortho') && $('#draw-snap-ortho').checked) {
            const dx = x - drawStart.x;
            const dy = y - drawStart.y;
            const len = Math.hypot(dx, dy);
            if (len > 1e-6) {
                const ang = Math.atan2(dy, dx);
                const step = Math.PI / 4;
                const q = Math.round(ang / step) * step;
                x = drawStart.x + Math.cos(q) * len;
                y = drawStart.y + Math.sin(q) * len;
                snapped = true;
            }
        }
        if ($('#draw-snap-grid') && $('#draw-snap-grid').checked) {
            const g = inner / 8;
            x = Math.round((x - DRAW_PAD) / g) * g + DRAW_PAD;
            y = Math.round((y - DRAW_PAD) / g) * g + DRAW_PAD;
            snapped = true;
        }
        x = Math.min(drawCssSize - DRAW_PAD, Math.max(DRAW_PAD, x));
        y = Math.min(drawCssSize - DRAW_PAD, Math.max(DRAW_PAD, y));
        return { x, y, snapped };
    }

    function redrawDrawCanvas() {
        if (!drawCanvas || !dctx) return;
        const inner = drawCssSize - 2 * DRAW_PAD;
        dctx.clearRect(0, 0, drawCssSize, drawCssSize);
        dctx.strokeStyle = 'rgba(255,255,255,0.08)';
        dctx.lineWidth = 1;
        dctx.beginPath();
        for (let i = 0; i <= 8; i++) {
            const v = DRAW_PAD + (inner * i) / 8;
            dctx.moveTo(v, DRAW_PAD);
            dctx.lineTo(v, drawCssSize - DRAW_PAD);
            dctx.moveTo(DRAW_PAD, v);
            dctx.lineTo(drawCssSize - DRAW_PAD, v);
        }
        dctx.stroke();
        dctx.strokeStyle = 'rgba(255,215,0,0.25)';
        dctx.strokeRect(DRAW_PAD, DRAW_PAD, inner, inner);
        dctx.strokeStyle = '#FFD700';
        dctx.lineWidth = 2;
        dctx.setLineDash([]);
        dctx.beginPath();
        drawSegments.forEach((s) => {
            dctx.moveTo(s.x1, s.y1);
            dctx.lineTo(s.x2, s.y2);
        });
        dctx.stroke();
        if (drawStart && drawHover) {
            dctx.strokeStyle = 'rgba(255,215,0,0.6)';
            dctx.setLineDash([6, 4]);
            dctx.beginPath();
            dctx.moveTo(drawStart.x, drawStart.y);
            dctx.lineTo(drawHover.x, drawHover.y);
            dctx.stroke();
            dctx.setLineDash([]);
        }
        const dot = drawHover || drawStart;
        if (dot) {
            dctx.fillStyle = '#FFD700';
            dctx.beginPath();
            dctx.arc(dot.x, dot.y, 3.5, 0, Math.PI * 2);
            dctx.fill();
        }
    }

    function canvasPos(e) {
        const r = drawCanvas.getBoundingClientRect();
        const scale = drawCssSize / (r.width || drawCssSize);
        return { x: (e.clientX - r.left) * scale, y: (e.clientY - r.top) * scale };
    }

    function scheduleDrawConvert(delay = 300) {
        if (drawDebounce) clearTimeout(drawDebounce);
        if (!drawSegments.length) {
            hideResults();
            return;
        }
        drawDebounce = setTimeout(() => {
            convert({ isLive: true, shouldScroll: false });
        }, delay);
    }

    if (drawCanvas && dctx) {
        drawCanvasResize();
        redrawDrawCanvas();
        addEventListener('resize', () => {
            drawCanvasResize();
            redrawDrawCanvas();
        });
        drawCanvas.addEventListener('pointermove', (e) => {
            drawHover = applyDrawSnap(canvasPos(e));
            redrawDrawCanvas();
        });
        drawCanvas.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            const p = applyDrawSnap(canvasPos(e));
            if (!drawStart) {
                drawStart = { x: p.x, y: p.y };
            } else {
                if (Math.hypot(p.x - drawStart.x, p.y - drawStart.y) > 2) {
                    drawSegments.push({ x1: drawStart.x, y1: drawStart.y, x2: p.x, y2: p.y });
                    scheduleDrawConvert();
                }
                drawStart = null;
            }
            drawHover = null;
            redrawDrawCanvas();
        });
        drawCanvas.addEventListener('pointerleave', () => {
            drawHover = null;
            redrawDrawCanvas();
        });
        const undoBtn = $('#draw-undo');
        if (undoBtn) {
            undoBtn.addEventListener('click', () => {
                if (drawStart) {
                    drawStart = null;
                } else {
                    drawSegments.pop();
                }
                redrawDrawCanvas();
                scheduleDrawConvert();
            });
        }
        const clearBtn = $('#draw-clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                drawSegments = [];
                drawStart = null;
                drawHover = null;
                redrawDrawCanvas();
                hideResults();
            });
        }
        const tileEl = $('#draw-tilemm');
        if (tileEl) {
            tileEl.addEventListener('change', () => scheduleDrawConvert(0));
        }
    }

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

    // Unidad de exportación (cm / pulgadas) — selector en "ESCALA Y GEOMETRÍA"
    function getExportUnit() {
        const checked = $('input[name="exportunit"]:checked');
        return checked ? checked.value : 'cm';
    }

    $$('input[name="exportunit"]').forEach((radio) => {
        radio.addEventListener('change', (e) => {
            e.stopPropagation();
            if (currentFile && mode === 'dxf') {
                convert({ isLive: true, shouldScroll: false });
            }
        });
    });

    // Tipo de patrón 00 (MODEL/DRAFTING)
    const patTypeHidden = $('#ctl-patterntype');    $$('input[name="patterntype"]').forEach((radio) => {
        radio.addEventListener('change', () => {
            if (patTypeHidden) {
                patTypeHidden.value = radio.value;
                patTypeHidden.dispatchEvent(new Event('change'));
            }
        });
    });

    if (patTypeHidden) {
        patTypeHidden.addEventListener('change', () => {
            const currentRadio = $(`input[name="patterntype"][value="${patTypeHidden.value}"]`);
            if (currentRadio && !currentRadio.checked) currentRadio.checked = true;
            if (mode !== 'draw' && !currentFile) return;
            if (mode === 'draw' && !drawSegments.length) return;
            convert({ isLive: true, shouldScroll: false });
        });
    }

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

        if (mode === 'draw') {
            await convertDrawing(opts);
            return;
        }

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
            fd.append('pattern_type', $('#ctl-patterntype').value);
            fd.append('unit', getExportUnit());
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
            fd.append('pattern_type', $('#ctl-patterntype').value);
            if (methodSel.value === 'precision') {
                const SAG = { 1: 0.02, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.4 };
                const sagVal = Math.round(num('ctl-sagitta'));
                const sagitta = SAG[sagVal] || 0.1;
                const precmode = ($('input[name="precmode"]:checked') || {}).value || 'auto';
                let corner = num('ctl-corner');
                let snap = num('ctl-snap');
                let fmode = '';
                if (precmode === 'centerline') fmode = 'centerline';
                if (precmode === 'region') fmode = 'region';
                if (precmode === 'strict') { fmode = 'centerline'; snap = 5; corner = 150; }
                fd.append('curves', 'fit');
                fd.append('corner_angle', String(corner));
                fd.append('sagitta_mm', String(sagitta));
                fd.append('budget', '400');
                fd.append('snap_tol', String(snap));
                fd.append('periodic', 'auto');
                if (fmode) fd.append('force_mode', fmode);
                fd.append('auto', pendingAutoTune ? 'true' : 'false');
                pendingAutoTune = false;
            }
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
                const msg = (json.detail && (json.detail.error || (typeof json.detail === 'string' ? json.detail : null))) || json.error || t('err.network');
                throw new Error(msg);
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

    async function convertDrawing(opts = {}) {
        const initial = !!opts.initial;
        const shouldScroll = !!opts.shouldScroll;

        if (!drawSegments.length) {
            hideResults();
            return;
        }
        if (drawDebounce) {
            clearTimeout(drawDebounce);
            drawDebounce = null;
        }
        hideAlert();
        if (initial) {
            hideResults();
        } else {
            results.classList.add('is-updating');
        }

        const mmPerPx = drawTileMm() / (drawCssSize - 2 * DRAW_PAD);
        const toMmX = (x) => (x - DRAW_PAD) * mmPerPx;
        const toMmY = (y) => ((drawCssSize - DRAW_PAD) - y) * mmPerPx;
        const segments = drawSegments.map((s) => [
            toMmX(s.x1), toMmY(s.y1), toMmX(s.x2), toMmY(s.y2),
        ]);

        if (activeAbortController) {
            activeAbortController.abort();
        }
        activeAbortController = new AbortController();
        const reqId = ++latestRequestId;

        try {
            const res = await fetch('/api/convert/segments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    segments,
                    tile_mm: drawTileMm(),
                    pattern_name: 'Draw_Pattern',
                    pattern_type: $('#ctl-patterntype').value,
                }),
                signal: activeAbortController.signal,
            });
            if (reqId !== latestRequestId) return;
            const json = await res.json();
            if (!res.ok) {
                const msg = (json.detail && (json.detail.error || (typeof json.detail === 'string' ? json.detail : null))) || json.error || t('err.network');
                throw new Error(msg);
            }
            json.debug = json.preview;
            renderResults(json, { initial, shouldScroll });
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

    /* ============ CAD Interactive Engine (Comparativa) ============ */
    const cadCanvas = $('#cad-canvas');
    const cadCtx = cadCanvas ? cadCanvas.getContext('2d') : null;
    const cadHintTxt = $('#cad-hint');

    let cadSegments = [];          // [[x1, y1, x2, y2], ...] in mm
    let cadInitialSegments = [];   // copy for reset
    let cadUndoStack = [];         // history stack of segment arrays
    let cadTool = 'trim';          // 'trim' | 'draw' | 'del'
    let cadBoundaryIdx = -1;       // index of active boundary segment (-1 if none)
    let cadHoverIdx = -1;          // hovered segment index
    let cadTileW = 100;            // tile width in mm
    let cadTileH = 100;            // tile height in mm
    let cadDrawStart = null;       // { x, y } in mm
    let cadHoverPoint = null;      // { x, y } in mm
    let cadRotation = 0;           // giros de 90° horarios aplicados (0-3)
    let cadInitialTileW = 100;     // dimensiones originales (para reset)
    let cadInitialTileH = 100;
    let cadDebounceTimer = null;
    let cadActiveAbort = null;
    let cadReqId = 0;

    function getCadTransform() {
        if (!cadCanvas) return { scale: 1, offX: 0, offY: 0, w: 100, h: 100, tileW: 100, tileH: 100 };
        const rect = cadCanvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const tileW = Math.max(cadTileW, 1);
        const tileH = Math.max(cadTileH, 1);
        const scale = Math.min(w / tileW, h / tileH);
        const offX = (w - tileW * scale) / 2;
        const offY = (h - tileH * scale) / 2;
        return { scale, offX, offY, w, h, tileW, tileH };
    }

    function getCadSegmentColor(s) {
        const dx = s[2] - s[0];
        const dy = s[3] - s[1];
        let deg = (Math.atan2(dy, dx) * 180) / Math.PI;
        if (deg < 0) deg += 180;
        if (deg >= 180) deg -= 180;
        if (deg < 12 || deg > 168) return '#DC3232'; // Rojo = horizontal
        if (Math.abs(deg - 90) < 12) return '#3232DC'; // Azul = vertical
        if (Math.abs(deg - 45) < 12) return '#32B432'; // Verde = 45°
        if (Math.abs(deg - 135) < 12) return '#B432B4'; // Morado = 135°
        return '#707070'; // Gris = otro
    }

    function cadMmToPx(x_mm, y_mm, tf) {
        return {
            x: tf.offX + x_mm * tf.scale,
            y: tf.offY + (tf.tileH - y_mm) * tf.scale
        };
    }

    function cadPxToMm(px_x, px_y, tf) {
        return {
            x: (px_x - tf.offX) / tf.scale,
            y: tf.tileH - ((px_y - tf.offY) / tf.scale)
        };
    }

    function distPtToSegmentPx(px, py, x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        const lenSq = dx * dx + dy * dy;
        if (lenSq < 1e-6) return Math.hypot(px - x1, py - y1);
        let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    }

    function findClosestSegment(px, py, tf, maxDistPx = 14) {
        let bestIdx = -1;
        let bestDist = maxDistPx;
        for (let i = 0; i < cadSegments.length; i++) {
            const s = cadSegments[i];
            const p1 = cadMmToPx(s[0], s[1], tf);
            const p2 = cadMmToPx(s[2], s[3], tf);
            const d = distPtToSegmentPx(px, py, p1.x, p1.y, p2.x, p2.y);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    function deleteSegmentsAt(px, py, tf, maxDistPx = 18) {
        const closestIdx = findClosestSegment(px, py, tf, maxDistPx);
        if (closestIdx < 0) return false;

        cadPushUndo();
        const toRemove = new Set();
        toRemove.add(closestIdx);

        const target = cadSegments[closestIdx];
        const tp1 = cadMmToPx(target[0], target[1], tf);
        const tp2 = cadMmToPx(target[2], target[3], tf);
        const tAng = Math.atan2(tp2.y - tp1.y, tp2.x - tp1.x);

        for (let i = 0; i < cadSegments.length; i++) {
            if (i === closestIdx) continue;
            const s = cadSegments[i];
            const p1 = cadMmToPx(s[0], s[1], tf);
            const p2 = cadMmToPx(s[2], s[3], tf);
            const d = distPtToSegmentPx(px, py, p1.x, p1.y, p2.x, p2.y);
            if (d <= maxDistPx) {
                const sAng = Math.atan2(p2.y - p1.y, p2.x - p1.x);
                let dAng = Math.abs(tAng - sAng);
                while (dAng > Math.PI) dAng = Math.abs(dAng - Math.PI);
                if (dAng < 0.2 || Math.abs(dAng - Math.PI) < 0.2) {
                    toRemove.add(i);
                }
            }
        }

        cadSegments = cadSegments.filter((_, i) => !toRemove.has(i));
        cadBoundaryIdx = -1;
        cadHoverIdx = -1;
        renderCadCanvas();
        scheduleCadSync();
        return true;
    }

    function intersectInfiniteLines(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y) {
        const dpx = p2x - p1x;
        const dpy = p2y - p1y;
        const dqx = q2x - q1x;
        const dqy = q2y - q1y;
        const det = dpx * dqy - dpy * dqx;
        if (Math.abs(det) < 1e-9) return null;
        const t = ((q1x - p1x) * dqy - (q1y - p1y) * dqx) / det;
        return {
            x: p1x + t * dpx,
            y: p1y + t * dpy
        };
    }

    function performTrimExtend(targetIdx, clickMm) {
        if (cadBoundaryIdx < 0 || cadBoundaryIdx >= cadSegments.length) return false;
        if (targetIdx === cadBoundaryIdx) return false;

        const b = cadSegments[cadBoundaryIdx];
        const s = cadSegments[targetIdx];

        const inter = intersectInfiniteLines(b[0], b[1], b[2], b[3], s[0], s[1], s[2], s[3]);
        if (!inter) return false;

        const sdx = s[2] - s[0];
        const sdy = s[3] - s[1];
        const sLenSq = sdx * sdx + sdy * sdy;
        if (sLenSq < 1e-6) return false;

        const u = ((inter.x - s[0]) * sdx + (inter.y - s[1]) * sdy) / sLenSq;

        cadPushUndo();

        // Si la intersección cae estrictamente dentro del segmento: TRIM
        if (u > 0.005 && u < 0.995) {
            const d1 = distPtToSegmentPx(clickMm.x, clickMm.y, s[0], s[1], inter.x, inter.y);
            const d2 = distPtToSegmentPx(clickMm.x, clickMm.y, inter.x, inter.y, s[2], s[3]);
            if (d1 < d2) {
                cadSegments[targetIdx] = [inter.x, inter.y, s[2], s[3]];
            } else {
                cadSegments[targetIdx] = [s[0], s[1], inter.x, inter.y];
            }
        } else {
            // Si la intersección cae fuera: EXTEND
            const distStart = Math.hypot(s[0] - inter.x, s[1] - inter.y);
            const distEnd = Math.hypot(s[2] - inter.x, s[3] - inter.y);
            if (distStart < distEnd) {
                cadSegments[targetIdx] = [inter.x, inter.y, s[2], s[3]];
            } else {
                cadSegments[targetIdx] = [s[0], s[1], inter.x, inter.y];
            }
        }
        return true;
    }

    function cadPushUndo() {
        cadUndoStack.push({
            segments: cadSegments.map((s) => [...s]),
            tileW: cadTileW,
            tileH: cadTileH,
            rotation: cadRotation,
        });
        if (cadUndoStack.length > 30) cadUndoStack.shift();
    }

    function setCadRotation(rot) {
        cadRotation = ((rot % 4) + 4) % 4;
        const dbg = $('#debug-img');
        if (dbg) {
            dbg.style.transform = cadRotation ? `rotate(${cadRotation * 90}deg)` : '';
        }
    }

    function cadRotate90() {
        if (!cadSegments.length) return;
        cadPushUndo();
        // 90° horario (Y-up): (x, y) → (y, W − x)
        const w = cadTileW;
        cadSegments = cadSegments.map((s) => [s[1], w - s[0], s[3], w - s[2]]);
        const t = cadTileW;
        cadTileW = cadTileH;
        cadTileH = t;
        setCadRotation(cadRotation + 1);
        cadBoundaryIdx = -1;
        cadHoverIdx = -1;
        cadDrawStart = null;
        renderCadCanvas();
        scheduleCadSync();
    }

    function cadUndo() {
        if (!cadUndoStack.length) return;
        const st = cadUndoStack.pop();
        cadSegments = st.segments;
        cadTileW = st.tileW;
        cadTileH = st.tileH;
        setCadRotation(st.rotation);
        cadBoundaryIdx = -1;
        cadHoverIdx = -1;
        cadDrawStart = null;
        renderCadCanvas();
        scheduleCadSync();
    }

    function cadReset() {
        if (!cadInitialSegments.length) return;
        cadPushUndo();
        cadSegments = cadInitialSegments.map((s) => [...s]);
        cadTileW = cadInitialTileW;
        cadTileH = cadInitialTileH;
        setCadRotation(0);
        cadBoundaryIdx = -1;
        cadHoverIdx = -1;
        cadDrawStart = null;
        renderCadCanvas();
        scheduleCadSync();
    }

    function cadFindSnap(mmX, mmY, tf) {
        let bestPt = null;
        let bestDistPx = 12;
        const curPx = cadMmToPx(mmX, mmY, tf);

        for (const s of cadSegments) {
            const ep1 = cadMmToPx(s[0], s[1], tf);
            const ep2 = cadMmToPx(s[2], s[3], tf);
            const d1 = Math.hypot(curPx.x - ep1.x, curPx.y - ep1.y);
            if (d1 < bestDistPx) {
                bestDistPx = d1;
                bestPt = { x: s[0], y: s[1] };
            }
            const d2 = Math.hypot(curPx.x - ep2.x, curPx.y - ep2.y);
            if (d2 < bestDistPx) {
                bestDistPx = d2;
                bestPt = { x: s[2], y: s[3] };
            }
        }
        if (bestPt) return bestPt;

        if (cadDrawStart) {
            const dx = mmX - cadDrawStart.x;
            const dy = mmY - cadDrawStart.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 1.5) {
                let ang = (Math.atan2(dy, dx) * 180) / Math.PI;
                const snapAngs = [0, 45, 90, 135, 180, -180, -135, -90, -45];
                let closestAng = ang;
                let minDiff = 8;
                for (const sa of snapAngs) {
                    const diff = Math.abs(ang - sa);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestAng = sa;
                    }
                }
                if (minDiff < 8) {
                    const rad = (closestAng * Math.PI) / 180;
                    return {
                        x: cadDrawStart.x + Math.cos(rad) * dist,
                        y: cadDrawStart.y + Math.sin(rad) * dist,
                    };
                }
            }
        }
        return { x: mmX, y: mmY };
    }

    function sizeCadCanvas() {
        if (!cadCanvas) return;
        const rect = cadCanvas.parentElement.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(Math.floor(rect.width), 100);
        const h = Math.max(Math.floor(rect.height || rect.width), 100);
        cadCanvas.width = w * dpr;
        cadCanvas.height = h * dpr;
        cadCanvas.style.width = w + 'px';
        cadCanvas.style.height = h + 'px';
        if (cadCtx) cadCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function renderCadCanvas() {
        if (!cadCanvas || !cadCtx) return;
        const tf = getCadTransform();
        cadCtx.clearRect(0, 0, tf.w, tf.h);

        // Marco del tile
        const p00 = cadMmToPx(0, 0, tf);
        const p11 = cadMmToPx(tf.tileW, tf.tileH, tf);
        const boxX = Math.min(p00.x, p11.x);
        const boxY = Math.min(p00.y, p11.y);
        const boxW = Math.abs(p11.x - p00.x);
        const boxH = Math.abs(p11.y - p00.y);

        cadCtx.strokeStyle = 'rgba(255, 215, 0, 0.16)';
        cadCtx.lineWidth = 1;
        cadCtx.setLineDash([4, 4]);
        cadCtx.strokeRect(boxX, boxY, boxW, boxH);
        cadCtx.setLineDash([]);

        // Guía infinita del vector límite activo
        if (cadBoundaryIdx >= 0 && cadBoundaryIdx < cadSegments.length) {
            const b = cadSegments[cadBoundaryIdx];
            const p1 = cadMmToPx(b[0], b[1], tf);
            const p2 = cadMmToPx(b[2], b[3], tf);
            const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
            const diag = Math.hypot(tf.w, tf.h) * 2;
            const infX1 = p1.x - Math.cos(angle) * diag;
            const infY1 = p1.y - Math.sin(angle) * diag;
            const infX2 = p1.x + Math.cos(angle) * diag;
            const infY2 = p1.y + Math.sin(angle) * diag;

            cadCtx.strokeStyle = 'rgba(0, 229, 255, 0.55)';
            cadCtx.lineWidth = 1.5;
            cadCtx.setLineDash([6, 6]);
            cadCtx.beginPath();
            cadCtx.moveTo(infX1, infY1);
            cadCtx.lineTo(infX2, infY2);
            cadCtx.stroke();
            cadCtx.setLineDash([]);
        }

        // Trazado de segmentos (en vista 'detect' dibuja toda la red; en otras vistas solo selección/hover interactivo)
        const showFullCadOverlay = (activeDebugView === 'detect');
        cadSegments.forEach((s, idx) => {
            const isBoundary = idx === cadBoundaryIdx;
            const isHover = idx === cadHoverIdx;
            if (!showFullCadOverlay && !isBoundary && !isHover) return;

            const p1 = cadMmToPx(s[0], s[1], tf);
            const p2 = cadMmToPx(s[2], s[3], tf);

            cadCtx.beginPath();
            cadCtx.moveTo(p1.x, p1.y);
            cadCtx.lineTo(p2.x, p2.y);

            if (isBoundary) {
                cadCtx.strokeStyle = '#00E5FF';
                cadCtx.lineWidth = 3;
                cadCtx.shadowColor = 'rgba(0, 229, 255, 0.6)';
                cadCtx.shadowBlur = 8;
            } else if (isHover) {
                const isDel = cadTool === 'del' || cadTool === 'delete';
                cadCtx.strokeStyle = isDel ? '#FF2222' : '#FFFFFF';
                cadCtx.lineWidth = 3.2;
                cadCtx.shadowColor = isDel ? 'rgba(255, 34, 34, 0.85)' : 'rgba(255, 255, 255, 0.85)';
                cadCtx.shadowBlur = 6;
            } else {
                cadCtx.strokeStyle = getCadSegmentColor(s);
                cadCtx.lineWidth = 2.0;
                cadCtx.shadowColor = 'transparent';
                cadCtx.shadowBlur = 0;
            }
            cadCtx.stroke();
            cadCtx.shadowBlur = 0;

            // Puntos extremos
            cadCtx.fillStyle = isBoundary ? '#00E5FF' : isHover ? '#FFD700' : 'rgba(255, 255, 255, 0.7)';
            cadCtx.beginPath();
            cadCtx.arc(p1.x, p1.y, 2, 0, Math.PI * 2);
            cadCtx.arc(p2.x, p2.y, 2, 0, Math.PI * 2);
            cadCtx.fill();
        });

        // Línea en construcción
        if (cadTool === 'draw' && cadDrawStart && cadHoverPoint) {
            const p1 = cadMmToPx(cadDrawStart.x, cadDrawStart.y, tf);
            const p2 = cadMmToPx(cadHoverPoint.x, cadHoverPoint.y, tf);
            cadCtx.strokeStyle = 'rgba(255, 215, 0, 0.9)';
            cadCtx.lineWidth = 2;
            cadCtx.setLineDash([5, 5]);
            cadCtx.beginPath();
            cadCtx.moveTo(p1.x, p1.y);
            cadCtx.lineTo(p2.x, p2.y);
            cadCtx.stroke();
            cadCtx.setLineDash([]);
        }

        // Indicador de snap
        if (cadTool === 'draw' && cadHoverPoint) {
            const hp = cadMmToPx(cadHoverPoint.x, cadHoverPoint.y, tf);
            cadCtx.fillStyle = '#00E5FF';
            cadCtx.beginPath();
            cadCtx.arc(hp.x, hp.y, 4, 0, Math.PI * 2);
            cadCtx.fill();
        }
    }

    function scheduleCadSync(delay = 250) {
        if (cadDebounceTimer) clearTimeout(cadDebounceTimer);
        cadDebounceTimer = setTimeout(async () => {
            if (!cadSegments.length) return;
            if (cadActiveAbort) cadActiveAbort.abort();
            cadActiveAbort = new AbortController();
            const reqId = ++cadReqId;

            const payload = {
                segments: cadSegments,
                tile_mm: Math.max(cadTileW, cadTileH),
                tile_w: cadTileW,
                tile_h: cadTileH,
                unit: mode === 'dxf' ? 'in' : 'mm',
                pattern_name: 'CAD_Pattern',
                pattern_type: $('#ctl-patterntype').value,
            };

            results.classList.add('is-updating');
            try {
                const res = await fetch('/api/convert/segments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: cadActiveAbort.signal,
                });
                if (reqId !== cadReqId) return;
                const json = await res.json();
                if (!res.ok) return;

                lastPat = json.pat_content;
                $('#pat-code').value = json.pat_content;
                $('#preview-img').src = json.preview || '';
                $('#stats').textContent = json.stats || '';
            } catch (err) {
                if (err.name !== 'AbortError') console.error(err);
            } finally {
                if (reqId === cadReqId) {
                    results.classList.remove('is-updating');
                }
            }
        }, delay);
    }

    if (cadCanvas) {
        cadCanvas.addEventListener('pointermove', (e) => {
            const rect = cadCanvas.getBoundingClientRect();
            const pxX = e.clientX - rect.left;
            const pxY = e.clientY - rect.top;
            const tf = getCadTransform();
            const mm = cadPxToMm(pxX, pxY, tf);

            if (cadTool === 'draw') {
                cadHoverPoint = cadFindSnap(mm.x, mm.y, tf);
                cadHoverIdx = -1;
                cadCanvas.style.cursor = 'crosshair';
                renderCadCanvas();
            } else {
                const idx = findClosestSegment(pxX, pxY, tf, 18);
                const isDel = cadTool === 'del' || cadTool === 'delete';
                if (isDel || cadTool === 'trim') {
                    cadCanvas.style.cursor = idx >= 0 ? 'pointer' : 'default';
                }
                if (idx !== cadHoverIdx) {
                    cadHoverIdx = idx;
                    renderCadCanvas();
                }
            }
        });

        cadCanvas.addEventListener('pointerleave', () => {
            cadHoverIdx = -1;
            cadHoverPoint = null;
            renderCadCanvas();
        });

        cadCanvas.addEventListener('pointerdown', (e) => {
            const rect = cadCanvas.getBoundingClientRect();
            const pxX = e.clientX - rect.left;
            const pxY = e.clientY - rect.top;
            const tf = getCadTransform();
            const mm = cadPxToMm(pxX, pxY, tf);

            if (cadTool === 'trim') {
                const idx = findClosestSegment(pxX, pxY, tf, 16);
                if (cadBoundaryIdx < 0) {
                    if (idx >= 0) {
                        cadBoundaryIdx = idx;
                    }
                } else {
                    if (idx >= 0 && idx !== cadBoundaryIdx) {
                        performTrimExtend(idx, mm);
                        scheduleCadSync();
                    } else if (idx < 0) {
                        cadBoundaryIdx = -1;
                    }
                }
                renderCadCanvas();
            } else if (cadTool === 'del' || cadTool === 'delete') {
                deleteSegmentsAt(pxX, pxY, tf, 18);
            } else if (cadTool === 'draw') {
                const snap = cadFindSnap(mm.x, mm.y, tf);
                if (!cadDrawStart) {
                    cadDrawStart = snap;
                } else {
                    const dist = Math.hypot(snap.x - cadDrawStart.x, snap.y - cadDrawStart.y);
                    if (dist > 0.5) {
                        cadPushUndo();
                        cadSegments.push([cadDrawStart.x, cadDrawStart.y, snap.x, snap.y]);
                        cadDrawStart = null;
                        renderCadCanvas();
                        scheduleCadSync();
                    }
                }
            }
        });
    }

    $$('.cad-btn[data-tool]').forEach((btn) => {
        btn.addEventListener('click', () => {
            cadTool = btn.dataset.tool;
            $$('.cad-btn[data-tool]').forEach((b) => b.classList.toggle('is-active', b === btn));
            cadBoundaryIdx = -1;
            cadDrawStart = null;
            if (cadHintTxt) cadHintTxt.textContent = t('cad.hint.' + cadTool);
            renderCadCanvas();
        });
    });

    const cadUndoBtn = $('#cad-undo');
    if (cadUndoBtn) cadUndoBtn.addEventListener('click', cadUndo);

    const cadResetBtn = $('#cad-reset');
    if (cadResetBtn) cadResetBtn.addEventListener('click', cadReset);

    const cadRotateBtn = $('#cad-rotate');
    if (cadRotateBtn) cadRotateBtn.addEventListener('click', cadRotate90);

    window.addEventListener('resize', () => {
        if (cadSegments.length) {
            sizeCadCanvas();
            renderCadCanvas();
        }
    });

    /* ============ Results ============ */
    let lastPat = null;
    const debugViews = { base: '', detect: '', types: '', heat: '' };
    let activeDebugView = 'detect';

    function setDebugView(v) {
        activeDebugView = v;
        $$('.cad-view-btn').forEach((b) => {
            b.classList.toggle('is-active', b.dataset.view === v);
        });
        const img = $('#debug-img');
        const src = (v === 'detect' && cadSegments.length && debugViews.base)
            ? debugViews.base
            : (debugViews[v] || debugViews.base || debugViews.detect || '');
        if (src) img.src = src;
        renderCadCanvas();
    }

    function updateAnalysisStrip(analysis) {
        const strip = $('#analysis-strip');
        if (!strip) return;
        const parts = [];
        if (analysis && Array.isArray(analysis.dominant_deg) && analysis.dominant_deg.length) {
            const doms = analysis.dominant_deg.map((d) => Math.round(d) + '°').join(', ');
            parts.push((lang === 'es' ? 'Dominantes: ' : 'Dominant: ') + doms);
        }
        const per = analysis && analysis.period;
        if (per && per.nx) {
            parts.push((lang === 'es' ? 'Período: ' : 'Period: ')
                + Math.round(per.cell_w) + '×' + Math.round(per.cell_h) + ' px ('
                + per.nx + '×' + per.ny + (lang === 'es' ? ' celdas)' : ' cells)'));
        }
        if (analysis && analysis.auto_params) {
            const ap = analysis.auto_params;
            parts.push('Auto[snap ' + parseFloat(ap.snap_tol).toFixed(1) + '°'
                + ', ' + parseFloat(ap.corner_angle).toFixed(0) + '°'
                + ', ' + parseFloat(ap.sagitta_mm).toFixed(2) + 'mm]');
        }
        strip.textContent = parts.join(' · ');
        strip.classList.toggle('hidden', parts.length === 0);
    }

    function applyAutoParamsToSliders(autoParams) {
        if (!autoParams) return;
        const SAG_INV = { 0.02: 1, 0.05: 2, 0.1: 3, 0.2: 4, 0.4: 5 };
        const setSlider = (id, val, fmt) => {
            const el = $('#' + id);
            if (!el || !isFinite(val)) return;
            el.value = val;
            setFill(el);
            const out = $('output[for="' + id + '"]');
            if (out) out.value = fmt ? fmt(val) : formatSliderOutput(id, val);
        };
        if (autoParams.sagitta_mm !== undefined) {
            const key = parseFloat(autoParams.sagitta_mm).toFixed(2);
            if (SAG_INV[key]) setSlider('ctl-sagitta', SAG_INV[key]);
        }
        if (autoParams.corner_angle !== undefined) {
            setSlider('ctl-corner', Math.round(parseFloat(autoParams.corner_angle)));
        }
        if (autoParams.snap_tol !== undefined) {
            setSlider('ctl-snap', parseFloat(autoParams.snap_tol));
        }
    }

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
        debugViews.base = (json.segments_data && json.segments_data.length && json.debug_base)
            ? json.debug_base
            : (json.debug || '');
        debugViews.detect = json.debug || debugViews.base;
        debugViews.types = json.debug_types || '';
        debugViews.heat = json.debug_heat || '';
        const btnTypes = $('#view-types');
        const btnHeat = $('#view-heat');
        if (btnTypes) btnTypes.classList.toggle('hidden', !debugViews.types);
        if (btnHeat) btnHeat.classList.toggle('hidden', !debugViews.heat);
        if (activeDebugView !== 'base' && activeDebugView !== 'detect' && !debugViews[activeDebugView]) {
            activeDebugView = 'detect';
        }
        updateAnalysisStrip(json.analysis);
        applyAutoParamsToSliders(json.analysis && json.analysis.auto_params);
        setCadRotation(0);

        $('#pat-code').value = json.pat_content;

        // Inicializar el lienzo interactivo CAD sobre la Comparativa
        if (json.segments_data && json.segments_data.length) {
            cadSegments = json.segments_data.map((s) => [...s]);
            cadInitialSegments = json.segments_data.map((s) => [...s]);
            cadUndoStack = [];
            cadBoundaryIdx = -1;
            cadHoverIdx = -1;
            cadDrawStart = null;
            cadTileW = json.tile_w || 100;
            cadTileH = json.tile_h || 100;
            cadInitialTileW = cadTileW;
            cadInitialTileH = cadTileH;
            setDebugView(activeDebugView);
            setTimeout(() => {
                sizeCadCanvas();
                renderCadCanvas();
            }, 60);
        } else {
            cadSegments = [];
            setDebugView(activeDebugView);
        }

        const wasHidden = results.classList.contains('hidden');
        results.classList.remove('hidden');

        if (shouldScroll || (wasHidden && !initial)) {
            results.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'nearest' });
        }
    }

    $$('.cad-view-btn').forEach((btn) => {
        btn.addEventListener('click', () => setDebugView(btn.dataset.view));
    });

    const copyBtn = $('#copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            if (!lastPat) return;
            try {
                await navigator.clipboard.writeText(lastPat);
            } catch (_) {
                const ta = $('#pat-code');
                if (ta) {
                    ta.select();
                    document.execCommand('copy');
                }
            }
            const label = $('#copy-btn span');
            if (label) {
                label.textContent = t('res.copied');
                setTimeout(() => { label.textContent = t('res.copy'); }, 1600);
            }
        });
    }

    async function saveFileWithFolderPicker(blob, defaultFilename, description, ext) {
        // Intenta abrir el diálogo nativo del sistema ("Guardar como...") para elegir carpeta
        if (window.showSaveFilePicker) {
            try {
                const acceptObj = {};
                acceptObj[blob.type || 'application/octet-stream'] = [ext];
                const handle = await window.showSaveFilePicker({
                    suggestedName: defaultFilename,
                    types: [{
                        description: description,
                        accept: acceptObj
                    }]
                });
                const writable = await handle.createWritable();
                await writable.write(blob);
                await writable.close();
                return true;
            } catch (err) {
                // Si el usuario canceló voluntariamente el diálogo de carpetas, no hacemos nada
                if (err && err.name === 'AbortError') return false;
            }
        }

        // Fallback estándar si no está disponible showSaveFilePicker
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = defaultFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1200);
        return true;
    }

    const downloadBtn = $('#download-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            if (!lastPat) return;
            const rawBase = currentFile ? currentFile.name.replace(/\.[^/.]+$/, '') : 'HATCH.it';
            const patType = ($('#ctl-patterntype') && $('#ctl-patterntype').value === 'DRAFTING')
                ? 'DISEÑO'
                : 'MODELO';
            const cleanBase = rawBase.replace(/_(MODELO|DISEÑO|DISENO|MODEL|DRAFTING)$/i, '');
            const filename = `${cleanBase}_${patType}.PAT`;
            const blob = new Blob([lastPat], { type: 'text/plain' });
            await saveFileWithFolderPicker(blob, filename, 'Patrón de relleno Revit (.PAT)', '.PAT');
        });
    }

    const downloadDxfBtn = $('#download-dxf-btn');
    if (downloadDxfBtn) {
        downloadDxfBtn.addEventListener('click', async () => {
            if (!cadSegments.length) return;
            try {
                const baseName = currentFile ? currentFile.name.replace(/\.[^/.]+$/, '') : 'HATCH.it';
                const res = await fetch('/api/export/dxf/segments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        segments: cadSegments,
                        pattern_name: baseName
                    })
                });
                const json = await res.json();
                if (!res.ok || !json.dxf_base64) throw new Error('Export error');
                const binary = atob(json.dxf_base64);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                const blob = new Blob([bytes], { type: 'application/dxf' });
                const filename = `${baseName}.dxf`;
                await saveFileWithFolderPicker(blob, filename, 'Archivo vectorial AutoCAD DXF (.dxf)', '.dxf');
            } catch (err) {
                showError(t('err.network'), { keepResults: true });
            }
        });
    }

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

    /* ============ FAQ Accordion ============ */
    const faqItems = $$('.faq-item');
    faqItems.forEach((item) => {
        const summary = item.querySelector('summary');
        if (!summary) return;
        summary.addEventListener('click', (e) => {
            if (item.open) {
                e.preventDefault();
                return;
            }
            faqItems.forEach((other) => {
                if (other !== item) other.open = false;
            });
        });
    });
    if (faqItems.length && !faqItems.some((i) => i.open)) {
        faqItems[0].open = true;
    }

    /* ============ Health check ============ */
    fetch('/api/health').catch(() => {});
})();