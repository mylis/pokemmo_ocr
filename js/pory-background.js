(function() {
    "use strict";

    var SYMBOLS = "01<>[]{}()/\\\\+-=*^!?%#@&~POKEMONporygon";
    var NORMAL_IMAGE_PATHS = [
        "./assets/pory1.png",
        "./assets/pory2.png",
        "./assets/pory3.png"
    ];
    var SHINY_IMAGE_PATHS = [
        "./assets/ShinyPory1.png",
        "./assets/ShinyPory2.png",
        "./assets/ShinyPory3.png"
    ];
    var SHINY_IMAGE_CHANCE = 0.20;
    var SYMBOL_SEQUENCES = [
        ["0", "1", "0", "1"],
        ["<", ">", "<", ">"],
        ["[", "]", "{", "}"],
        ["/", "\\", "-", "="],
        ["P", "O", "R", "Y"],
        ["g", "o", "n", "0"]
    ];
    var LAYERS = [
        { color: "rgba(34, 211, 238, 0.95)", speed: 1.15, size: 1.45 },
        { color: "rgba(217, 70, 239, 0.9)", speed: 1.05, size: 1.25 },
        { color: "rgba(34, 211, 238, 0.48)", speed: 0.68, size: 1.0 },
        { color: "rgba(217, 70, 239, 0.4)", speed: 0.6, size: 0.95 },
        { color: "rgba(34, 211, 238, 0.2)", speed: 0.42, size: 0.8 },
        { color: "rgba(217, 70, 239, 0.18)", speed: 0.36, size: 0.78 }
    ];

    var canvas;
    var ctx;
    var particles = [];
    var width = 0;
    var height = 0;
    var animationId = 0;
    var running = false;
    var initialized = false;
    var reduceMotion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    var loadedNormalImages = [];
    var loadedShinyImages = [];

    function particleCount() {
        return Math.max(30, Math.floor(Math.max(window.innerWidth, window.innerHeight) / 16));
    }

    function imageFrequency() {
        return window.innerWidth < 768 ? 0.03 : 0.06;
    }

    function loadImageSet(paths) {
        return Promise.all(paths.map(function(path) {
            return new Promise(function(resolve) {
                var img = new Image();
                img.onload = function() { resolve(img); };
                img.onerror = function() { resolve(null); };
                img.src = path;
            });
        })).then(function(images) {
            return images.filter(Boolean);
        });
    }

    function loadImages() {
        return Promise.all([
            loadImageSet(NORMAL_IMAGE_PATHS),
            loadImageSet(SHINY_IMAGE_PATHS)
        ]).then(function(results) {
            loadedNormalImages = results[0];
            loadedShinyImages = results[1];
        });
    }

    function pickImage() {
        var useShiny = loadedShinyImages.length > 0 && Math.random() < SHINY_IMAGE_CHANCE;
        var pool = useShiny ? loadedShinyImages : loadedNormalImages;
        if (!pool.length) {
            pool = loadedNormalImages.length ? loadedNormalImages : loadedShinyImages;
        }
        if (!pool.length) return null;
        return pool[Math.floor(Math.random() * pool.length)];
    }

    function Particle(layerIndex) {
        this.layer = LAYERS[layerIndex];
        this.isImage = (loadedNormalImages.length > 0 || loadedShinyImages.length > 0) && Math.random() < imageFrequency();
        this.image = this.isImage ? pickImage() : null;
        this.sequence = SYMBOL_SEQUENCES[Math.floor(Math.random() * SYMBOL_SEQUENCES.length)];
        this.sequenceIndex = Math.floor(Math.random() * this.sequence.length);
        this.staticChar = this.sequence[this.sequenceIndex] || SYMBOLS.charAt(Math.floor(Math.random() * SYMBOLS.length));
        this.char = this.staticChar;
        this.size = this.isImage ? (20 + Math.random() * 64) : (10 + Math.random() * 6);
        this.phaseTicks = 0;
        this.phaseDelay = 10 + Math.floor(Math.random() * 20);
        this.reset(true);
    }

    Particle.prototype.reset = function(initial) {
        this.x = Math.random() * width;
        this.y = initial ? Math.random() * height : (-80 - Math.random() * 120);
        this.speed = 0.18 + Math.random() * 0.42;
        this.sequenceIndex = Math.floor(Math.random() * this.sequence.length);
        this.staticChar = this.sequence[this.sequenceIndex];
        this.char = this.staticChar;
        this.phaseTicks = 0;
        this.phaseDelay = 12 + Math.floor(Math.random() * 24);
    };

    Particle.prototype.update = function() {
        this.y += this.speed * this.layer.speed;
        if (this.y > height + 120) this.reset(false);

        if (this.isImage) return;

        this.phaseTicks++;
        if (this.phaseTicks >= this.phaseDelay) {
            this.phaseTicks = 0;
            this.sequenceIndex = (this.sequenceIndex + 1) % this.sequence.length;
            this.char = this.sequence[this.sequenceIndex];
        }
    };

    Particle.prototype.draw = function() {
        if (this.isImage && this.image) {
            ctx.globalAlpha = 0.14 + (this.layer.size * 0.08);
            ctx.drawImage(this.image, this.x, this.y, this.size, this.size);
            ctx.globalAlpha = 1;
            return;
        }

        ctx.fillStyle = this.layer.color;
        ctx.font = (this.size * this.layer.size) + "px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        ctx.fillText(this.char, this.x, this.y);
    };

    function createCanvas() {
        canvas = document.createElement("canvas");
        canvas.id = "poryBackground";
        canvas.setAttribute("aria-hidden", "true");
        document.body.prepend(canvas);
        ctx = canvas.getContext("2d");
        resize();
    }

    function resize() {
        if (!canvas) return;
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        createParticles();
    }

    function createParticles() {
        var count = particleCount();
        particles = [];
        for (var i = 0; i < count; i++) {
            particles.push(new Particle(Math.floor(Math.random() * LAYERS.length)));
        }
    }

    function drawBackdrop() {
        var gradient = ctx.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, "rgba(2, 6, 23, 0.98)");
        gradient.addColorStop(0.55, "rgba(15, 23, 42, 0.95)");
        gradient.addColorStop(1, "rgba(7, 10, 24, 0.98)");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        var cyanGlow = ctx.createRadialGradient(
            width * 0.5,
            height * 0.28,
            0,
            width * 0.5,
            height * 0.28,
            Math.max(width, height) * 0.42
        );
        cyanGlow.addColorStop(0, "rgba(34, 211, 238, 0.08)");
        cyanGlow.addColorStop(0.45, "rgba(34, 211, 238, 0.035)");
        cyanGlow.addColorStop(1, "rgba(34, 211, 238, 0)");
        ctx.fillStyle = cyanGlow;
        ctx.fillRect(0, 0, width, height);

        var magentaGlow = ctx.createRadialGradient(
            width * 0.5,
            height,
            0,
            width * 0.5,
            height,
            Math.max(width, height) * 0.5
        );
        magentaGlow.addColorStop(0, "rgba(217, 70, 239, 0.06)");
        magentaGlow.addColorStop(0.4, "rgba(217, 70, 239, 0.03)");
        magentaGlow.addColorStop(1, "rgba(217, 70, 239, 0)");
        ctx.fillStyle = magentaGlow;
        ctx.fillRect(0, 0, width, height);
    }

    function render() {
        if (!running || !ctx) return;

        drawBackdrop();

        for (var i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();
        }

        animationId = window.requestAnimationFrame(render);
    }

    function stop() {
        running = false;
        if (animationId) {
            window.cancelAnimationFrame(animationId);
            animationId = 0;
        }
    }

    function start() {
        if (!canvas || running) return;
        running = true;
        render();
    }

    function handleVisibility() {
        if (document.hidden) stop();
        else if (!reduceMotion || !reduceMotion.matches) start();
    }

    function setup() {
        if (initialized) return Promise.resolve();
        initialized = true;

        return loadImages().then(function() {
            createCanvas();
            drawBackdrop();

            if (!reduceMotion || !reduceMotion.matches) start();

            window.addEventListener("resize", resize);
            document.addEventListener("visibilitychange", handleVisibility);

            if (reduceMotion && reduceMotion.addEventListener) {
                reduceMotion.addEventListener("change", function(event) {
                    if (event.matches) {
                        stop();
                        drawBackdrop();
                    } else {
                        start();
                    }
                });
            }
        });
    }

    window.PoryBackground = {
        setup: setup,
        start: start,
        stop: stop
    };
})();
