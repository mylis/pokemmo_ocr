/* video-frame-tool.js
   ------------------------------------------------------------
   Always extracts BOTH:
   - Whole frame PNGs (no ROI box)
   - ROI crop PNGs (no ROI box)
   Provides:
   - Download ZIP (Whole)
   - Download ZIP (ROI)
   ROI box is only drawn on preview canvas.
   ------------------------------------------------------------ */

(function() {
    "use strict";

    var VideoFrameTool = (function() {
        // Preview canvas (shows ROI box)
        var previewCanvas = null;
        var previewCtx = null;

        // Offscreen frame canvas (NEVER draws ROI box; used for saving)
        var frameCanvas = null;
        var frameCtx = null;

        // Video element
        var video = null;

        // ROI tracking
        var prevROI = null;
        var roi = { x: 100, y: 100, w: 200, h: 80 };
        var isDragging = false;
        var startX = 0;
        var startY = 0;

        // Extracted results
        var extractedWholeFiles = []; // Array<File>
        var extractedRoiFiles = []; // Array<File>
        var zipWhole = null; // JSZip
        var zipRoi = null; // JSZip

        // Hooks
        var onFramesToOcr = null;
        var onFramesToFirestore = null;
        var debugEnabled = false;

        // Cached elements
        var el = {
            slider: null,
            thresholdValue: null,
            videoInputModal: null,

            extractBtn: null,
            sendBtn: null,
            sendFirestoreBtn: null,
            sendRoiFirestoreBtn: null,

            downloadWholeZipBtn: null,
            downloadRoiZipBtn: null,

            selectWholeFrameBtn: null,

            progress: null,
            count: null,
            status: null
        };

        function init(opts) {
            opts = opts || {};
            onFramesToOcr = opts.onToOcr || null;
            onFramesToFirestore = opts.onToFirestore || null;

            debugEnabled = isDebugEnabledFromQuery();

            previewCanvas = document.getElementById("hiddenCanvas");
            if (!previewCanvas) return;

            previewCtx = previewCanvas.getContext("2d", { willReadFrequently: true });

            // Offscreen (saving)
            frameCanvas = document.createElement("canvas");
            frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

            // Video
            video = document.createElement("video");
            video.muted = true;
            video.playsInline = true;

            // Cache elements
            el.slider = document.getElementById("diffThreshold");
            el.thresholdValue = document.getElementById("thresholdValue");
            el.videoInputModal = document.getElementById("videoInputModal");

            el.extractBtn = document.getElementById("extractFramesBtn");
            el.sendWholeBtn = document.getElementById("sendWholeToOcrBtn");
            el.sendRoiBtn = document.getElementById("sendRoiToOcrBtn");

            el.sendFirestoreBtn = document.getElementById("sendToFirestoreBtn");
            el.sendRoiFirestoreBtn = document.getElementById("sendRoiToFirestoreBtn");

            el.downloadWholeZipBtn = document.getElementById("downloadWholeZipBtn");
            el.downloadRoiZipBtn = document.getElementById("downloadRoiZipBtn");

            el.selectWholeFrameBtn = document.getElementById("selectWholeFrameBtn");

            el.progress = document.getElementById("progressWindow");
            el.count = document.getElementById("savedCountLabel");
            el.status = document.getElementById("frameStatus");

            // Threshold UI
            if (el.slider && el.thresholdValue) {
                el.thresholdValue.textContent = el.slider.value;
                el.slider.addEventListener("input", function() {
                    el.thresholdValue.textContent = el.slider.value;
                });
            }

            // ROI selection handlers on preview canvas
            setupROISelector();

            // Modal upload input
            if (el.videoInputModal) {
                el.videoInputModal.addEventListener("change", handleVideoLoadFromModal);
            }

            // Buttons
            if (el.extractBtn) el.extractBtn.addEventListener("click", extractFrames);
            if (el.sendWholeBtn) el.sendWholeBtn.addEventListener("click", sendWholeFramesToOcr);
            if (el.sendRoiBtn) el.sendRoiBtn.addEventListener("click", sendRoiFramesToOcr);

            if (el.downloadWholeZipBtn) el.downloadWholeZipBtn.addEventListener("click", downloadWholeZip);
            if (el.downloadRoiZipBtn) el.downloadRoiZipBtn.addEventListener("click", downloadRoiZip);

            // Debug-only Firestore send
            if (debugEnabled) {
                if (el.sendFirestoreBtn) {
                    el.sendFirestoreBtn.classList.remove("hidden");
                    el.sendFirestoreBtn.addEventListener("click", sendWholeFramesToFirestore);
                }
                if (el.sendRoiFirestoreBtn) {
                    el.sendRoiFirestoreBtn.classList.remove("hidden");
                    el.sendRoiFirestoreBtn.addEventListener("click", sendRoiFramesToFirestore);
                }
            } else {
                if (el.sendFirestoreBtn) el.sendFirestoreBtn.classList.add("hidden");
                if (el.sendRoiFirestoreBtn) el.sendRoiFirestoreBtn.classList.add("hidden");
            }


            // Select whole frame
            if (el.selectWholeFrameBtn) {
                el.selectWholeFrameBtn.addEventListener("click", function() {
                    selectWholeFrame();
                    drawPreview();
                    setStatus("ROI set to whole frame.");
                });
            }

            setActionEnabled(false);
            setCount(0);
            setStatus("");
        }

        // -------------------------
        // Debug flag
        // -------------------------
        function isDebugEnabledFromQuery() {
            try {
                var sp = new URLSearchParams(window.location.search);
                var v = String(sp.get("debug") || "").toLowerCase();
                return v === "1" || v === "true" || v === "yes" || v === "on";
            } catch {
                return false;
            }
        }

        // -------------------------
        // UI helpers
        // -------------------------
        function setStatus(msg) {
            if (el.status) el.status.textContent = msg || "";
        }

        function setCount(n) {
            if (el.count) el.count.textContent = String(n || 0);
        }

        function clearProgress() {
            if (el.progress) el.progress.textContent = "";
        }

        function appendProgressLine(text) {
            if (!el.progress) return;
            var div = document.createElement("div");
            div.textContent = text;
            el.progress.appendChild(div);
            el.progress.scrollTop = el.progress.scrollHeight;
        }

        function getThreshold() {
            return el.slider ? +el.slider.value : 3;
        }

        function setActionEnabled(enabled) {
            if (el.sendWholeBtn) el.sendWholeBtn.disabled = !enabled;
            if (el.sendRoiBtn) el.sendRoiBtn.disabled = !enabled;

            if (el.downloadWholeZipBtn) el.downloadWholeZipBtn.disabled = !enabled;
            if (el.downloadRoiZipBtn) el.downloadRoiZipBtn.disabled = !enabled;

            if (el.sendFirestoreBtn && debugEnabled)
                el.sendFirestoreBtn.disabled = !enabled;
            if (el.sendRoiFirestoreBtn && debugEnabled)
                el.sendRoiFirestoreBtn.disabled = !enabled;
        }


        // -------------------------
        // Video loading
        // -------------------------
        function handleVideoLoadFromModal(e) {
            var file = e.target && e.target.files && e.target.files[0];
            if (!file) return;

            resetExtraction();

            video.src = URL.createObjectURL(file);
            video.load();

            video.onloadedmetadata = function() {
                // Size both canvases
                previewCanvas.width = video.videoWidth;
                previewCanvas.height = video.videoHeight;

                frameCanvas.width = video.videoWidth;
                frameCanvas.height = video.videoHeight;

                clampRoiToCanvas();

                video.currentTime = 0;
            };

            video.onseeked = function() {
                // Draw a clean frame into offscreen, then preview from that
                drawFrameToOffscreen();
                drawPreview();
                setStatus("Video loaded. Drag to select region, then click “Extract frames”.");
            };
        }

        function resetExtraction() {
            extractedWholeFiles = [];
            extractedRoiFiles = [];
            zipWhole = null;
            zipRoi = null;
            prevROI = null;

            setActionEnabled(false);
            clearProgress();
            setCount(0);
            setStatus("");
        }

        // -------------------------
        // ROI selection
        // -------------------------
        function setupROISelector() {
            function getScaledCoords(clientX, clientY) {
                var rect = previewCanvas.getBoundingClientRect();
                var scaleX = previewCanvas.width / rect.width;
                var scaleY = previewCanvas.height / rect.height;
                return {
                    x: (clientX - rect.left) * scaleX,
                    y: (clientY - rect.top) * scaleY
                };
            }

            function startDragAt(clientX, clientY) {
                isDragging = true;
                var pos = getScaledCoords(clientX, clientY);
                startX = pos.x;
                startY = pos.y;
            }

            function moveDragTo(clientX, clientY) {
                if (!isDragging) return;
                var pos = getScaledCoords(clientX, clientY);

                roi.x = Math.min(startX, pos.x);
                roi.y = Math.min(startY, pos.y);
                roi.w = Math.abs(pos.x - startX);
                roi.h = Math.abs(pos.y - startY);

                clampRoiToCanvas();
                drawPreview();
            }

            function endDrag() {
                isDragging = false;
            }

            previewCanvas.addEventListener("mousedown", function(e) { startDragAt(e.clientX, e.clientY); });
            previewCanvas.addEventListener("mousemove", function(e) { moveDragTo(e.clientX, e.clientY); });
            previewCanvas.addEventListener("mouseup", endDrag);
            previewCanvas.addEventListener("mouseleave", endDrag);

            previewCanvas.addEventListener("touchstart", function(e) {
                if (e.touches.length === 1) {
                    var t = e.touches[0];
                    startDragAt(t.clientX, t.clientY);
                    e.preventDefault();
                }
            }, { passive: false });

            previewCanvas.addEventListener("touchmove", function(e) {
                if (e.touches.length === 1) {
                    var t = e.touches[0];
                    moveDragTo(t.clientX, t.clientY);
                    e.preventDefault();
                }
            }, { passive: false });

            previewCanvas.addEventListener("touchend", endDrag, { passive: false });
            previewCanvas.addEventListener("touchcancel", endDrag, { passive: false });
        }

        function clampRoiToCanvas() {
            if (!previewCanvas) return;
            roi.x = Math.max(0, Math.min(roi.x, previewCanvas.width));
            roi.y = Math.max(0, Math.min(roi.y, previewCanvas.height));
            roi.w = Math.max(0, Math.min(roi.w, previewCanvas.width - roi.x));
            roi.h = Math.max(0, Math.min(roi.h, previewCanvas.height - roi.y));
        }

        function selectWholeFrame() {
            roi.x = 0;
            roi.y = 0;
            roi.w = previewCanvas.width;
            roi.h = previewCanvas.height;
        }

        // -------------------------
        // Drawing (important!)
        // -------------------------
        function drawFrameToOffscreen() {
            // Clean frame ONLY
            frameCtx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
            frameCtx.drawImage(video, 0, 0);
        }

        function drawPreview() {
            // Preview shows clean frame + ROI box
            previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
            previewCtx.drawImage(frameCanvas, 0, 0);

            if (roi.w > 2 && roi.h > 2) {
                previewCtx.strokeStyle = "red";
                previewCtx.lineWidth = 3;
                previewCtx.strokeRect(roi.x, roi.y, roi.w, roi.h);
            }
        }

        // -------------------------
        // Frame processing helpers
        // -------------------------
        function seekTo(time) {
            return new Promise(function(resolve) {
                video.currentTime = time;
                video.onseeked = function() { resolve(); };
            });
        }

        function preprocess(data) {
            var out = new Uint8Array(data.length / 4);
            for (var i = 0, j = 0; i < data.length; i += 4, j++) {
                var g = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                out[j] = g > 140 ? 255 : 0;
            }
            return out;
        }

        function compareROI(imageData) {
            var curr = preprocess(imageData.data);

            if (!prevROI) {
                prevROI = curr;
                return 999;
            }

            var diff = 0;
            for (var i = 0; i < curr.length; i++) diff += Math.abs(curr[i] - prevROI[i]);

            var avg = diff / curr.length;
            prevROI = curr;
            return avg;
        }

        function canvasToPngBlob(srcCanvas) {
            return new Promise(function(resolve) {
                srcCanvas.toBlob(function(blob) { resolve(blob); }, "image/png", 1.0);
            });
        }

        function canvasToJpegBlob(srcCanvas) {
            return new Promise(function(resolve) {
                srcCanvas.toBlob(function(blob) { resolve(blob); }, "image/jpeg", 0.92);
            });
        }

        function makeRoiCropCanvasFromFrame() {
            var c = document.createElement("canvas");
            c.width = Math.max(1, Math.floor(roi.w));
            c.height = Math.max(1, Math.floor(roi.h));
            var cctx = c.getContext("2d");

            // Crop from CLEAN frame canvas (no box)
            cctx.drawImage(
                frameCanvas,
                roi.x, roi.y, roi.w, roi.h,
                0, 0, c.width, c.height
            );
            return c;
        }

        async function getVideoFpsOr60(videoEl) {
            // Ensure metadata
            if (!videoEl.duration || !isFinite(videoEl.duration)) {
                await new Promise(resolve => {
                    videoEl.onloadedmetadata = resolve;
                });
            }

            // Best: use requestVideoFrameCallback to measure mediaTime deltas
            if (typeof videoEl.requestVideoFrameCallback === "function") {
                // Start from beginning for consistent deltas
                try {
                    videoEl.pause();
                    videoEl.currentTime = 0;
                } catch {}

                // Collect a small sample of mediaTime deltas
                const deltas = [];
                let lastT = null;
                let lastPresented = -1;

                // We only need ~20-40 deltas for a good estimate
                const targetDeltas = 30;

                await new Promise(resolve => {
                    let done = false;

                    function finish() {
                        if (done) return;
                        done = true;
                        try { videoEl.pause(); } catch {}
                        resolve();
                    }

                    // Play so frames advance
                    const p = videoEl.play();
                    if (p && typeof p.catch === "function") p.catch(() => { /* ignore */ });

                    function cb(now, meta) {
                        if (done) return;

                        // Skip duplicate callbacks for same frame
                        if (meta && typeof meta.presentedFrames === "number") {
                            if (meta.presentedFrames === lastPresented) {
                                videoEl.requestVideoFrameCallback(cb);
                                return;
                            }
                            lastPresented = meta.presentedFrames;
                        }

                        const t = (meta && typeof meta.mediaTime === "number") ?
                            meta.mediaTime :
                            videoEl.currentTime;

                        if (lastT != null) {
                            const dt = t - lastT;
                            // filter out weird / zero / huge jumps
                            if (dt > 0.001 && dt < 0.2) deltas.push(dt);
                        }
                        lastT = t;

                        if (deltas.length >= targetDeltas || videoEl.ended || videoEl.currentTime >= videoEl.duration) {
                            finish();
                            return;
                        }

                        videoEl.requestVideoFrameCallback(cb);
                    }

                    videoEl.requestVideoFrameCallback(cb);
                });

                if (deltas.length >= 5) {
                    // Median delta is more robust than average
                    deltas.sort((a, b) => a - b);
                    const mid = Math.floor(deltas.length / 2);
                    const medianDt = deltas.length % 2 ? deltas[mid] : (deltas[mid - 1] + deltas[mid]) / 2;

                    const estFps = 1 / medianDt;

                    // Snap to common FPS values
                    return snapToCommonFps(estFps);
                }
            }

            // Fallback when RVFC isn't available
            return 60;

            function snapToCommonFps(v) {
                const common = [24, 25, 29.97, 30, 50, 59.94, 60, 120, 144, 240];
                let best = common[0];
                let bestDiff = Infinity;

                for (const c of common) {
                    const diff = Math.abs(v - c);
                    if (diff < bestDiff) {
                        bestDiff = diff;
                        best = c;
                    }
                }

                // If it's close enough to a common value, use it; else round sensibly
                if (bestDiff <= 1.0) return best === 29.97 || best === 59.94 ? best : Math.round(best);
                return Math.max(1, Math.min(Math.round(v), 240));
            }
        }



        // -------------------------
        // Extract BOTH
        // -------------------------
        async function extractFrames() {
            if (!video || !video.src) {
                alert("Upload a video inside the modal first.");
                return;
            }
            if (roi.w < 5 || roi.h < 5) {
                alert("Select a region by dragging on the preview first.");
                return;
            }

            resetExtraction();

            zipWhole = window.JSZip ? new window.JSZip() : null;
            zipRoi = window.JSZip ? new window.JSZip() : null;

            setStatus("Extracting frames…");

            var fps = await getVideoFpsOr60(video);
            var frameDuration = 1 / fps;

            var currentTime = 0;
            var frameNum = 0;
            var savedCount = 0;

            while (currentTime < video.duration) {
                await seekTo(currentTime);

                // Draw clean frame to offscreen
                drawFrameToOffscreen();

                // Preview update (shows ROI box, but this is NOT saved)
                drawPreview();

                // ROI diff check uses clean frame context
                var roiData = frameCtx.getImageData(roi.x, roi.y, roi.w, roi.h);
                var score = compareROI(roiData);

                if (score > getThreshold()) {
                    // WHOLE FRAME (clean)
                    var wholeBlob = await canvasToJpegBlob(frameCanvas);

                    // ROI CROP (clean)
                    var cropCanvas = makeRoiCropCanvasFromFrame();
                    var roiBlob = await canvasToJpegBlob(cropCanvas);

                    var filename = "frame_" + String(savedCount).padStart(5, "0") + ".jpg";

                    var wholeFile = new File([wholeBlob], filename, { type: "image/jpeg" });
                    var roiFile = new File([roiBlob], filename, { type: "image/jpeg" });

                    extractedWholeFiles.push(wholeFile);
                    extractedRoiFiles.push(roiFile);

                    if (zipWhole) zipWhole.file(filename, wholeBlob);
                    if (zipRoi) zipRoi.file(filename, roiBlob);

                    savedCount++;
                    setCount(savedCount);
                }

                appendProgressLine(
                    "Frame " + frameNum + ": " + currentTime.toFixed(2) +
                    "s • Diff " + score.toFixed(2)
                );

                frameNum++;
                currentTime += frameDuration;
            }

            if (savedCount === 0) {
                setStatus("No frames captured. Try lowering the threshold or selecting a different region.");
                setActionEnabled(false);
                return;
            }

            setStatus(
                "Done. Extracted " + savedCount +
                " frames (whole + ROI). Choose OCR or download either ZIP."
            );
            setActionEnabled(true);
        }

        // -------------------------
        // Send Frames
        // -------------------------
        async function sendWholeFramesToOcr() {
            if (!extractedWholeFiles.length) return;

            // fallback to OcrApp if not wired
            if (!onFramesToOcr) {
                if (window.OcrApp && typeof window.OcrApp.processFiles === "function") {
                    setStatus("Sending " + extractedWholeFiles.length + " whole frames to OCR…");
                    await window.OcrApp.processFiles(extractedWholeFiles, "parse");
                    setStatus("Sent " + extractedWholeFiles.length + " whole frames to OCR.");
                    return;
                }
                alert("OCR handler is not wired yet.");
                return;
            }

            setStatus("Sending " + extractedWholeFiles.length + " whole frames to OCR…");
            await onFramesToOcr(extractedWholeFiles);
            setStatus("Sent " + extractedWholeFiles.length + " whole frames to OCR.");
        }

        async function sendWholeFramesToFirestore() {
            if (!debugEnabled) return;
            if (!extractedWholeFiles.length) return;

            if (!onFramesToFirestore) {
                if (window.OcrApp && typeof window.OcrApp.processFiles === "function") {
                    setStatus("Sending " + extractedWholeFiles.length + " whole frames to Firestore…");
                    await window.OcrApp.processFiles(extractedWholeFiles, "firestore");
                    setStatus("Sent " + extractedWholeFiles.length + " whole frames to Firestore.");
                    return;
                }
                alert("Firestore handler is not wired yet.");
                return;
            }

            setStatus("Sending " + extractedWholeFiles.length + " whole frames to Firestore…");
            await onFramesToFirestore(extractedWholeFiles);
            setStatus("Sent " + extractedWholeFiles.length + " whole frames to Firestore.");
        }

        async function sendRoiFramesToOcr() {
            if (!extractedRoiFiles.length) return;

            if (window.OcrApp && typeof window.OcrApp.processFiles === "function") {
                setStatus("Sending " + extractedRoiFiles.length + " ROI crops to OCR…");
                await window.OcrApp.processFiles(extractedRoiFiles, "parse");
                setStatus("Sent " + extractedRoiFiles.length + " ROI crops to OCR.");
                return;
            }

            alert("OCR handler not available.");
        }

        async function sendRoiFramesToFirestore() {
            if (!debugEnabled) return;
            if (!extractedRoiFiles.length) return;

            if (!onFramesToFirestore) {
                if (window.OcrApp && typeof window.OcrApp.processFiles === "function") {
                    setStatus("Sending " + extractedRoiFiles.length + " ROI crops to Firestore…");
                    await window.OcrApp.processFiles(extractedRoiFiles, "firestore");
                    setStatus("Sent " + extractedRoiFiles.length + " ROI crops to Firestore.");
                    return;
                }
                alert("Firestore handler is not wired yet.");
                return;
            }

            setStatus("Sending " + extractedRoiFiles.length + " ROI crops to Firestore…");
            await onFramesToFirestore(extractedRoiFiles);
            setStatus("Sent " + extractedRoiFiles.length + " ROI crops to Firestore.");
        }

        // -------------------------
        // Download ZIPs
        // -------------------------
        async function downloadWholeZip() {
            if (!zipWhole) {
                alert("ZIP not available. Make sure JSZip is loaded and extract frames first.");
                return;
            }
            var blob = await zipWhole.generateAsync({ type: "blob" });
            var link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "unique_frames_whole.zip";
            link.click();
        }

        async function downloadRoiZip() {
            if (!zipRoi) {
                alert("ZIP not available. Make sure JSZip is loaded and extract frames first.");
                return;
            }
            var blob = await zipRoi.generateAsync({ type: "blob" });
            var link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "unique_frames_roi.zip";
            link.click();
        }

        return { init: init };
    })();

    window.VideoFrameTool = VideoFrameTool;
})();