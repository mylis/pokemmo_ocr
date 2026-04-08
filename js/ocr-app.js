/* ocr-app.js
   ------------------------------------------------------------
   Core OCR app logic extracted from inline <script> into a file.
   - Handles: query params, upload/paste/drag-drop, API calls,
     Firestore/debug mode, loading overlay + cancel, rendering
     table/cards, copy buttons, delete, CSV export, OTS toggle.
   - Exposes: window.OcrApp.processFiles(files, mode)
              window.OcrApp.selectionType(files)
              window.OcrApp.clearAll()
              window.OcrApp.setStatus(msg)
   ------------------------------------------------------------ */

(function() {
    "use strict";

    // -------------------------
    // Query string config
    // -------------------------
    function getQueryParam(name) {
        var url = new URL(window.location.href);
        return url.searchParams.get(name);
    }

    // -------------------------
    // Elements
    // -------------------------
    var fileInput = document.getElementById("fileInput");
    var drop = document.getElementById("drop");
    var statusEl = document.getElementById("status");
    var btnClear = document.getElementById("btnClear");
    var btnCsv = document.getElementById("btnCsv");
    var tbl = document.getElementById("tbl");
    var tbody = document.getElementById("tbody");
    var emptyState = document.getElementById("emptyState");
    var btnFirestore = document.getElementById("btnFirestore");
    var resultsWrap = document.getElementById("resultsWrap");
    var jsonWrap = document.getElementById("jsonWrap");
    var loadingOverlay = document.getElementById("loadingOverlay");
    var btnCancel = document.getElementById("btnCancel");

    // Responsive cards/table containers
    var cardsWrap = document.getElementById("cardsWrap");
    var cards = document.getElementById("cards");

    // Copy all bars
    var btnCopyAllPokePasteTable = document.getElementById("btnCopyAllPokePasteTable");
    var btnCopyAllPokePasteMobile = document.getElementById("btnCopyAllPokePasteMobile");
    var mobileCopyAllBar = document.getElementById("mobileCopyAllBar");

    // OTS
    var btnOtsTable = document.getElementById("btnOtsTable");
    var btnOtsMobile = document.getElementById("btnOtsMobile");
    var otsEnabled = false;

    // Loading UI pieces
    var loadingTitle = document.getElementById("loadingTitle");
    var loadingSub = document.getElementById("loadingSub");
    var loadingBar = document.getElementById("loadingBar");

    let hidePokePasteFromCsv = false;

    const btnHideTable = document.getElementById("btnHidePpTable");
    const btnHideMobile = document.getElementById("btnHidePpMobile");

    // -------------------------
    // CSV PokePaste toggle styling
    // -------------------------
    function applyHidePpButtonState(btn) {
        if (!btn) return;

        const disabled = hidePokePasteFromCsv; // true = hidden

        btn.textContent = disabled ? "CSV PokePaste: Off" : "CSV PokePaste: On";

        btn.classList.toggle("bg-emerald-500/15", !disabled);
        btn.classList.toggle("text-emerald-200", !disabled);
    }

    function setHidePpUi() {
        applyHidePpButtonState(btnHideTable);
        applyHidePpButtonState(btnHideMobile);
    }


    function toggleHidePp() {
        hidePokePasteFromCsv = !hidePokePasteFromCsv;
        localStorage.setItem("hideCsvPp", hidePokePasteFromCsv ? "1" : "0");
        setHidePpUi();
    }

    hidePokePasteFromCsv = localStorage.getItem("hideCsvPp") === "1";
    setHidePpUi();
    if (btnHideTable) btnHideTable.addEventListener("click", toggleHidePp);
    if (btnHideMobile) btnHideMobile.addEventListener("click", toggleHidePp);


    // -------------------------
    // State
    // -------------------------
    var cancelRequested = false;

    // keep data between loads until cleared/closed
    var store = {
        parse: [],
        firestore: []
    };

    var lastMode = "parse";
    var _rowSeq = 0;

    function ensureRowIds(rows) {
        rows.forEach(function(r) {
            if (!r) return;
            if (!r._rid) r._rid = "r" + (++_rowSeq);
        });
    }

    function getActiveRows(mode) {
        return (mode === "firestore") ? store.firestore : store.parse;
    }

    function setActiveRows(mode, rows) {
        if (mode === "firestore") store.firestore = rows;
        else store.parse = rows;
    }

    function allRowsCount() {
        return store.parse.length + store.firestore.length;
    }

    // -------------------------
    // Config
    // -------------------------
    var apiBase = (getQueryParam("api") || "https://api.mylis.net").replace(/\/+$/, "");
    var debugEnabled = String(getQueryParam("debug") || "").toLowerCase();
    debugEnabled = (debugEnabled === "1" || debugEnabled === "true" || debugEnabled === "yes" || debugEnabled === "on");

    if (resultsWrap && debugEnabled) resultsWrap.classList.add("show-debug");
    if (!debugEnabled && btnFirestore) {
        btnFirestore.style.display = "none";
    }

    if (!debugEnabled && sendToFirestoreBtn) {
        sendToFirestoreBtn.style.display = "none";
    }

    if (!debugEnabled && sendRoiToFirestoreBtn) {
        sendRoiToFirestoreBtn.style.display = "none";
    }

    // -------------------------
    // Status + clear
    // -------------------------
    function setStatus(msg) {
        if (!statusEl) return;
        statusEl.textContent = msg || "";
    }

    function clearAll() {
        store.parse = [];
        store.firestore = [];

        if (tbody) tbody.innerHTML = "";
        if (cards) cards.innerHTML = "";
        if (jsonWrap) jsonWrap.innerHTML = "";

        if (tbl) tbl.style.display = "none";
        if (cardsWrap) cardsWrap.classList.add("hidden");
        if (jsonWrap) jsonWrap.classList.add("hidden");
        if (emptyState) emptyState.style.display = "";

        if (btnCsv) btnCsv.disabled = true;
        setStatus("Cleared.");
    }

    if (btnClear) btnClear.addEventListener("click", clearAll);

    // -------------------------
    // Delete row
    // -------------------------
    function deleteRowById(mode, rid) {
        var ok = window.confirm("Are you sure you want to delete this pokemon?");
        if (!ok) return;

        var rows = getActiveRows(mode);
        rows = rows.filter(function(r) {
            return r && r._rid !== rid;
        });
        setActiveRows(mode, rows);

        if (mode === "firestore") renderFirestore(rows);
        else renderRows(rows);

        if (btnCsv) btnCsv.disabled = (mode === "firestore") || !(rows && rows.length);

        if (!rows.length && allRowsCount() === 0) setStatus("No results yet.");
        else setStatus("Pokemon deleted.");
    }

    // -------------------------
    // File type helpers
    // -------------------------
    function isVideoFile(f) {
        return f && f.type && f.type.indexOf("video/") === 0;
    }

    function isImageFile(f) {
        return f && f.type && f.type.indexOf("image/") === 0;
    }

    function selectionType(files) {
        if (!files || !files.length) return "empty";
        var hasVid = files.some(isVideoFile);
        var hasImg = files.some(isImageFile);
        if (hasVid && hasImg) return "mixed";
        if (hasVid) return "video";
        return "images";
    }

    function filesFromDrop(e) {
        var dt = e.dataTransfer;
        if (!dt || !dt.files) return [];
        var arr = Array.from(dt.files);

        return arr.filter(function(f) {
            return f.type && (f.type.indexOf("image/") === 0 || f.type.indexOf("video/") === 0);
        });
    }

    function filesFromClipboardEvent(e) {
        var cd = e.clipboardData;
        if (!cd || !cd.items) return [];

        var files = [];
        for (var i = 0; i < cd.items.length; i++) {
            var it = cd.items[i];
            if (it.kind === "file") {
                var f = it.getAsFile();
                if (f) files.push(f);
            }
        }

        files = files.filter(function(f) {
            return f && f.type && (f.type.indexOf("image/") === 0 || f.type.indexOf("video/") === 0);
        });

        return files;
    }

    // -------------------------
    // Clipboard paste support
    // -------------------------
    document.addEventListener("paste", function(e) {
        var files = filesFromClipboardEvent(e);
        if (!files.length) return;

        e.preventDefault();
        var mode = (lastMode === "firestore") ? "firestore" : "parse";

        setStatus("Pasted " + files.length + " item(s) from clipboard. Processing…");
        processFiles(files, mode);
    });

    // -------------------------
    // Dropzone styling + drop
    // -------------------------
    if (drop) {
        drop.addEventListener("dragover", function(e) {
            e.preventDefault();
            drop.classList.add("border-emerald-300/60");
            drop.classList.add("bg-emerald-400/10");
        });

        drop.addEventListener("dragleave", function() {
            drop.classList.remove("border-emerald-300/60");
            drop.classList.remove("bg-emerald-400/10");
        });

        drop.addEventListener("drop", function(e) {
            e.preventDefault();
            drop.classList.remove("border-emerald-300/60");
            drop.classList.remove("bg-emerald-400/10");

            var files = filesFromDrop(e);
            if (!files.length) return;
            processFiles(files, (lastMode === "firestore") ? "firestore" : "parse");
        });
    }

    // -------------------------
    // File input handlers
    // -------------------------
    if (fileInput) {
        fileInput.addEventListener("change", function() {
            var files = Array.from(fileInput.files || []);
            if (!files.length) return;

            var mode = (lastMode === "firestore") ? "firestore" : "parse";
            setStatus("Processing " + files.length + " file(s)…");
            processFiles(files, mode);
        });
    }

    if (btnFirestore) {
        btnFirestore.addEventListener("click", function() {
            var files = Array.from((fileInput && fileInput.files) || []);
            if (!files.length) return setStatus("Select files first.");
            processFiles(files, "firestore");
        });
    }

    // -------------------------
    // Loading overlay + cancel
    // -------------------------
    function showLoadingWithProgress(title, done, total) {
        if (loadingTitle && title) loadingTitle.textContent = title;
        if (loadingSub) loadingSub.textContent = (done || 0) + " / " + (total || 0) + " completed";

        var pct = total ? Math.round((done / total) * 100) : 0;
        if (loadingBar) loadingBar.style.width = pct + "%";

        if (loadingOverlay) {
            loadingOverlay.classList.remove("hidden");
            loadingOverlay.classList.add("flex");
        }

        if (btnClear) btnClear.disabled = true;
        if (btnCsv) btnCsv.disabled = true;

        if (cancelRequested && btnCancel) {
            btnCancel.disabled = true;
            btnCancel.textContent = "Cancelling…";
        }
    }

    function hideLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.add("hidden");
            loadingOverlay.classList.remove("flex");
        }

        if (btnClear) btnClear.disabled = false;

        // CSV only for parse store
        if (btnCsv) btnCsv.disabled = !(store.parse && store.parse.length);

        if (loadingBar) loadingBar.style.width = "0%";
        if (loadingSub) loadingSub.textContent = "0 / 0 completed";

        cancelRequested = false;
        if (btnCancel) {
            btnCancel.disabled = false;
            btnCancel.textContent = "Cancel";
        }
    }

    if (btnCancel) {
        btnCancel.addEventListener("click", function() {
            cancelRequested = true;
            btnCancel.disabled = true;
            btnCancel.textContent = "Cancelling…";
            setStatus("Cancelling… finishing current file.");
        });
    }

    // -------------------------
    // Processing
    // -------------------------
    async function processFiles(files, mode) {
        if (!files || !files.length) return;
        files = Array.from(files).filter(function(f) {
            return isImageFile(f) || isVideoFile(f);
        });
        if (!files.length) return;

        lastMode = mode;
        cancelRequested = false;

        if (btnClear) btnClear.disabled = true;
        if (btnFirestore) btnFirestore.disabled = true;
        if (btnCsv) btnCsv.disabled = true;

        try {
            var total = files.length;
            var done = 0;
            var titleBase = (mode === "firestore") ? "Processing uploads (Firestore)…" : "Processing uploads…";

            showLoadingWithProgress(titleBase, done, total);
            setStatus(titleBase + " " + total + " file(s)…");

            for (var i = 0; i < files.length; i++) {
                if (cancelRequested) break;

                var f = files[i];
                var isVid = isVideoFile(f);
                var endpoint = isVid ?
                    ((mode === "firestore") ? "/parse_video_firestore" : "/parse_video") :
                    ((mode === "firestore") ? "/parse_firestore" : "/parse");
                var kind = isVid ? "video" : "image";
                var currentTitle = (mode === "firestore" ? "Firestore " : "Processing ") + kind + ": " + f.name;
                showLoadingWithProgress(currentTitle, done, total);

                var fd = new FormData();
                if (isVid) fd.append("file", f);
                else fd.append("files", f);

                var resp = await fetch(apiBase + endpoint, { method: "POST", body: fd });
                if (!resp.ok) {
                    var txt = await resp.text();
                    throw new Error("Failed on " + f.name + ": " + resp.status + " " + txt);
                }

                var data = await resp.json();
                var newRows = (data && Array.isArray(data.rows)) ? data.rows : [];

                ensureRowIds(newRows);

                var active = getActiveRows(mode).concat(newRows);
                setActiveRows(mode, active);

                if (mode === "firestore") renderFirestore(active);
                else renderRows(active);

                done++;
                showLoadingWithProgress(titleBase, done, total);
                if (isVid && data && (typeof data.frames_total === "number") && (typeof data.frames_unique === "number")) {
                    setStatus(
                        "Processed " + done + " / " + total + " file(s). " +
                        f.name + ": " + data.frames_unique + " unique / " + data.frames_total + " sampled."
                    );
                } else {
                    setStatus("Processed " + done + " / " + total + " file(s)…");
                }
            }

            if (cancelRequested) setStatus("Cancelled. Processed " + done + " / " + total + " file(s).");
            else setStatus("Done. Processed " + done + " / " + total + " file(s).");

            var active2 = getActiveRows(mode);
            if (btnCsv) btnCsv.disabled = (mode === "firestore") || (active2.length === 0);
        } catch (err) {
            console.error(err);
            setStatus(String(err && err.message ? err.message : err));
        } finally {
            hideLoading();

            if (btnClear) btnClear.disabled = false;
            if (btnFirestore) btnFirestore.disabled = false;

            var active3 = getActiveRows(mode);
            if (btnCsv) {
                if (mode === "firestore") btnCsv.disabled = true;
                else btnCsv.disabled = !(active3 && active3.length);
            }
        }
    }

    // -------------------------
    // Firestore renderer
    // -------------------------
    function escapeHtml(s) {
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function escapeAttr(s) {
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/"/g, "&quot;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    function renderFirestore(items) {
        if (tbl) tbl.style.display = "none";
        if (cardsWrap) cardsWrap.classList.add("hidden");
        if (emptyState) emptyState.style.display = items.length ? "none" : "";

        if (!jsonWrap) return;
        jsonWrap.classList.remove("hidden");
        jsonWrap.innerHTML = "";

        var bundlePretty = JSON.stringify(items, null, 2);

        var bundleCard = document.createElement("div");
        bundleCard.className = "rounded-2xl border border-white/10 bg-slate-900/60 p-4 shadow-lg";

        bundleCard.innerHTML =
            '<div class="flex items-center justify-between gap-3">' +
            '<div class="min-w-0">' +
            '<div class="text-sm font-semibold text-slate-100 truncate">All records (single JSON)</div>' +
            '<div class="text-xs text-slate-400">' + items.length + ' Pokémon</div>' +
            "</div>" +
            '<button class="copyJsonBtn rounded-lg border border-white/10 bg-emerald-500/15 px-3 py-2 text-xs font-semibold text-emerald-200 hover:bg-emerald-500/25" ' +
            'data-json="' + escapeAttr(bundlePretty) + '">' +
            "Copy ALL JSON" +
            "</button>" +
            "</div>" +
            '<div class="mt-3 rounded-lg border border-white/10 bg-black/30 p-3 overflow-auto max-h-[320px]">' +
            '<pre class="font-mono whitespace-pre text-[11px] leading-snug text-slate-100/90">' +
            escapeHtml(bundlePretty) +
            "</pre>" +
            "</div>";

        jsonWrap.appendChild(bundleCard);

        items.forEach(function(obj, idx) {
            var species = (obj && obj.species) ? obj.species : "";
            var nickname = (obj && obj.nickname) ? obj.nickname : "";
            var title = (nickname ? (nickname + " (" + species + ")") : species) || ("Pokémon " + (idx + 1));
            var pretty = JSON.stringify(obj, null, 2);

            var card = document.createElement("div");
            card.className = "rounded-2xl border border-white/10 bg-slate-900/40 p-4 shadow-lg";

            card.innerHTML =
                '<div class="flex items-center justify-between gap-3">' +
                '<div class="min-w-0">' +
                '<div class="text-sm font-semibold text-slate-100 truncate">' + escapeHtml(title) + "</div>" +
                '<div class="text-xs text-slate-400">Firestore JSON</div>' +
                "</div>" +
                '<button class="copyJsonBtn rounded-lg border border-white/10 bg-indigo-500/15 px-3 py-2 text-xs font-semibold text-indigo-200 hover:bg-indigo-500/25" ' +
                'data-json="' + escapeAttr(pretty) + '">' +
                "Copy JSON" +
                "</button>" +
                "</div>" +
                '<div class="mt-3 rounded-lg border border-white/10 bg-black/30 p-3 overflow-auto">' +
                '<pre class="font-mono whitespace-pre text-[11px] leading-snug text-slate-100/90">' +
                escapeHtml(pretty) +
                "</pre>" +
                "</div>";

            jsonWrap.appendChild(card);
        });
    }

    if (jsonWrap) {
        jsonWrap.addEventListener("click", function(e) {
            var btn = e.target.closest(".copyJsonBtn");
            if (!btn) return;

            var txt = btn.getAttribute("data-json") || "";
            navigator.clipboard.writeText(txt).then(function() {
                btn.classList.add("ring-1", "ring-emerald-400/60", "bg-emerald-500/15");
                setTimeout(function() {
                    btn.classList.remove("ring-1", "ring-emerald-400/60", "bg-emerald-500/15");
                }, 700);
                setStatus("JSON copied to clipboard.");
            }).catch(function() {
                setStatus("Could not copy JSON.");
            });
        });
    }

    // -------------------------
    // Responsive layout switch
    // -------------------------
    function isMobile() {
        return window.matchMedia("(max-width: 1270px)").matches;
    }

    var _lastLayoutMobile = isMobile();
    window.addEventListener("resize", function() {
        var now = isMobile();
        if (now !== _lastLayoutMobile) {
            _lastLayoutMobile = now;
            var rows = getActiveRows(lastMode) || [];
            if (lastMode === "firestore") renderFirestore(rows);
            else renderRows(rows);
        }
    });

    // -------------------------
    // Debug formatter
    // -------------------------
    function formatDebug(r) {
        if (!r || !r.debug) return "";

        var left = [];
        var right = [];

        if (r.debug.conf) {
            left.push("CONFIDENCE");
            Object.keys(r.debug.conf).forEach(function(k) {
                var v = r.debug.conf[k];
                if (typeof v === "number" && v > 0) {
                    left.push(k.padEnd(10, " ") + ": " + (v * 100).toFixed(1) + "%");
                }
            });
        }

        if (r.debug.gender_detection) {
            var gd = r.debug.gender_detection;
            left.push("");
            left.push("GENDER");
            left.push("result".padEnd(10, " ") + ": " + String(gd.result || r.gender || "unknown"));
            if (gd.level_found != null) {
                left.push("level_found".padEnd(10, " ") + ": " + (gd.level_found ? "yes" : "no"));
            }
            if (gd.row_token_count != null) {
                left.push("row_tokens".padEnd(10, " ") + ": " + String(gd.row_token_count));
            }
            if (gd.name_anchor_right != null) {
                left.push("anchor_x".padEnd(10, " ") + ": " + String(gd.name_anchor_right));
            }
            if (gd.anchor_source) {
                left.push("source".padEnd(10, " ") + ": " + String(gd.anchor_source));
            }
            if (gd.gender_ratio != null) {
                left.push("ratio".padEnd(10, " ") + ": " + String(gd.gender_ratio));
            }
            if (gd.pixel_anchor_right != null) {
                left.push("pixel_x".padEnd(10, " ") + ": " + String(gd.pixel_anchor_right));
            }
            if (gd.row_refine && gd.row_refine.refined_row_y != null) {
                left.push("row_y".padEnd(10, " ") + ": " + String(gd.row_refine.refined_row_y));
            }
            if (gd.primary_scores) {
                left.push("primary".padEnd(10, " ") + ": " + "M " + (gd.primary_scores.male || 0) + " / F " + (gd.primary_scores.female || 0));
            }
            if (gd.used_fallback || gd.roi_fallback) {
                left.push("fallback".padEnd(10, " ") + ": " + "M " + (((gd.fallback_scores || {}).male) || 0) + " / F " + (((gd.fallback_scores || {}).female) || 0));
            }
            if (gd.symbol_search && gd.symbol_search.picked_label) {
                left.push("symbol".padEnd(10, " ") + ": " + gd.symbol_search.picked_label);
            }
            if (gd.ocr && Array.isArray(gd.ocr.texts)) {
                var ocrTexts = gd.ocr.texts.map(function(t) {
                    return (t.text || "?") + " " + (((t.conf || 0) * 100).toFixed(1)) + "%";
                }).join(", ");
                left.push("ocr".padEnd(10, " ") + ": " + (ocrTexts || "none"));
            }
            if (gd.fixed_roi) {
                left.push("fixed".padEnd(10, " ") + ": " + "M " + (gd.fixed_roi.male || 0) + " / F " + (gd.fixed_roi.female || 0));
                if (gd.fixed_roi.candidate_count != null) {
                    left.push("cand".padEnd(10, " ") + ": " + String(gd.fixed_roi.candidate_count));
                }
            }
        }

        if (Array.isArray(r.debug.move_candidates) && r.debug.move_candidates.length) {
            right.push("MOVE CANDIDATES");
            r.debug.move_candidates.forEach(function(mc) {
                var name = mc[0] || "";
                var conf = mc[1];
                right.push(name.padEnd(20, " ") + " " + (conf * 100).toFixed(1) + "%");
            });
        }

        if (r.debug.gender_detection) {
            var gd2 = r.debug.gender_detection;
            right.push("");
            right.push("GENDER ROI");
            if (Array.isArray(gd2.roi_primary)) {
                right.push("primary".padEnd(20, " ") + gd2.roi_primary.join(","));
            }
            if (Array.isArray(gd2.roi_fallback)) {
                right.push("fallback".padEnd(20, " ") + gd2.roi_fallback.join(","));
            }
            if (gd2.ocr && Array.isArray(gd2.ocr.crop)) {
                right.push("ocr_crop".padEnd(20, " ") + gd2.ocr.crop.join(","));
            }
            if (gd2.symbol_search && Array.isArray(gd2.symbol_search.search)) {
                right.push("symbol_search".padEnd(20, " ") + gd2.symbol_search.search.join(","));
            }
            if (gd2.symbol_search && Array.isArray(gd2.symbol_search.picked)) {
                right.push("symbol_box".padEnd(20, " ") + gd2.symbol_search.picked.join(","));
            }
            if (gd2.row_refine && Array.isArray(gd2.row_refine.search)) {
                right.push("row_refine".padEnd(20, " ") + gd2.row_refine.search.join(","));
            }
            if (gd2.fixed_roi && Array.isArray(gd2.fixed_roi.roi)) {
                right.push("fixed_roi".padEnd(20, " ") + gd2.fixed_roi.roi.join(","));
            }
            if (gd2.level_right != null) {
                right.push("level_right".padEnd(20, " ") + String(gd2.level_right));
            }
            if (gd2.row_right != null) {
                right.push("row_right".padEnd(20, " ") + String(gd2.row_right));
            }
        }

        var maxLines = Math.max(left.length, right.length);
        var out = [];
        var colWidth = 28;

        for (var i = 0; i < maxLines; i++) {
            var l = left[i] || "";
            var rr = right[i] || "";
            out.push(l.padEnd(colWidth, " ") + "  " + rr);
        }

        return out.join("\n");
    }

    // -------------------------
    // Copy all pokepaste
    // -------------------------
    function applyOtsButtonState(btn) {
        if (!btn) return;
        btn.textContent = otsEnabled ? "OTS: On" : "OTS: Off";
        btn.classList.toggle("bg-emerald-500/15", otsEnabled);
        btn.classList.toggle("text-emerald-200", otsEnabled);
    }

    function setOtsUi() {
        applyOtsButtonState(btnOtsTable);
        applyOtsButtonState(btnOtsMobile);
    }

    function toggleOts() {
        otsEnabled = !otsEnabled;
        setOtsUi();
        var rows = getActiveRows(lastMode) || [];
        if (lastMode === "firestore") renderFirestore(rows);
        else renderRows(rows);
    }

    if (btnOtsTable) btnOtsTable.addEventListener("click", toggleOts);
    if (btnOtsMobile) btnOtsMobile.addEventListener("click", toggleOts);
    setOtsUi();

    async function copyAllPokePaste() {
        if (lastMode === "firestore") return;
        var rows = store.parse;
        if (!rows || !rows.length) return setStatus("No rows to copy.");

        var blocks = rows.map(function(r) { return buildPokePaste(r); });
        var allText = blocks.join("\n\n");

        try {
            await navigator.clipboard.writeText(allText);
            setStatus("Copied ALL PokePaste (" + rows.length + " Pokémon).");
        } catch {
            setStatus("Could not copy all PokePaste.");
        }
    }

    if (btnCopyAllPokePasteTable) btnCopyAllPokePasteTable.addEventListener("click", copyAllPokePaste);
    if (btnCopyAllPokePasteMobile) btnCopyAllPokePasteMobile.addEventListener("click", copyAllPokePaste);

    function updateCopyAllButtons(rows) {
        var show = (lastMode !== "firestore") && rows && rows.length;

        if (btnCopyAllPokePasteTable) btnCopyAllPokePasteTable.classList.toggle("hidden", !show);
        if (btnOtsTable) btnOtsTable.classList.toggle("hidden", !show);

        if (mobileCopyAllBar) mobileCopyAllBar.classList.toggle("hidden", !show || !isMobile());

        setOtsUi();
    }

    // -------------------------
    // Renderer entry
    // -------------------------
    function renderRows(rows) {
        if (tbody) tbody.innerHTML = "";
        if (cards) cards.innerHTML = "";

        if (!rows || !rows.length) {
            if (tbl) tbl.style.display = "none";
            if (cardsWrap) cardsWrap.classList.add("hidden");
            if (emptyState) emptyState.style.display = "";
            return;
        }

        if (emptyState) emptyState.style.display = "none";

        if (isMobile()) {
            if (tbl) tbl.style.display = "none";
            if (cardsWrap) cardsWrap.classList.remove("hidden");
            renderCards(rows);
        } else {
            if (cardsWrap) cardsWrap.classList.add("hidden");
            if (tbl) tbl.style.display = "";
            renderTableRows(rows);
        }
        updateCopyAllButtons(rows);
    }

    // -------------------------
    // Sprites helpers
    // -------------------------
    function pokemonDbSlug(name) {
        if (!name) return "";
        var n = String(name).trim();
        n = n.replace(/\u2019/g, "'");
        n = n.replace(/\u2640/g, "♀").replace(/\u2642/g, "♂");

        var map = {
            "Shaymin": "shaymin-land",
            "Mr. Mime": "mr-mime",
            "Mime Jr.": "mime-jr",
            "Farfetch'd": "farfetchd",
            "Nidoran♀": "nidoran-f",
            "Nidoran♂": "nidoran-m",
            "Type: Null": "type-null",
            "Jangmo-o": "jangmo-o",
            "Hakamo-o": "hakamo-o",
            "Kommo-o": "kommo-o",
            "Tapu Koko": "tapu-koko",
            "Tapu Lele": "tapu-lele",
            "Tapu Bulu": "tapu-bulu",
            "Tapu Fini": "tapu-fini",
            "Mr. Rime": "mr-rime",
            "Flabébé": "flabebe",
            "Zygarde 10%": "zygarde-10",
            "Zygarde 50%": "zygarde-50",
            "Zygarde Complete": "zygarde-complete",
            "Ho-Oh": "ho-oh",
            "Porygon-Z": "porygon-z",
            "Rockruff (Own Tempo)": "rockruff-own-tempo",
            "Lycanroc (Midday)": "lycanroc-midday",
            "Lycanroc (Midnight)": "lycanroc-midnight",
            "Lycanroc (Dusk)": "lycanroc-dusk"
        };

        if (map[n]) return map[n];

        var slug = n.toLowerCase();
        slug = slug.replace(/\./g, "");
        slug = slug.replace(/'/g, "");
        slug = slug.replace(/♀/g, "-f");
        slug = slug.replace(/♂/g, "-m");
        slug = slug.replace(/\s+/g, "-");
        slug = slug.replace(/[^a-z0-9-]/g, "");
        return slug;
    }

    function pokemonDbSpriteUrl(pokemonName, isShiny) {
        var slug = pokemonDbSlug(pokemonName);
        if (!slug) return "";
        var folder = isShiny ? "shiny" : "normal";
        return "https://img.pokemondb.net/sprites/black-white/" + folder + "/" + slug + ".png";
    }

    function badgeImg(src, title) {
        return (
            '<img src="' + src + '" ' +
            'class="inline-block h-4 w-4 align-text-top ml-1 translate-y-[0.20em]" ' +
            'title="' + title + '" ' +
            'alt="' + title + '" />'
        );
    }

    function genderSymbol(gender) {
        var g = String(gender || "").trim().toLowerCase();
        if (g === "male") return "♂";
        if (g === "female") return "♀";
        return "";
    }

    function genderBadgeHtml(gender) {
        var g = String(gender || "").trim().toLowerCase();
        if (g === "male") return '<span title="Male" aria-label="Male" style="color:#60a5fa;">♂</span>';
        if (g === "female") return '<span title="Female" aria-label="Female" style="color:#f472b6;">♀</span>';
        return "";
    }

    function genderLabel(gender) {
        var g = String(gender || "").trim().toLowerCase();
        if (g === "male") return "Male";
        if (g === "female") return "Female";
        return "";
    }

    function genderCsvValue(gender) {
        var g = String(gender || "").trim().toLowerCase();
        if (g === "male") return "Male";
        if (g === "female") return "Female";
        return "";
    }

    function genderPokePasteSuffix(gender) {
        var g = String(gender || "").trim().toLowerCase();
        if (g === "male") return " (M)";
        if (g === "female") return " (F)";
        return "";
    }

    // -------------------------
    // Build pokepaste
    // -------------------------
    function buildPokePaste(r) {
        var nick = (r.nickname && String(r.nickname).trim()) ?
            (r.nickname + " (" + (r.pokemon || "") + ")") :
            (r.pokemon || "");

        var item = (r.item && String(r.item).trim()) ? (" @ " + r.item) : "";
        var lines = [];
        var genderSuffix = otsEnabled ? "" : genderPokePasteSuffix(r.gender);
        lines.push((nick + genderSuffix + item).trim());

        if (r.ability) lines.push("Ability: " + r.ability);
        if (r.level) lines.push("Level: " + r.level);
        if (r.shiny === true) lines.push("Shiny: Yes");

        if (!otsEnabled) {
            var evParts = [];
            if (+r.ev_hp) evParts.push(r.ev_hp + " HP");
            if (+r.ev_atk) evParts.push(r.ev_atk + " Atk");
            if (+r.ev_def) evParts.push(r.ev_def + " Def");
            if (+r.ev_spa) evParts.push(r.ev_spa + " SpA");
            if (+r.ev_spd) evParts.push(r.ev_spd + " SpD");
            if (+r.ev_spe) evParts.push(r.ev_spe + " Spe");
            if (evParts.length) lines.push("EVs: " + evParts.join(" / "));

            if (r.nature) lines.push(r.nature + " Nature");

            var ivParts = [];
            if (r.iv_hp != null && r.iv_hp !== "") ivParts.push(r.iv_hp + " HP");
            if (r.iv_atk != null && r.iv_atk !== "") ivParts.push(r.iv_atk + " Atk");
            if (r.iv_def != null && r.iv_def !== "") ivParts.push(r.iv_def + " Def");
            if (r.iv_spa != null && r.iv_spa !== "") ivParts.push(r.iv_spa + " SpA");
            if (r.iv_spd != null && r.iv_spd !== "") ivParts.push(r.iv_spd + " SpD");
            if (r.iv_spe != null && r.iv_spe !== "") ivParts.push(r.iv_spe + " Spe");
            if (ivParts.length) lines.push("IVs: " + ivParts.join(" / "));
        }

        [r.move1, r.move2, r.move3, r.move4].filter(Boolean).forEach(function(m) {
            lines.push("- " + m);
        });

        return lines.join("\n");
    }

    // -------------------------
    // Desktop table renderer
    // -------------------------
    function renderTableRows(rows) {
        if (!tbody) return;
        tbody.innerHTML = "";

        rows.forEach(function(r, idx) {
            var evs = (r.ev_hp || 0) + "/" + (r.ev_atk || 0) + "/" + (r.ev_def || 0) + "/" + (r.ev_spa || 0) + "/" + (r.ev_spd || 0) + "/" + (r.ev_spe || 0);
            var ivs =
                (r.iv_hp != null ? r.iv_hp : "") + "/" +
                (r.iv_atk != null ? r.iv_atk : "") + "/" +
                (r.iv_def != null ? r.iv_def : "") + "/" +
                (r.iv_spa != null ? r.iv_spa : "") + "/" +
                (r.iv_spd != null ? r.iv_spd : "") + "/" +
                (r.iv_spe != null ? r.iv_spe : "");

            var moves = [r.move1, r.move2, r.move3, r.move4].filter(Boolean).join("\n");
            var debug = formatDebug(r);
            var pokepaste = buildPokePaste(r);
            var genderBadge = genderBadgeHtml(r.gender);

            var sprite = pokemonDbSpriteUrl(r.pokemon, r.shiny === true);
            var shinyBadge = r.shiny ? " ⭐" : "";
            var alphaBadge = r.alpha ? badgeImg("./assets/alpha.png", "Alpha Pokémon") : "";
            var haBadge = r.ha ? " 💎" : "";

            var monCell =
                '<td class="px-3 py-3 align-top">' +
                '<div class="items-center gap-3">' +
                (sprite ?
                    '<img src="' + sprite + '" class="h-12 w-12 rounded-xl border border-white/10 bg-white/5 object-contain" ' +
                    'alt="' + escapeHtml(r.pokemon || "") + '" ' +
                    'title="' + escapeHtml(r.pokemon || "") + '" ' +
                    'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'block\';" />' +
                    '<div class="hidden h-10 w-10 rounded-lg border border-white/10 bg-white/5"></div>' :
                    '<div class="h-10 w-10 rounded-lg border border-white/10 bg-white/5"></div>') +
                '<button type="button" class="deleteBtn rounded-lg border border-white/10 bg-rose-500/10 px-2 py-1 text-[11px] font-semibold text-rose-200 hover:bg-rose-500/20" ' +
                'data-rid="' + escapeAttr(r._rid) + '" title="Delete this Pokémon">Delete</button>' +
                "</td>";

            var tr = document.createElement("tr");
            tr.className = (idx % 2 === 0) ? "bg-white/0" : "bg-white/[0.02]";

            tr.innerHTML =
                monCell +
                '<td class="px-3 py-3 align-top text-slate-200/90 debug-col">' + escapeHtml(r.source_file || "") + "</td>" +
                '<td class="px-3 py-3 align-top font-semibold">' + escapeHtml(r.nickname || "") + "</td>" +
                '<td class="px-3 py-3 align-top">' + escapeHtml(r.pokemon || "") + shinyBadge + alphaBadge + haBadge + (genderBadge ? " " + genderBadge : "") + "</td>" +
                '<td class="px-3 py-3 align-top">' + escapeHtml(r.item || "") + "</td>" +
                '<td class="px-3 py-3 align-top">' + escapeHtml(r.ability || "") + "</td>" +
                '<td class="px-3 py-3 align-top">' + escapeHtml(String(r.level || "")) + "</td>" +
                '<td class="px-3 py-3 align-top">' + escapeHtml(r.nature || "") + "</td>" +
                '<td class="px-3 py-3 align-top font-mono whitespace-pre-wrap text-slate-200/90">' + escapeHtml(evs) + "</td>" +
                '<td class="px-3 py-3 align-top font-mono whitespace-pre-wrap text-slate-200/90">' + escapeHtml(ivs) + "</td>" +
                '<td class="px-3 py-3 align-top font-mono whitespace-pre-wrap text-slate-200/90">' + escapeHtml(moves) + "</td>" +

                '<td class="px-3 py-3 align-top w-[360px] min-w-[320px]">' +
                '<button class="copyBtn w-full inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-indigo-500/15 px-3 py-2 text-xs font-semibold text-indigo-200 hover:bg-indigo-500/25 active:scale-[0.99]" ' +
                'title="Copy PokePaste" data-paste="' + escapeAttr(pokepaste) + '">' +
                '<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">' +
                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16h8M8 12h8m-6 8h6a2 2 0 002-2V6a2 2 0 00-2-2h-2l-2-2H8a2 2 0 00-2 2v12a2 2 0 002 2z" />' +
                "</svg>" +
                "Copy PokePaste" +
                "</button>" +
                '<div class="mt-2 rounded-lg border border-white/10 bg-white/5 p-3">' +
                '<pre class="font-mono whitespace-pre-wrap text-[11px] leading-snug text-slate-100/90">' +
                escapeHtml(pokepaste) +
                "</pre>" +
                "</div>" +
                "</td>" +

                '<td class="px-3 py-3 align-top font-mono whitespace-pre-wrap text-slate-200/90 debug-col min-w-[420px] w-[420px]">' +
                escapeHtml(debug) +
                "</td>";

            tbody.appendChild(tr);
        });
    }

    // -------------------------
    // Mobile cards renderer
    // -------------------------
    function renderCards(rows) {
        if (!cards) return;
        cards.innerHTML = "";

        rows.forEach(function(r) {
            var evs = (r.ev_hp || 0) + "/" + (r.ev_atk || 0) + "/" + (r.ev_def || 0) + "/" + (r.ev_spa || 0) + "/" + (r.ev_spd || 0) + "/" + (r.ev_spe || 0);
            var ivs =
                (r.iv_hp != null ? r.iv_hp : "") + "/" +
                (r.iv_atk != null ? r.iv_atk : "") + "/" +
                (r.iv_def != null ? r.iv_def : "") + "/" +
                (r.iv_spa != null ? r.iv_spa : "") + "/" +
                (r.iv_spd != null ? r.iv_spd : "") + "/" +
                (r.iv_spe != null ? r.iv_spe : "");

            var movesInline = [r.move1, r.move2, r.move3, r.move4].filter(Boolean).join(", ");
            var pokepaste = buildPokePaste(r);
            var genderBadge = genderBadgeHtml(r.gender);

            var sprite = pokemonDbSpriteUrl(r.pokemon, r.shiny === true);
            var shinyBadge = r.shiny ? " ⭐" : "";
            var alphaBadge = r.alpha ? badgeImg("./assets/alpha.png", "Alpha Pokémon") : "";
            var haBadge = r.ha ? " 💎" : "";

            var card = document.createElement("div");
            card.className = "rounded-2xl border border-white/10 bg-slate-900/40 p-4 shadow-lg";

            card.innerHTML =
                '<div class="flex items-center gap-3">' +
                (sprite ?
                    '<img src="' + sprite + '" class="h-12 w-12 rounded-xl border border-white/10 bg-white/5 object-contain" ' +
                    'alt="' + escapeHtml(r.pokemon || "") + '" title="' + escapeHtml(r.pokemon || "") + '" ' +
                    'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'block\';" />' +
                    '<div class="hidden h-12 w-12 rounded-xl border border-white/10 bg-white/5"></div>' :
                    '<div class="h-12 w-12 rounded-xl border border-white/10 bg-white/5"></div>') +
                '<div class="min-w-0">' +
                '<div class="text-sm font-semibold text-slate-100 truncate">' +
                escapeHtml(r.pokemon || "") + shinyBadge + alphaBadge + haBadge + (genderBadge ? " " + genderBadge : "") +
                "</div>" +
                '<div class="text-xs text-slate-400 truncate">' + escapeHtml(r.nickname || "") + "</div>" +
                "</div>" +
                '<button type="button" class="deleteBtn rounded-lg border border-white/10 bg-rose-500/10 px-3 py-1.5 text-xs font-semibold text-rose-200 hover:bg-rose-500/20" ' +
                'data-rid="' + escapeAttr(r._rid) + '" title="Delete this Pokémon">Delete</button>' +
                "</div>" +

                '<div class="mt-3 grid grid-cols-2 gap-2 text-xs">' +
                '<div class="text-slate-400">Item</div><div class="text-slate-100">' + escapeHtml(r.item || "") + "</div>" +
                '<div class="text-slate-400">Ability</div><div class="text-slate-100">' + escapeHtml(r.ability || "") + "</div>" +
                '<div class="text-slate-400">Level</div><div class="text-slate-100">' + escapeHtml(String(r.level || "")) + "</div>" +
                '<div class="text-slate-400">Nature</div><div class="text-slate-100">' + escapeHtml(r.nature || "") + "</div>" +
                '<div class="text-slate-400">EVs</div><div class="text-slate-100 font-mono">' + escapeHtml(evs) + "</div>" +
                '<div class="text-slate-400">IVs</div><div class="text-slate-100 font-mono">' + escapeHtml(ivs) + "</div>" +
                '<div class="text-slate-400">Moves</div><div class="text-slate-100">' + escapeHtml(movesInline) + "</div>" +
                "</div>" +

                '<button class="copyBtn mt-3 w-full inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-indigo-500/15 px-3 py-2 text-xs font-semibold text-indigo-200 hover:bg-indigo-500/25 active:scale-[0.99]" ' +
                'data-paste="' + escapeAttr(pokepaste) + '">' +
                '<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">' +
                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16h8M8 12h8m-6 8h6a2 2 0 002-2V6a2 2 0 00-2-2h-2l-2-2H8a2 2 0 00-2 2v12a2 2 0 002 2z" />' +
                "</svg>" +
                "Copy PokePaste" +
                "</button>" +

                '<div class="mt-2 rounded-lg border border-white/10 bg-white/5 p-3">' +
                '<pre class="font-mono whitespace-pre-wrap text-[11px] leading-snug text-slate-100/90">' +
                escapeHtml(pokepaste) +
                "</pre>" +
                "</div>";

            if (debugEnabled) {
                var debug = formatDebug(r);
                card.innerHTML +=
                    '<div class="mt-3 rounded-lg border border-white/10 bg-black/30 p-3">' +
                    '<div class="text-[11px] font-semibold text-slate-300">Debug</div>' +
                    '<div class="mt-1 text-[10px] text-slate-400">Source: ' + escapeHtml(r.source_file || "") + "</div>" +
                    '<pre class="mt-2 font-mono text-[10px] whitespace-pre-wrap text-slate-300/80">' + escapeHtml(debug) + "</pre>" +
                    "</div>";
            }

            cards.appendChild(card);
        });
    }

    // -------------------------
    // Copy handlers + delete handlers
    // -------------------------
    async function copyHandler(e) {
        var btn = e.target.closest(".copyBtn");
        if (!btn) return;

        var txt = btn.getAttribute("data-paste") || "";
        try {
            await navigator.clipboard.writeText(txt);
            btn.classList.add("ring-1", "ring-emerald-400/60", "bg-emerald-500/15");
            setTimeout(function() {
                btn.classList.remove("ring-1", "ring-emerald-400/60", "bg-emerald-500/15");
            }, 700);
            setStatus("PokePaste copied to clipboard.");
        } catch {
            setStatus("Could not copy PokePaste.");
        }
    }

    if (tbody) tbody.addEventListener("click", copyHandler);
    if (cards) cards.addEventListener("click", copyHandler);

    if (tbody) {
        tbody.addEventListener("click", function(e) {
            var btn = e.target.closest(".deleteBtn");
            if (!btn) return;
            var rid = btn.getAttribute("data-rid");
            if (!rid) return;
            deleteRowById(lastMode, rid);
        });
    }

    if (cards) {
        cards.addEventListener("click", function(e) {
            var btn = e.target.closest(".deleteBtn");
            if (!btn) return;
            var rid = btn.getAttribute("data-rid");
            if (!rid) return;
            deleteRowById(lastMode, rid);
        });
    }

    // -------------------------
    // CSV export
    // -------------------------
    function csvCell(v) {
        var s = String(v != null ? v : "");
        if (/[,"\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
        return s;
    }

    if (btnCsv) {
        btnCsv.addEventListener("click", function() {
            var lastRows = store.parse;
            if (!lastRows.length) return;

            var headers = [
                "Nickname", "Pokemon", "Gender", "Item", "Ability", "Hidden Ability", "Shiny", "Level",
                "EV HP", "EV Atk", "EV Def", "EV SpA", "EV SpD", "EV Spe",
                "Nature",
                "IV HP", "IV Atk", "IV Def", "IV SpA", "IV SpD", "IV Spe",
                "Move 1", "Move 2", "Move 3", "Move 4"
            ];

            if (!hidePokePasteFromCsv) {
                headers.push("PokePaste Output");
            }

            var lines = [];
            lines.push(headers.map(csvCell).join(","));

            lastRows.forEach(function(r) {
                var row = [
                    r.nickname || "",
                    r.pokemon || "",
                    genderCsvValue(r.gender),
                    r.item || "",
                    r.ability || "",
                    r.ha ? "Yes" : "No",
                    r.shiny ? "Yes" : "No",
                    (r.level != null ? r.level : ""),
                    (r.ev_hp != null ? r.ev_hp : 0),
                    (r.ev_atk != null ? r.ev_atk : 0),
                    (r.ev_def != null ? r.ev_def : 0),
                    (r.ev_spa != null ? r.ev_spa : 0),
                    (r.ev_spd != null ? r.ev_spd : 0),
                    (r.ev_spe != null ? r.ev_spe : 0),
                    r.nature || "",
                    (r.iv_hp != null ? r.iv_hp : ""),
                    (r.iv_atk != null ? r.iv_atk : ""),
                    (r.iv_def != null ? r.iv_def : ""),
                    (r.iv_spa != null ? r.iv_spa : ""),
                    (r.iv_spd != null ? r.iv_spd : ""),
                    (r.iv_spe != null ? r.iv_spe : ""),
                    r.move1 || "",
                    r.move2 || "",
                    r.move3 || "",
                    r.move4 || ""
                ];

                // Only include PokePaste value if we're NOT hiding it
                if (!hidePokePasteFromCsv) {
                    row.push(buildPokePaste(r));
                }

                lines.push(row.map(csvCell).join(","));
            });

            var blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = "pokemon_import_sheet_format.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        });
    }

    // -------------------------
    // Expose API for other modules (VideoFrameTool, etc.)
    // -------------------------
    window.OcrApp = window.OcrApp || {};
    window.OcrApp.processFiles = processFiles;
    window.OcrApp.selectionType = selectionType;
    window.OcrApp.clearAll = clearAll;
    window.OcrApp.setStatus = setStatus;

})();
