// app.js
(function() {
    if (!window.VideoFrameTool || !window.OcrApp) return;

    window.VideoFrameTool.init({
        onToOcr: async(files) => window.OcrApp.processFiles(files, "parse"),
        onToFirestore: async(files) => window.OcrApp.processFiles(files, "firestore"),
    });
})();