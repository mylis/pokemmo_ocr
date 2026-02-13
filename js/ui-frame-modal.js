// ui-frame-modal.js
(function() {
    const modal = document.getElementById("frameModal");
    const openBtn = document.getElementById("btnExtractFrames");
    const closeBtn = document.getElementById("closeFrameModal");
    const cancelBtn = document.getElementById("cancelFrameBtn");

    if (!modal) return;

    function openModal() {
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");
        document.body.classList.add("modal-open");
    }

    function closeModal() {
        modal.classList.remove("is-open");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("modal-open");
    }

    openBtn && openBtn.addEventListener("click", openModal);
    closeBtn && closeBtn.addEventListener("click", closeModal);
    cancelBtn && cancelBtn.addEventListener("click", closeModal);

    modal.addEventListener("click", (e) => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modal.classList.contains("is-open")) closeModal();
    });

    window.FrameModal = { open: openModal, close: closeModal };
})();