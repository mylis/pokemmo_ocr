// ui-example-modal.js
(function() {
    "use strict";

    const modal = document.getElementById("exampleModal");
    const openBtn = document.getElementById("openExampleBtn");
    const closeBtn = document.getElementById("closeExampleBtn");

    const tabShot = document.getElementById("tabShot");
    const tabVid = document.getElementById("tabVid");
    const panelShot = document.getElementById("panelShot");
    const panelVid = document.getElementById("panelVid");

    if (!modal) return;

    function openModal() {
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");
        document.body.classList.add("modal-open");
        // default tab when opening
        setTab("shot");
    }

    function closeModal() {
        modal.classList.remove("is-open");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("modal-open");
    }

    function setTab(which) {
        const isShot = which === "shot";

        if (panelShot) panelShot.classList.toggle("is-hidden", !isShot);
        if (panelVid) panelVid.classList.toggle("is-hidden", isShot);

        if (tabShot) tabShot.classList.toggle("is-active", isShot);
        if (tabVid) tabVid.classList.toggle("is-active", !isShot);
    }

    // Modal open/close
    if (openBtn) openBtn.addEventListener("click", openModal);
    if (closeBtn) closeBtn.addEventListener("click", closeModal);

    modal.addEventListener("click", (e) => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modal.classList.contains("is-open")) closeModal();
    });

    // Tabs
    if (tabShot) tabShot.addEventListener("click", (e) => { e.preventDefault();
        setTab("shot"); });
    if (tabVid) tabVid.addEventListener("click", (e) => { e.preventDefault();
        setTab("vid"); });

    // Initial state (in case modal HTML is visible in DOM)
    setTab("shot");

    // Expose if you want
    window.ExampleModal = { open: openModal, close: closeModal, setTab };
})();