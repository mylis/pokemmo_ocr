// ui-copy-discord.js
(function() {
    var btn = document.getElementById("copyDiscord");
    var label = document.getElementById("discordLabel");
    if (!btn || !label) return;

    var username = ".mylis";
    var originalText = label.textContent;

    btn.addEventListener("click", async function() {
        try {
            await navigator.clipboard.writeText(username);

            label.textContent = "Copied!";
            btn.classList.add("is-success");

            setTimeout(function() {
                label.textContent = originalText;
                btn.classList.remove("is-success");
            }, 900);
        } catch (e) {
            label.textContent = "Copy failed";
            setTimeout(function() {
                label.textContent = originalText;
            }, 900);
        }
    });
})();