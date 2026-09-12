/* One live lecture worker per reading page. Posters remain useful without JS. */
(() => {
    if (window.fragileGasEmbedsInstalled) return;
    window.fragileGasEmbedsInstalled = true;
    let active;
    function pause() {
        active?.frame.contentWindow?.postMessage(
            { type: "fragile-lecture-pause" },
            location.origin,
        );
    }
    function unload() {
        if (!active) return;
        observer.unobserve(active.figure);
        active.frame.remove();
        active.poster.hidden = false;
        active.button.textContent = "Load experiment";
        active.button.setAttribute("aria-expanded", "false");
        active = null;
    }
    document.addEventListener("click", (event) => {
        const button = event.target.closest("[data-gas-url]");
        if (!button) return;
        if (active?.button === button) {
            unload();
            return;
        }
        unload();
        const figure = button.closest(".gas-demo");
        const poster = figure.querySelector(".gas-demo-poster");
        const frame = document.createElement("iframe");
        const url = new URL(button.dataset.gasUrl, location.href);
        url.searchParams.set("embed", "1");
        frame.src = url.href;
        frame.title =
            figure.querySelector(".gas-demo-title").textContent +
            " interactive experiment";
        frame.className = "gas-demo-frame";
        frame.loading = "lazy";
        poster.after(frame);
        poster.hidden = true;
        button.textContent = "Close experiment";
        button.setAttribute("aria-expanded", "true");
        active = { figure, poster, frame, button };
        observer.observe(figure);
    });
    window.addEventListener("message", (event) => {
        if (
            event.origin !== location.origin ||
            event.source !== active?.frame.contentWindow ||
            event.data?.type !== "fragile-lecture-height" ||
            !Number.isFinite(event.data.height)
        )
            return;
        active.frame.style.height =
            Math.max(650, Math.min(4000, event.data.height)) + "px";
    });
    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries)
                if (entry.target === active?.figure && !entry.isIntersecting)
                    pause();
        },
        { threshold: 0.01 },
    );
    const modeObserver = new MutationObserver(() => {
        if (active && !active.figure.getClientRects().length) unload();
    });
    modeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class", "data-theme"],
    });
    modeObserver.observe(document.body, {
        attributes: true,
        attributeFilter: ["class"],
    });
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) pause();
    });
    window.addEventListener("pagehide", unload);
})();
