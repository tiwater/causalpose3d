document.addEventListener("DOMContentLoaded", () => {
  // 1. Interactive Benchmark Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });

  // 2. Copy BibTeX to Clipboard
  const copyBtn = document.getElementById("copy-bibtex-btn");
  const bibtexCode = document.getElementById("bibtex-content");
  if (copyBtn && bibtexCode) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(bibtexCode.innerText.trim());
        const originalText = copyBtn.innerText;
        copyBtn.innerText = "Copied!";
        setTimeout(() => {
          copyBtn.innerText = originalText;
        }, 2000);
      } catch (err) {
        console.error("Failed to copy BibTeX:", err);
      }
    });
  }

  // 3. Figure Lightbox Modal
  const modal = document.getElementById("lightbox-modal");
  const modalImg = document.getElementById("lightbox-img");
  const closeBtn = document.getElementById("lightbox-close");

  document.querySelectorAll(".zoomable-fig").forEach((img) => {
    img.addEventListener("click", () => {
      if (modal && modalImg) {
        modalImg.src = img.src;
        modalImg.alt = img.alt;
        modal.classList.add("open");
      }
    });
  });

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal || e.target === closeBtn) {
        modal.classList.remove("open");
      }
    });
  }
});
