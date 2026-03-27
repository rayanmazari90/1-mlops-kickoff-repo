(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Scroll reveal */
  function initReveal() {
    if (reduceMotion) {
      document.querySelectorAll(".reveal").forEach(function (el) {
        el.classList.add("is-visible");
      });
      return;
    }
    var els = document.querySelectorAll(".reveal");
    if (!els.length || !("IntersectionObserver" in window)) {
      els.forEach(function (el) {
        el.classList.add("is-visible");
      });
      return;
    }
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            io.unobserve(entry.target);
          }
        });
      },
      { root: null, rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
    );
    els.forEach(function (el) {
      io.observe(el);
    });
  }

  /* Count-up for [data-count] */
  function easeOutQuart(t) {
    return 1 - Math.pow(1 - t, 4);
  }

  function animateCount(el) {
    if (reduceMotion) {
      el.textContent = el.getAttribute("data-count");
      return;
    }
    var target = parseFloat(el.getAttribute("data-count"));
    var suffix = el.getAttribute("data-suffix") || "";
    var prefix = el.getAttribute("data-prefix") || "";
    var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
    var duration = 1400;
    var start = null;

    function frame(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      var eased = easeOutQuart(p);
      var current = target * eased;
      var text;
      if (decimals > 0) {
        text = current.toFixed(decimals);
      } else {
        text = Math.round(current).toString();
      }
      el.textContent = prefix + text + suffix;
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function initCounts() {
    var nodes = document.querySelectorAll("[data-count][data-animate]");
    if (!nodes.length) return;
    if (reduceMotion) {
      nodes.forEach(function (el) {
        var target = el.getAttribute("data-count");
        var suffix = el.getAttribute("data-suffix") || "";
        var prefix = el.getAttribute("data-prefix") || "";
        el.textContent = prefix + target + suffix;
      });
      return;
    }
    if (!("IntersectionObserver" in window)) {
      nodes.forEach(animateCount);
      return;
    }
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animateCount(entry.target);
            entry.target.removeAttribute("data-animate");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.35 }
    );
    nodes.forEach(function (el) {
      io.observe(el);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initReveal();
    initCounts();
  });
})();
