// Run in the browser console on /demo. Resolves to a reusable UI evidence receipt.
(async () => {
  const checks = [];
  const wait = async (test, label) => {
    const until = performance.now() + 15000;
    while (performance.now() < until) {
      if (test()) return;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    throw new Error(`Timed out: ${label}`);
  };
  const check = (condition, name, detail = null) => {
    if (!condition) throw new Error(name);
    checks.push({ name, passed: true, detail });
  };
  const lens = name => [...document.querySelectorAll("nav[aria-label='Review lens'] button")]
    .find(button => button.textContent.includes(name));
  lens("Decision lens").click();
  await wait(() => document.getElementById("share") && !document.querySelector("video"), "decision view");
  check(!document.querySelector("[aria-label='Qc']") && !document.querySelector("[aria-label='Td']"),
    "Decision view has no opponent camera cards or video");
  const body = await (await fetch("/api/demo/wpt/review")).json();
  check(!body.hindsight && !JSON.stringify(body).includes('"Qc"'), "Default API redacts hindsight");
  check(body.decision.extra_call === 2140000 && body.decision.called_final_pot === 7290000,
    "Exact replay arithmetic");
  for (const [percent, expected] of [[0, "-2,140,000"], [20, "-682,000"], [30, "+47,000"], [100, "+5,150,000"]]) {
    const input = document.getElementById("share");
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, String(percent));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await wait(() => document.querySelector("[data-testid='scenario-ev']").textContent.includes(expected), `slider ${percent}`);
    check(true, `Slider ${percent}% renders ${expected} chips`);
  }
  lens("Broadcast & outcome").click();
  await wait(() => document.querySelector("video")?.readyState >= 2, "broadcast video loaded");
  const video = document.querySelector("video");
  check(video.videoWidth > 0 && !video.error, "Actual video decodes", { width: video.videoWidth, height: video.videoHeight, duration: video.duration });
  check(document.querySelector("[aria-label='Qc']") && document.querySelector("[aria-label='Td']"),
    "Explicit broadcast lens reveals camera cards");
  const moment = [...document.querySelectorAll("[aria-label='Action timeline'] button")]
    .find(button => button.textContent.includes("Ian bets 800000"));
  moment.click();
  await wait(() => Math.abs(video.currentTime - 189) < .3 && !video.seeking && video.readyState >= 2, "timeline seek");
  check(true, "Timeline seeks decoded video to 03:09", { currentTime: video.currentTime });
  await video.play();
  await new Promise(resolve => setTimeout(resolve, 400));
  video.pause();
  check(video.currentTime > 189.1, "Video playback advances after explicit interaction", { currentTime: video.currentTime });
  check(document.body.textContent.includes("-1.49M") && document.body.textContent.includes("-3.63M"),
    "Full-hand outcome and conditional counterfactual are separate");
  check([...document.querySelectorAll("a")].some(a => a.textContent === "CC BY 3.0"),
    "License credit is visible beside video");
  lens("Decision lens").click();
  await wait(() => document.getElementById("share") && !document.querySelector("video"), "return to decision");
  check(!document.querySelector("[aria-label='Qc']"), "Returning to decision removes hindsight");
  const nativeFetch = window.fetch;
  try {
    window.fetch = (...args) => String(args[0]).includes("/api/demo/wpt/review")
      ? Promise.reject(new Error("Deliberate UI verification outage")) : nativeFetch(...args);
    lens("Broadcast & outcome").click();
    await wait(() => document.querySelector("[role='alert']"), "explicit error");
    check(document.body.textContent.includes("Deliberate UI verification outage") && !document.querySelector("video"),
      "Request failure shows an explicit error, not stale analysis");
  } finally { window.fetch = nativeFetch; }
  [...document.querySelectorAll("button")].find(button => button.textContent === "Retry verification").click();
  await wait(() => document.querySelector("video"), "retry recovers");
  lens("Decision lens").click();
  await wait(() => document.getElementById("share") && !document.querySelector("video"), "final safe lens");
  const frame = document.createElement("iframe");
  frame.style.cssText = "width:390px;height:900px;border:0;position:fixed;left:0;top:0;z-index:10000";
  frame.src = "/demo";
  document.body.append(frame);
  try {
    await wait(() => frame.contentDocument?.getElementById("share"), "mobile viewport");
    const root = frame.contentDocument.documentElement;
    check(root.scrollWidth <= root.clientWidth, "390px mobile viewport has no horizontal page overflow",
      { scrollWidth: root.scrollWidth, clientWidth: root.clientWidth });
    check(!frame.contentDocument.querySelector("video"), "Mobile default lens does not load video");
  } finally { frame.remove(); }
  const receipt = { url: location.href, finished_at: new Date().toISOString(), checks };
  window.__reviewDemoProof = receipt;
  console.log(JSON.stringify(receipt, null, 2));
  return receipt;
})()
