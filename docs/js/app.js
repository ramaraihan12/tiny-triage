import { deployText } from "./clean.js";

const $ = (id) => document.getElementById(id);
const MODEL_URL = "model/";
const EXAMPLE = {
  title: "Crash when saving a project with an empty name",
  body:
    "After updating to 2.4.1 the app closes without an error message when I press Save on a " +
    "project whose name field is empty.\n\n" +
    "Steps:\n1. Create a new project\n2. Clear the name field\n3. Press Ctrl+S\n\n" +
    "Expected: a validation message. Actual: the window disappears.\n\n" +
    "Windows 11, 2.4.1. It worked in 2.3.x.",
};

let summary = null;
let classifier = null;
let boiler = new Set();
let loading = null;

const fmt = (x, d = 3) => Number(x).toFixed(d);
const count = (x) => Number(x).toLocaleString("en-US");

function show(el, on = true) {
  el.hidden = !on;
}

function message(text) {
  const el = $("msg");
  el.textContent = text || "";
  show(el, Boolean(text));
}

// ---------------------------------------------------------------- headline numbers

function fillSummary(s) {
  const t = s.test;
  const values = {
    repos_total: s.repos.total,
    repos_train: s.repos.train,
    repos_val: s.repos.val,
    repos_test: s.repos.test,
    n_cat: count(t.n_category),
    n_ni: count(t.n_needs_info),
    cat_model: fmt(t.category.model),
    cat_base: fmt(t.category.baseline),
    cat_major: fmt(t.category.majority),
    cat_diff: (t.category.diff >= 0 ? "+" : "") + fmt(t.category.diff),
    cat_ci: `${fmt(t.category.ci_low)} to ${fmt(t.category.ci_high)}`,
    ni_model: fmt(t.needs_info.model),
    ni_base: fmt(t.needs_info.baseline),
    ni_model_default: fmt(t.needs_info.model_default),
    ni_base_default: fmt(t.needs_info.baseline_default),
    ni_diff: fmt(t.needs_info.diff),
    ni_ci: `${fmt(t.needs_info.ci_low)} to ${fmt(t.needs_info.ci_high)}`,
    demo_cat_model: fmt(s.demo.category.model),
    demo_cat_base: fmt(s.demo.category.baseline),
    demo_ni_model: fmt(s.demo.needs_info.model),
    demo_ni_base: fmt(s.demo.needs_info.baseline),
    download_mb: s.download_mb,
    model_mb: s.model_mb,
    thr: s.needs_info_threshold,
  };
  for (const el of document.querySelectorAll("[data-s]")) {
    const v = values[el.dataset.s];
    if (v !== undefined) el.textContent = v;
  }
}

// ---------------------------------------------------------------- table

const state = { rows: [], sort: null, dir: 1 };

function cell(pred, truth) {
  const td = document.createElement("td");
  td.className = "pred";
  const cat = document.createElement("span");
  cat.textContent = pred.category;
  cat.className = truth ? (pred.category === truth.category ? "ok" : "bad") : "plain";
  td.append(cat);
  const ni = document.createElement("span");
  const has = truth && truth.needs_info;
  const label = pred.needs_info === "unparsed" ? "unparsed" : `info: ${pred.needs_info}`;
  ni.textContent = label;
  ni.className = has ? (pred.needs_info === truth.needs_info ? "ok" : "bad") : "plain";
  td.append(ni);
  return td;
}

function renderTable() {
  const repo = $("f-repo").value;
  const onlyDiff = $("f-diff").checked;
  let rows = state.rows.filter((r) => !repo || r.repo === repo);
  if (onlyDiff) {
    rows = rows.filter((r) =>
      ["baseline", "model", "llama"].some(
        (k) =>
          r[k].category !== r.maintainer.category ||
          (r.maintainer.needs_info && r[k].needs_info !== r.maintainer.needs_info),
      ),
    );
  }
  if (state.sort) {
    const key = (r) =>
      state.sort === "issue"
        ? `${r.repo}#${String(r.number).padStart(9, "0")}`
        : `${(r[state.sort] || r.maintainer).category}/${(r[state.sort] || r.maintainer).needs_info}`;
    rows = [...rows].sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0) * state.dir);
  }
  const body = document.querySelector("#table tbody");
  body.replaceChildren();
  for (const r of rows) {
    const tr = document.createElement("tr");
    const issue = document.createElement("td");
    issue.className = "issue";
    const a = document.createElement("a");
    a.href = r.url;
    a.textContent = `${r.repo}#${r.number}`;
    a.className = "ref";
    a.rel = "noopener";
    const title = document.createElement("span");
    title.className = "ttl";
    title.textContent = r.title;
    issue.append(a, title);
    const m = document.createElement("td");
    m.className = "pred";
    m.dataset.label = "Maintainer";
    const mc = document.createElement("span");
    mc.className = "plain";
    mc.textContent = r.maintainer.category;
    m.append(mc);
    if (r.maintainer.needs_info) {
      const mn = document.createElement("span");
      mn.className = "plain";
      mn.textContent = `info: ${r.maintainer.needs_info}`;
      m.append(mn);
    }
    tr.append(issue, m);
    for (const [k, label] of [
      ["baseline", "TF-IDF"],
      ["model", "tiny-triage"],
      ["llama", "Llama 3.3"],
    ]) {
      const td = cell(r[k], r.maintainer);
      td.dataset.label = label;
      tr.append(td);
    }
    body.append(tr);
  }
  document.querySelectorAll("#table th[data-sort]").forEach((th) => {
    th.setAttribute(
      "aria-sort",
      state.sort === th.dataset.sort ? (state.dir > 0 ? "ascending" : "descending") : "none",
    );
  });
}

function setupTable(data) {
  state.rows = data.rows;
  for (const el of document.querySelectorAll('[data-s="seed"]')) el.textContent = `seed ${data.seed}`;
  const repos = [...new Set(data.rows.map((r) => r.repo))].sort();
  for (const r of repos) $("f-repo").append(new Option(r, r));
  const s = data.summary;
  const c = s.category;
  const n = s.needs_info;
  const un = s.llama_unparsed;
  $("table-summary").textContent =
    `Category matches the maintainer on ${c.baseline} of ${data.n} (TF-IDF), ` +
    `${c.model} of ${data.n} (tiny-triage) and ${c.llama} of ${data.n} (Llama 3.3). ` +
    `needs_info, on the ${n.n} issues whose repo has that label: ${n.baseline} (TF-IDF), ` +
    `${n.model} (tiny-triage), ${n.llama} (Llama 3.3). ` +
    `For scale, always answering the most common maintainer class would match ` +
    `${data.summary.majority.category.count} of ${data.n} on category ("${data.summary.majority.category.class}") and ` +
    `${data.summary.majority.needs_info.count} of ${n.n} on needs_info ("${data.summary.majority.needs_info.class}"), ` +
    (data.summary.majority.needs_info.count > Math.max(n.baseline, n.model, n.llama)
      ? `which is more than any system gets on needs_info here. `
      : ``) +
    `Llama replies that could not be parsed: ${un.category} for category, ${un.needs_info} for needs_info. ` +
    `Small sample, so read these as anecdotes rather than a ranking.`;
  $("f-repo").addEventListener("change", renderTable);
  $("f-diff").addEventListener("change", renderTable);
  document.querySelectorAll("#table th[data-sort]").forEach((th) => {
    th.tabIndex = 0;
    const go = () => {
      const k = th.dataset.sort;
      state.dir = state.sort === k ? -state.dir : 1;
      state.sort = k;
      renderTable();
    };
    th.addEventListener("click", go);
    th.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        go();
      }
    });
  });
  renderTable();
}

// ---------------------------------------------------------------- live box

function runLabel() {
  $("run").textContent = classifier
    ? "Classify"
    : `Run in my browser (downloads about ${summary ? summary.download_mb : "?"} MB, once)`;
}

async function loadModel() {
  if (classifier) return classifier;
  if (!loading) {
    loading = (async () => {
      const progress = $("progress");
      show(progress, true);
      progress.removeAttribute("value");
      $("loadhint").textContent = "Downloading the model and the WebAssembly runtime...";
      const { Classifier } = await import("./model.js");
      const clf = await Classifier.load(MODEL_URL, (got, total) => {
        if (total) {
          progress.value = got / total;
          $("loadhint").textContent = `Downloading the model: ${fmt(got / 1e6, 1)} of ${fmt(total / 1e6, 1)} MB`;
        }
      });
      const cleaning = clf.labels.cleaning || {};
      if (cleaning.boilerplate) {
        const res = await fetch(new URL(MODEL_URL + cleaning.boilerplate, document.baseURI));
        boiler = new Set((await res.json()).lines);
      }
      classifier = clf;
      show(progress, false);
      $("loadhint").textContent = `Model ready, loaded in ${fmt(clf.loadMs / 1000, 1)} s. It stays in this tab; nothing you type is uploaded.`;
      runLabel();
      return clf;
    })().catch((err) => {
      loading = null;
      show($("progress"), false);
      throw err;
    });
  }
  return loading;
}

function renderResult(res, text) {
  show($("result"), true);
  $("r-cat").textContent = res.category;
  const bars = $("r-bars");
  bars.replaceChildren();
  const order = Object.entries(res.categoryProbs).sort((a, b) => b[1] - a[1]);
  order.forEach(([name, p], i) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    track.append(fill);
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = `${(p * 100).toFixed(1)}%`;
    if (i === 0) li.className = "top";
    li.append(label, track, num);
    bars.append(li);
  });
  $("r-ni").textContent = res.needsInfo === "yes" ? "likely asks for more info" : "probably enough to act on";
  $("r-ni-p").textContent =
    `P(yes) = ${res.needsInfoProb.toFixed(2)}, threshold ${classifier.labels.needs_info_threshold}` +
    ` (the threshold was tuned on validation repos and does not transfer well).`;
  $("r-time").textContent =
    `Model loaded in ${fmt(classifier.loadMs / 1000, 1)} s. This issue: ${res.ms.toFixed(0)} ms ` +
    `for ${res.tokens} tokens (tokenizer and model, one thread, WebAssembly).`;
  $("r-text").textContent = text;
}

function onSubmit(e) {
  e.preventDefault();
  message("");
  return classify();
}

async function classify() {
  const title = $("title").value;
  const body = $("body").value;
  if (!title.trim() && !body.trim()) {
    message("Paste an issue first: a title, a body, or a GitHub issue URL.");
    return;
  }
  const button = $("run");
  button.disabled = true;
  try {
    await loadModel();
    const text = deployText(title, body, boiler);
    renderResult(await classifier.classify(text), text);
  } catch (err) {
    console.error(err);
    message(`Could not run the model: ${err.message || err}`);
  } finally {
    button.disabled = false;
    runLabel();
  }
}

const ISSUE_URL = /^(?:https?:\/\/)?(?:www\.)?github\.com\/([\w.-]+)\/([\w.-]+)\/(?:issues|pull)\/(\d+)/i;

async function fetchIssue() {
  message("");
  const m = ISSUE_URL.exec($("url").value.trim());
  if (!m) {
    message("That does not look like a GitHub issue URL, for example https://github.com/owner/repo/issues/123.");
    return;
  }
  const [, owner, repo, num] = m;
  const button = $("fetch");
  button.disabled = true;
  try {
    const res = await fetch(`https://api.github.com/repos/${owner}/${repo}/issues/${num}`, {
      headers: { accept: "application/vnd.github+json" },
    });
    if (res.status === 404) {
      message("GitHub says that issue does not exist, or the repo is private.");
      return;
    }
    if (res.status === 403 || res.status === 429) {
      const reset = Number(res.headers.get("x-ratelimit-reset"));
      const when = reset ? ` Try again after ${new Date(reset * 1000).toLocaleTimeString()}.` : "";
      message(`GitHub's limit for requests without a token (60 an hour per IP address) is used up.${when} You can paste the title and body instead.`);
      return;
    }
    if (!res.ok) {
      message(`GitHub answered HTTP ${res.status}. You can paste the title and body instead.`);
      return;
    }
    const issue = await res.json();
    $("title").value = issue.title || "";
    $("body").value = issue.body || "";
    if (issue.pull_request) {
      message("That is a pull request. The model only saw issues, so expect a worse guess.");
    }
    await classify();
  } catch (err) {
    console.error(err);
    message("Could not reach api.github.com from this browser. You can paste the title and body instead.");
  } finally {
    button.disabled = false;
  }
}

async function main() {
  $("example").addEventListener("click", () => {
    $("title").value = EXAMPLE.title;
    $("body").value = EXAMPLE.body;
  });
  $("form").addEventListener("submit", onSubmit);
  $("fetch").addEventListener("click", fetchIssue);
  runLabel();
  $("loadhint").textContent =
    "The first run downloads the model, its tokenizer and the WebAssembly runtime. Nothing you type is uploaded.";
  const [s, t] = await Promise.all([
    fetch("data/summary.json").then((r) => r.json()),
    fetch("data/table.json").then((r) => r.json()),
  ]);
  summary = s;
  fillSummary(s);
  runLabel();
  setupTable(t);
}

main().catch((err) => {
  console.error(err);
  message("The page data could not be loaded. Reload the page, or see the README on GitHub.");
});
