// Port of src/tiny_triage/clean.py. Output must match the Python code exactly
// (tests/fixtures/clean_cases.json, tests/clean.test.mjs).
//
// Python's \s and str.strip() use the Unicode whitespace set, JS's differ
// (JS also counts U+FEFF; Python also counts U+001C..U+001F and U+0085),
// and Python's \w is Unicode while JS's is ASCII, so the classes are spelled out.

// Known differences that are not handled, both vanishingly rare (none of the 8,797 val and test
// issues hit them): Python's case-insensitive match also treats U+0130 and U+0131 as i, which
// affects only the HTML tag and "no response" patterns, and Python 3.11 uses Unicode 14 for \w
// while JS uses newer tables, so a letter added after Unicode 14 counts as a word character in
// JS only.

export const MAX_BODY_CHARS = 2000;
export const MAX_CODE_LINES = 40;

const WS_CODES = [
  [0x09, 0x0d], [0x1c, 0x20], [0x85, 0x85], [0xa0, 0xa0], [0x1680, 0x1680],
  [0x2000, 0x200a], [0x2028, 0x2029], [0x202f, 0x202f], [0x205f, 0x205f], [0x3000, 0x3000],
];
const hex = (n) => String.fromCharCode(92) + "u{" + n.toString(16) + "}"; // \u{...} for the RegExp
const WS = WS_CODES.map(([a, b]) => (a === b ? hex(a) : hex(a) + "-" + hex(b))).join("");
const WORD = String.raw`\p{L}\p{N}_`;

const HTML_COMMENT = /<!--[\s\S]*?-->/gu;
const FENCE = /(```|~~~)[^\n]*\n([\s\S]*?)(?:\n[ \t]*\1[ \t]*(?=\n|$)|$)/gu;
const URL_RE = new RegExp(`https?://[^${WS}]+`, "gu");
const MENTION = new RegExp(`(?<![${WORD}/])@[A-Za-z0-9][A-Za-z0-9-]*(?:/[${WORD}-]+)?`, "gu");
const IMAGE = /!\[[^\]]*\]\([^)]*\)/gu;
const HTML_TAG = new RegExp(
  `</?(?:details|summary|img|br|p|div|span|kbd|sub|sup)(?![${WORD}])[^>]*>`,
  "giu",
);
const CHECKBOX = new RegExp(`^[${WS}]*[-*][${WS}]*\\[[ xX]\\]`, "u");
const NO_RESPONSE = new RegExp(`^[${WS}]*_?no response_?[${WS}]*$`, "iu");
const BLANKS = /\n{3,}/gu;
const SPACE = new RegExp(`[${WS}]`, "u");
const isSpace = (ch) => SPACE.test(ch);

// str.rstrip() / str.strip() with Python's whitespace set (loops, so no regex backtracking)
function rstrip(s) {
  let end = s.length;
  while (end > 0 && isSpace(s[end - 1])) end--;
  return s.slice(0, end);
}
function strip(s) {
  let start = 0;
  while (start < s.length && isSpace(s[start])) start++;
  return rstrip(s.slice(start));
}

function shrinkCode(whole, _fence, code) {
  const lines = code.split("\n").length;
  return lines > MAX_CODE_LINES ? "\n[long code block removed]\n" : whole;
}

export function cleanBody(body) {
  let text = (body || "").replaceAll("\r\n", "\n").replaceAll("\r", "\n");
  text = text.replace(HTML_COMMENT, "");
  text = text.replace(FENCE, shrinkCode);
  text = text.replace(IMAGE, () => "[image]");
  text = text.replace(URL_RE, () => "<url>");
  text = text.replace(MENTION, () => "@user");
  text = text.replace(HTML_TAG, "");
  const kept = text
    .split("\n")
    .map(rstrip)
    .filter((ln) => !CHECKBOX.test(ln) && !NO_RESPONSE.test(ln));
  return strip(kept.join("\n").replace(BLANKS, "\n\n"));
}

export function dropLines(body, boiler) {
  const kept = body.split("\n").filter((ln) => !boiler.has(strip(ln)));
  return strip(kept.join("\n").replace(BLANKS, "\n\n"));
}

export function buildText(title, body) {
  const cut = Array.from(body).slice(0, MAX_BODY_CHARS).join("");
  return strip(`${strip(title)}\n\n${strip(cut)}`);
}

// What the demo feeds the model: clean_body, drop the frozen template lines, title + body.
export function deployText(title, body, boiler = new Set()) {
  return buildText(title, dropLines(cleanBody(body), boiler));
}
