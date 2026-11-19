// Run with: node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { buildText, cleanBody, deployText, dropLines } from "../docs/js/clean.js";

const fixtures = JSON.parse(
  readFileSync(new URL("./fixtures/clean_cases.json", import.meta.url), "utf8"),
);
const boiler = new Set(fixtures.boilerplate);

for (const c of fixtures.cases) {
  test(`clean parity: ${c.name}`, () => {
    const cleaned = cleanBody(c.body);
    assert.equal(cleaned, c.clean_body);
    const dropped = dropLines(cleaned, boiler);
    assert.equal(dropped, c.drop_lines);
    assert.equal(buildText(c.title, dropped), c.build_text);
    assert.equal(deployText(c.title, c.body), c.deploy_no_list);
    assert.equal(deployText(c.title, c.body, boiler), c.build_text);
  });
}
