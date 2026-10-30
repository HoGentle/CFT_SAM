/* 验证两种模式的数据就绪判定、自动校验、预览后折叠和图例归属。 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {value: "", open: true, textContent: "", setAttribute() {},
      classList: {add: (v) => classes.add(v), remove: (v) => classes.delete(v),
        contains: (v) => classes.has(v), toggle(v, on) { if (on) classes.add(v); else classes.delete(v); }}});
  }
  return elements.get(id);
}
const context = vm.createContext({assert, console,
  document: {getElementById: element, querySelectorAll: () => [], addEventListener() {}},
  window: {addEventListener() {}},
});
const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
vm.runInContext(source.replace(/\ninit\(\);\s*$/, ""), context);
vm.runInContext(`
  let validations = 0, polls = 0;
  api = async (url) => {
    assert.equal(url, "/api/validate");
    validations++;
    return {session: state.session, errors: [], warnings: [], report: {}};
  };
  renderFiles = () => {};
  renderValidation = () => { state.validated = true; };
  renderProgress = () => {};
  toast = () => {};
  pollJob = () => { polls++; };
  showPreview = () => {};
  refreshSession = async () => { state.session.has_input_preview = true; };
  state.session = {files: {preview: {name: "result..tif"}}, has_input_preview: false};
`, context);
async function run() {
  await vm.runInContext("autoValidateInput()", context);
  vm.runInContext(`assert.equal(validations, 0); state.mode = "annotation"; state.importing = 1;`, context);
  await vm.runInContext("autoValidateInput()", context);
  vm.runInContext(`assert.equal(validations, 0); state.importing = 0;`, context);
  await vm.runInContext("autoValidateInput()", context);
  vm.runInContext(`
    assert.equal(validations, 1);
    assert.equal(polls, 1);
    assert.equal($("card-params").classList.contains("hidden"), true);
    assert.equal($("btn-run").disabled, true);
  `, context);
  await vm.runInContext("onJobFinished({result: {input_preview: {}}})", context);
  vm.runInContext(`
    assert.equal($("card-input").open, false);
    assert.equal($("card-validation").open, false);
    assert.equal($("card-regions").open, true);
    assert.equal($("card-regions").classList.contains("hidden"), false);
    state.mode = "prescription";
    Object.assign(state.session.files, {B1:{},B2:{},B3:{},B4:{}});
  `, context);
  await vm.runInContext("autoValidateInput()", context);
  vm.runInContext(`
    assert.equal(validations, 2);
    assert.equal($("card-params").classList.contains("hidden"), false);
    state.result = {prescription_preview_range: {min:9, max:11}, gradient_css: "测试渐变"};
    assert.ok(resultLegendHtml("prescription").includes("推荐施肥量"));
    state.mode = "annotation";
    assert.equal(resultLegendHtml("prescription"), "");
  `, context);
  console.log("模式切换、自动校验、预览后折叠及独立图例验证通过。");
}
run().catch((error) => {console.error(error); process.exitCode = 1;});
