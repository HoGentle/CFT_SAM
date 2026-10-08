/* 验证点提示编辑的确认、取消和过期响应处理；不依赖浏览器。 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const elements = new Map();
const timers = new Map();
let timerId = 0;
const listeners = {};
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: id === "roi-method" ? "sam" : "1",
    checked: false,
    addEventListener(type, handler) { this.listeners ||= {}; this.listeners[type] = handler; },
    getBoundingClientRect() { return {left:0, top:0}; },
    classList: { toggle() {}, add() {}, remove() {}, contains() { return true; } },
  });
  return elements.get(id);
}
const context = vm.createContext({
  document: { getElementById: element, createElement: () => ({}), querySelectorAll: () => [],
              addEventListener: (type, handler) => { listeners[type] = handler; } },
  window: { addEventListener: (type, handler) => { listeners[type] = handler; } },
  console, assert,
  setTimeout: (handler) => { const id = ++timerId; timers.set(id, handler); return id; },
  clearTimeout: (id) => timers.delete(id),
  flushTimer: (id) => { const handler = timers.get(id); if (handler) { timers.delete(id); return handler(); } },
});
const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
vm.runInContext(source.replace(/\ninit\(\);\s*$/, ""), context);

vm.runInContext(`
  state.session = { preview_id: "image-1", georef: {} };
  $("sam-model").value = "sam1_vit_h";
  $("sam-hole-area").value = "10";
  $("sam-gap-width").value = "0.2";
  viewer.naturalW = 100;
  viewer.naturalH = 100;
  viewer.scale = 1;
  roi.editing = true;
  updateRoiPanel = () => {};
  refreshRoiArea = () => {};
  let resolvePrediction, rejectPrediction;
  const requests = [];
  api = (url, payload) => { assert.equal(url, "/api/roi/sam"); requests.push(payload); return new Promise((resolve, reject) => { resolvePrediction = resolve; rejectPrediction = reject; }); };
  const result = () => ({ preview_id: "image-1", model_id: $("sam-model").value, model_label: "测试模型", regions: [
    { hull: [{x:1,y:1,lon:113,lat:30},{x:8,y:1,lon:113.1,lat:30},{x:8,y:8,lon:113.1,lat:30.1}], holes: [] }
  ], points: [{x:3,y:3,lon:113.01,lat:30.01,label:1}] });
`, context);

async function run() {
  vm.runInContext(`
    state.config = { sam_models: [
      { id:"sam1_vit_h", label:"原模型", available:true },
      { id:"sam3", label:"第三代", available:false, reason:"缺少依赖" }
    ] };
    $("sam-model").options = [];
    $("sam-model").appendChild = (option) => $("sam-model").options.push(option);
    initSamModels();
    assert.equal($("sam-model").options.length, 2);
    assert.equal($("sam-model").options[1].disabled, true);
    assert.equal($("sam-model").value, "sam1_vit_h");
  `, context);
  vm.runInContext(`addSamPoint(3, 3, false); assert.equal(sam.prompts[0].label, 1);`, context);
  let request = vm.runInContext("predictSamRegion()", context);
  vm.runInContext("clearSam(); resolvePrediction(result());", context);
  await request;
  vm.runInContext(`assert.equal(sam.result, null); assert.equal(sam.busy, false); assert.equal(roi.regions.length, 0);`, context);

  vm.runInContext("addSamPoint(3, 3, false);", context);
  request = vm.runInContext("predictSamRegion()", context);
  vm.runInContext("resolvePrediction(result());", context);
  await request;
  vm.runInContext(`
    assert.ok(sam.result);
    assert.equal(roi.regions.length, 0);
    acceptSamRegion();
    assert.equal(roi.regions.length, 1);
    assert.equal(roi.regions[0].source, "sam");
    assert.equal(serializeRoiRegion(roi.regions[0]).point_labels[0], 1);
    assert.equal(sam.prompts.length, 0);
    undoRoiPoint();
    assert.equal(roi.regions.length, 0);
    assert.equal(sam.prompts[0].label, 1);
    assert.equal(roi.current.length, 0);
  `, context);

  request = vm.runInContext("predictSamRegion()", context);
  vm.runInContext('state.session.preview_id = "image-2"; resolvePrediction(result());', context);
  await request;
  vm.runInContext("assert.equal(sam.result, null); state.session.preview_id = 'image-1';", context);

  request = vm.runInContext("predictSamRegion()", context);
  vm.runInContext("roi.regions.push({...result().regions[0], points:[]}); resolvePrediction(result());", context);
  await request;
  vm.runInContext("assert.equal(sam.result, null);", context);

  vm.runInContext("roi.regions = [];", context);
  request = vm.runInContext("predictSamRegion()", context);
  vm.runInContext('$("sam-model").value = "hq_sam_vit_l"; changeSamModel(); resolvePrediction({...result(), model_id:"sam1_vit_h"});', context);
  await request;
  vm.runInContext(`
    assert.equal(sam.result, null);
    assert.equal(sam.prompts.length, 1);
    assert.equal(sam.prompts[0].label, 1);
  `, context);
  request = vm.runInContext("predictSamRegion()", context);
  vm.runInContext("resolvePrediction(result());", context);
  await request;
  vm.runInContext(`
    acceptSamRegion();
    assert.equal(roi.regions[0].model_id, "hq_sam_vit_l");
    assert.equal(serializeRoiRegion(roi.regions[0]).model_id, "hq_sam_vit_l");
    $("sam-model").value = "sam3";
    undoRoiPoint();
    assert.equal($("sam-model").value, "hq_sam_vit_l");
  `, context);

  vm.runInContext(`
    clearSam(); addSamPoint(80, 80, true);
    assert.equal(sam.prompts[0].label, 0);
    assert.equal($("btn-sam-predict").disabled, true);
    addSamPoint(80, 80, true);
    assert.equal(sam.prompts.length, 0);
  `, context);
  // 自动生成、按住 m 的左右键、多点合并，以及输入控件中按 m 不触发快捷键。
  vm.runInContext(`
    clearSam(); roi.regions = []; $("roi-method").value = "sam";
    initViewer();
    addSamPoint(3, 3, false);
    flushTimer(sam.timer);
    assert.equal(sam.busy, true);
    assert.equal(requests.at(-1).points[0].label, 1);
    handleSamContextMenu({preventDefault(){}});
    assert.equal(sam.confirmVersion, sam.version);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(roi.regions.length, 1);
    assert.equal(sam.prompts.length, 0);
    roi.regions = [];
    $("sam-label").value = "0";
    handleSamKeyDown({key:"m",target:{closest:()=>true},preventDefault(){}});
    assert.equal(sam.multi, false);
    handleSamKeyDown({key:"M",target:{},preventDefault(){}});
    assert.equal(sam.multi, true);
    $("viewer-stage").listeners.pointerdown({button:0,clientX:3,clientY:3});
    $("viewer-stage").listeners.pointerup({button:0,clientX:3,clientY:3});
    handleSamContextMenu({clientX:80,clientY:80,preventDefault(){}});
    assert.equal(sam.prompts.length, 2);
    assert.equal(sam.prompts[0].label, 1);
    assert.equal(sam.prompts[1].label, 0);
    assert.equal(sam.busy, false);
    assert.equal(sam.timer, null);
    handleSamKeyUp({key:"m"});
    flushTimer(sam.timer);
    assert.equal(sam.busy, true);
    assert.equal(requests.at(-1).points.length, 2);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.ok(sam.result);
    handleSamContextMenu({preventDefault(){}});
    assert.equal(roi.regions.length, 1);
    roi.regions = [];
    $("sam-label").value = "1";
    addSamPoint(3, 3); flushTimer(sam.timer);
    const firstRequest = requests.at(-1);
    addSamPoint(25, 25);
    assert.equal(firstRequest.points.length, 1);
    assert.equal(sam.prompts.length, 2);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(sam.result, null);
    assert.equal(sam.pending, true);
    flushTimer(sam.timer);
    assert.equal(requests.at(-1).points.length, 2);
    $("sam-fill-holes").checked = true;
    $("sam-hole-area").value = "12.5";
    $("sam-smooth-boundary").checked = true;
    $("sam-ignore-boundaries").checked = true;
    $("sam-gap-width").value = "0.3";
    changeSamPostprocess();
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(sam.result, null);
    flushTimer(sam.timer);
    assert.equal(requests.at(-1).postprocess.fill_holes, true);
    assert.equal(requests.at(-1).postprocess.max_hole_area_m2, 12.5);
    assert.equal(requests.at(-1).postprocess.smooth_boundary, true);
    assert.equal(requests.at(-1).postprocess.ignore_small_boundaries, true);
    assert.equal(requests.at(-1).postprocess.max_gap_width_m, 0.3);
    resolvePrediction({...result(), postprocess:requests.at(-1).postprocess});
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    acceptSamRegion();
    assert.equal(serializeRoiRegion(roi.regions[0]).postprocess.max_hole_area_m2, 12.5);
    $("sam-fill-holes").checked = false;
    $("sam-ignore-boundaries").checked = false;
    undoRoiPoint();
    assert.equal($("sam-fill-holes").checked, true);
    assert.equal($("sam-ignore-boundaries").checked, true);
    assert.equal(Number($("sam-gap-width").value), 0.3);
    clearSam();
    handleSamKeyDown({key:"m",target:{},preventDefault(){}});
    addSamPoint(3,3,false);
    releaseSamMulti();
    assert.equal(sam.multi, false);
    assert.ok(sam.timer != null);
    clearSam();
    flushTimer(sam.timer);
    assert.equal(sam.busy, false);
  `, context);
  vm.runInContext(`
    roi.regions = []; addSamPoint(3,3,false); flushTimer(sam.timer); resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    handleSamKeyDown({key:"b",target:{closest:()=>true},preventDefault(){}});
    assert.equal(linePrompt.held, false);
    handleSamKeyDown({key:"B",target:{},preventDefault(){}});
    assert.equal(linePrompt.held, true);
    const promptCount = sam.prompts.length;
    const oldTx = viewer.tx;
    $("viewer-stage").listeners.pointerdown({button:0,clientX:3,clientY:3});
    $("viewer-stage").listeners.pointermove({clientX:7,clientY:7});
    assert.equal(sam.prompts.length, promptCount);
    assert.equal(viewer.tx, oldTx);
    assert.equal(linePrompt.stroke.points.length, 2);
    assert.equal(linePrompt.stroke.radius, undefined);
    $("viewer-stage").listeners.pointerup({button:0,clientX:7,clientY:7});
    assert.equal(sam.busy, true);
    assert.equal(requests.at(-1).scribbles.length, 1);
    assert.equal(requests.at(-1).model_id, $("sam-model").value);
    assert.equal(requests.at(-1).regions, undefined);
    assert.equal(requests.at(-1).strokes, undefined);
    assert.equal(requests.at(-1).points.length, promptCount);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(sam.scribbles.length, 1);
    undoRoiPoint();
    assert.equal(sam.scribbles.length, 0);
    handleSamKeyUp({key:"b"}); flushTimer(sam.timer);
    assert.equal(requests.at(-1).scribbles.length, 0);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    handleSamKeyDown({key:"b",target:{},preventDefault(){}});
    $("viewer-stage").listeners.pointerdown({button:0,clientX:3,clientY:3});
    $("viewer-stage").listeners.pointermove({clientX:7,clientY:7});
    // 先松开 B 提交划线，之后鼠标抬起不能误添加离散提示点。
    handleSamKeyUp({key:"b"});
    const beforeUp = sam.prompts.length;
    $("viewer-stage").listeners.pointerup({button:0,clientX:7,clientY:7});
    assert.equal(sam.prompts.length, beforeUp);
    assert.equal(linePrompt.held, false);
    assert.equal(sam.busy, true);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(sam.scribbles.length, 1);
    changeSamPostprocess(); flushTimer(sam.timer);
    assert.equal(requests.at(-1).scribbles.length, 1);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    acceptSamRegion();
    assert.equal(roi.regions[0].scribbles.length, 1);
    assert.equal(serializeRoiRegion(roi.regions[0]).scribble_count, 1);
    undoRoiPoint();
    assert.equal(sam.scribbles.length, 1);
    flushTimer(sam.timer); resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    handleSamKeyDown({key:"b",target:{},preventDefault(){}});
    $("viewer-stage").listeners.pointerdown({button:0,clientX:3,clientY:3});
    $("viewer-stage").listeners.pointerup({button:0,clientX:7,clientY:7});
    clearSam();
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(sam.result, null);
    assert.equal(sam.scribbles.length, 0);
    assert.equal(sam.busy, false);
    addSamPoint(3,3,false); flushTimer(sam.timer); resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    handleSamKeyDown({key:"b",target:{},preventDefault(){}});
    $("viewer-stage").listeners.pointerdown({button:0,clientX:3,clientY:3});
    $("viewer-stage").listeners.pointerup({button:0,clientX:7,clientY:7});
    handleSamKeyUp({key:"b"});
    handleSamContextMenu({preventDefault(){}});
    assert.equal(sam.confirmVersion, sam.version);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(roi.regions.length, 1);
    assert.equal(roi.regions[0].scribbles.length, 1);
    undoRoiPoint();
    assert.equal(sam.scribbles.length, 1);
    const restoredPrompts = sam.prompts.length;
    // 撤销线提示优先于撤销离散点，并由模型重新识别。
    undoRoiPoint();
    assert.equal(sam.scribbles.length, 0);
    assert.equal(sam.prompts.length, restoredPrompts);
    flushTimer(sam.timer);
    assert.equal(requests.at(-1).scribbles.length, 0);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext("clearSam();", context);
  vm.runInContext(`
    roi.regions = []; addSamPoint(3,3,false);
    sam.prompts.push({x:30,y:30,label:0});
    flushTimer(sam.timer); resolvePrediction({...result(), points:sam.prompts.map(p=>({...p}))});
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    handleSamKeyDown({key:"b",target:{},preventDefault(){}});
    $("viewer-stage").listeners.pointerdown({button:0,clientX:3,clientY:3});
    $("viewer-stage").listeners.pointerup({button:0,clientX:7,clientY:7});
    assert.equal(requests.at(-1).points[1].label, 0);
    handleSamKeyUp({key:"b"});
    rejectPrediction(new Error("线提示与排除点冲突"));
  `, context);
  await Promise.resolve();
  vm.runInContext(`
    assert.equal(sam.busy, false);
    assert.equal(sam.result, null);
    assert.equal(sam.scribbles.length, 1);
    assert.equal(sam.message, "线提示与排除点冲突");
    undoRoiPoint(); flushTimer(sam.timer);
    assert.equal(requests.at(-1).scribbles.length, 0);
    assert.equal(requests.at(-1).points[1].label, 0);
    resolvePrediction(result());
  `, context);
  await Promise.resolve();
  vm.runInContext("clearSam();", context);
  console.log("前端自动识别、多点快捷键、右键确认、B 划线模型提示／撤销／保留及过期响应测试通过。");
}
run().catch((err) => { console.error(err); process.exitCode = 1; });
