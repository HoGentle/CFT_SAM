/* 手工区域的内部点、预览、完成、撤销及删除回归验证。 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const context = vm.createContext({
  assert, console,
  document: { addEventListener() {} },
  window: { addEventListener() {} },
});
const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
vm.runInContext(source.replace(/\ninit\(\);\s*$/, ""), context);
vm.runInContext(`
  const elements = new Map();
  document.getElementById = (id) => {
    if (!elements.has(id)) elements.set(id, {
      value: "manual", width: 400, height: 400, src: "预览图", style: {display: "block"},
      classList: { contains: () => false, toggle() {} },
      scrollIntoView() { this.scrolled = true; },
      getContext: () => ctx,
    });
    return elements.get(id);
  };
  let traced = [], transforms = [], labels = [], drawStyle = {};
  const ctx = new Proxy({}, { get: (_, name) => (...args) => {
    if (name === "moveTo") traced = [args];
    if (name === "lineTo") traced.push(args);
    if (name === "setTransform") transforms.push(args);
    if (name === "fillText") labels.push(args);
  }, set: (_, name, value) => { drawStyle[name] = value; return true; } });
  updateSamControls = () => {};
  updateRoiPanel = () => {};
  refreshRoiArea = () => {};
  toast = () => {};
  clearSam = () => {};
  previewLonLat = (x, y) => ({ lon: x, lat: y });
  viewer.scale = 1;
  state.session = {has_input_preview: true, georef: {}};
  document.querySelectorAll = () => [$("card-input"), $("card-validation"), $("card-params"), $("card-regions")];
  setRoiEditing(true);
  assert.equal($("card-input").open, false);
  assert.equal($("card-validation").open, false);
  assert.equal($("card-params").open, false);
  assert.equal($("card-regions").open, true);
  assert.equal($("card-regions").scrolled, true);
  roi.editing = true;
  const corners = [[35, 70], [242, 95], [255, 215], [60, 180]];
  corners.forEach(([x, y]) => addRoiPoint(x, y));
  addRoiPoint(156, 124);
  const boundary = manualBoundary(roi.current);
  assert.equal(boundary.length, 5);
  const centerIndex = boundary.findIndex((p) => p.x === 156);
  assert.equal(boundary[(centerIndex + 4) % 5].x, 35);
  assert.equal(boundary[(centerIndex + 1) % 5].x, 242);
  assert.deepEqual(traced, boundary.map((p) => [p.x, p.y]));
  assert.equal(pointInPolygon({x:156, y:100}, boundary), false);
  assert.equal(pointInPolygon({x:156, y:150}, boundary), true);
  finalizeCurrentRegion();
  assert.deepEqual(roi.regions[0].hull, boundary);
  assert.deepEqual(serializeRoiRegion(roi.regions[0]).hull, boundary.map((p) => [p.lon, p.lat]));
  undoRoiPoint();
  assert.equal(roi.regions.length, 0);
  assert.equal(roi.current.length, 0);
  undoRoiPoint();
  assert.equal(roi.current.length, 0);
  corners.forEach(([x, y]) => addRoiPoint(x, y));
  addRoiPoint(156, 124);
  addRoiPoint(156, 124);
  assert.equal(roi.current.length, 4);
  roi.current = [{x:0,y:0}, {x:30,y:0}, {x:60,y:0}];
  finalizeCurrentRegion();
  assert.equal(roi.regions.length, 0);

  // 快捷键与撤销按钮复用同一逻辑，输入框中的撤销由浏览器处理。
  roi.current = [{x:10,y:10}, {x:50,y:10}];
  let prevented = 0;
  const undoKey = {key:"z", ctrlKey:true, target:{}, preventDefault() {prevented++;}};
  handleRoiUndoKeyDown(undoKey);
  assert.equal(roi.current.length, 1);
  handleRoiUndoKeyDown({...undoKey, target:{closest:()=>true}});
  handleRoiUndoKeyDown({...undoKey, shiftKey:true});
  assert.equal(roi.current.length, 1);
  assert.equal(prevented, 1);
  roi.current = [];
  roi.regions = [{source:"manual", hull:corners.map(([x,y])=>({x,y})), points:corners.map(([x,y])=>({x,y}))}];
  handleRoiUndoKeyDown(undoKey);
  assert.equal(roi.regions.length, 0);
  assert.equal(roi.current.length, 0);

  // 连续撤销直接依次删除区域，并同步分组编号，不恢复进行中的标点。
  const completed = {source:"manual", hull:corners.map(([x,y])=>({x,y})), points:corners.map(([x,y])=>({x,y}))};
  roi.regions = [completed, {...completed}];
  roi.groups = [{regions:[0,1]}];
  handleRoiUndoKeyDown(undoKey);
  assert.equal(roi.regions.length, 1);
  assert.deepEqual(roi.groups[0].regions, [0]);
  assert.equal(roi.current.length, 0);
  handleRoiUndoKeyDown(undoKey);
  assert.equal(roi.regions.length, 0);
  assert.equal(roi.current.length, 0);

  // 放大只改变绘制变换，画布始终匹配视口与设备像素比。
  roi.regions = [completed];
  roi.current = [];
  $("viewer-stage").clientWidth = 640;
  $("viewer-stage").clientHeight = 480;
  window.devicePixelRatio = 2;
  viewer.scale = 8;
  viewer.tx = 31; viewer.ty = -17;
  drawRoi();
  assert.equal($("roi-canvas").width, 1280);
  assert.equal($("roi-canvas").height, 960);
  assert.deepEqual(transforms.at(-1), [16,0,0,16,62,-34]);
  assert.ok(drawStyle.font.includes("1.875px"));
  assert.equal(labels.at(-1)[0], "区域1");
  viewer.scale = 16;
  drawRoi();
  assert.equal($("roi-canvas").width, 1280);
  assert.deepEqual(transforms.at(-1), [32,0,0,32,62,-34]);
  window.confirm = () => false;
  clearAllRoi();
  assert.equal(roi.regions.length, 1);
  window.confirm = () => true;
  clearAllRoi();
  assert.equal(roi.regions.length, 0);
  assert.equal(roi.current.length, 0);
`, context);
console.log("手工区域、快捷键、编辑跳转、清空确认及高清画布验证通过。");

vm.runInContext(`
  let exportClicks = 0;
  let exportRequest = null;
  let exportMessage = "";
  toast = (message) => { exportMessage = message; };
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/api/roi/export-shp");
    exportClicks++;
    exportRequest = JSON.parse(options.body);
    return {ok: true, json: async () => ({cancelled: false, directory: "测试目录"})};
  };
`, context);
(async () => {
  await vm.runInContext("exportRoiShp()", context);
  vm.runInContext(`
    assert.equal(exportRequest, null);
    assert.equal(exportClicks, 0);
    roi.current = [{x:5,y:5}];
    roi.regions = [{source: "manual", points: [], hull: [
      {lon:113, lat:30}, {lon:114, lat:30}, {lon:114, lat:31}
    ]}];
  `, context);
  await vm.runInContext("exportRoiShp()", context);
  vm.runInContext(`
    assert.equal(exportRequest.regions.length, 1);
    assert.equal(exportRequest.regions[0].source, "manual");
    assert.equal(exportRequest.regions[0].hull[0][0], 113);
    assert.equal(exportClicks, 1);
    assert.equal($("btn-roi-export").disabled, false);
    assert.ok(exportMessage.includes("当前编辑区域需右键完成"));
    fetch = async () => ({ok: true, json: async () => ({cancelled: true})});
  `, context);
  await vm.runInContext("exportRoiShp()", context);
  vm.runInContext(`
    assert.equal(exportClicks, 1);
    assert.ok(exportMessage.includes("测试目录"));
    assert.equal($("btn-roi-export").disabled, false);
    fetch = async () => ({ok: false, json: async () => ({error: "测试导出失败"})});
  `, context);
  await vm.runInContext("exportRoiShp()", context);
  vm.runInContext(`
    assert.equal(exportMessage, "测试导出失败");
    assert.equal($("btn-roi-export").disabled, false);
  `, context);
  console.log("区域保存按钮的空区域提示、保存、取消及失败恢复验证通过。");
})().catch((error) => { console.error(error); process.exitCode = 1; });
