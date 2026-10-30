/* 细节图视野、像素预算、取消请求和旧图内存释放。 */
"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const assert = require("node:assert/strict");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {style: {}, hidden: true,
    clientWidth: 1200, clientHeight: 800, removeAttribute(key) { delete this[key]; }});
  return elements.get(id);
}
let timers = new Map(), nextTimer = 0, created = [], revoked = [];
const context = vm.createContext({assert, console, AbortController, URLSearchParams,
  document: {getElementById: element, addEventListener() {}}, window: {devicePixelRatio: 2},
  setTimeout(fn) { timers.set(++nextTimer, fn); return nextTimer; },
  clearTimeout(id) { timers.delete(id); },
  URL: {createObjectURL() {const url = `测试图片${created.length}`; created.push(url); return url;},
    revokeObjectURL(url) {revoked.push(url);}},
  Image: class {decode() {return Promise.resolve();}},
  fetch: async () => ({ok: true, status: 200, blob: async () => ({})}),
});
const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
vm.runInContext(source.replace(/\ninit\(\);\s*$/, ""), context);
(async () => {
  await vm.runInContext(`(async () => {
    state.currentView = "input";
    state.session = {preview_id:"当前图", georef:{source_width:80000,source_height:40000}};
    viewer.naturalW=2000; viewer.naturalH=1000; viewer.scale=3; viewer.tx=-300; viewer.ty=-150;
    detail.ready=true;
    assert.equal(viewerMaxScale(),160);
    let view=detailView();
    assert.equal(view.x,100); assert.equal(view.y,50);
    assert.equal(view.w,400); assert.equal(view.h,800/3);
    assert.ok(view.width <= 2048 && view.height <= 2048 && view.width * view.height <= 2000000);
    const version=detail.version;
    await loadViewerDetail(view,version);
    assert.equal(document.getElementById("viewer-detail").hidden,false);
    assert.equal(document.getElementById("viewer-detail").style.left,"100px");
    await loadViewerDetail(view,version);
    const previous=detail.url;
    let aborted=false;
    detail.controller={abort(){aborted=true;}};
    scheduleViewerDetail();
    assert.equal(aborted,true);
    assert.ok(detail.version>version);
    clearViewerDetail();
    assert.equal(detail.url,null);
    assert.equal(document.getElementById("viewer-detail").hidden,true);
    // 解码期间发生视野切换，过期图片不得覆盖当前视图。
    Image.prototype.decode=async function(){detail.version++;};
    await loadViewerDetail(view,detail.version);
    assert.equal(detail.url,null);
    assert.equal(document.getElementById("viewer-detail").hidden,true);
    viewer.tx=5000;
    assert.equal(detailView(),null);
    viewer.tx=0; viewer.scale=0.5;
    assert.equal(detailView(),null);
    viewer.scale=3; state.currentView="class";
    assert.equal(detailView(),null);
  })()`, context);
  assert.deepEqual(revoked.sort(), created.sort());
  assert.equal(timers.size, 0);
  console.log("细节视野、像素预算、请求取消、过期响应与图片释放验证通过。");
})().catch(error => { console.error(error); process.exitCode = 1; });
