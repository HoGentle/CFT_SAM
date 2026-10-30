/* 多选、快捷键、编号重排、分组继承与过期操作保护。 */
"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const assert = require("node:assert/strict");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: "manual", classList: {toggle() {},add() {},remove() {},contains:()=>true}, scrollIntoView() {},
  });
  return elements.get(id);
}
const context = vm.createContext({assert,console,
  document:{getElementById:element,addEventListener(){}}, window:{addEventListener(){}},
});
const source = fs.readFileSync(path.join(__dirname,"../static/app.js"),"utf8");
vm.runInContext(source.replace(/\ninit\(\);\s*$/, ""),context);
vm.runInContext(`
  drawRoi = () => {};
  updateRoiPanel = () => {};
  refreshRoiArea = () => {};
  clearSam = () => {};
  toast = () => {};
  roi.editing = true;
  state.session = {preview_id:"test"};
  const region = (tag, x=0) => ({tag, source:"edited", points:[], hull:[
    {x,y:0,lon:113,lat:30},{x:x+8,y:0,lon:114,lat:30},
    {x:x+8,y:8,lon:114,lat:31},{x,y:8,lon:113,lat:31}
  ], holes:[]});
  roi.regions = [region("单区域")];
  let prevented=0, shortcut=null;
  const actualRun = runRoiEdit;
  runRoiEdit = action => {shortcut=action;};
  const key = {key:"H",target:{},preventDefault(){prevented++;}};
  handleRoiEditKeyDown(key);
  assert.equal(shortcut,null);
  assert.equal(prevented,0);
  toggleRoiSelection({x:4,y:4});
  assert.deepEqual([...roiEdit.selected],[0]);
  handleRoiEditKeyDown(key);
  assert.equal(prevented,0);
  roi.regions.push(region("第二个",20));
  toggleRoiSelection({x:24,y:4});
  assert.equal(roiEdit.selected.size,2);
  handleRoiEditKeyDown(key);
  assert.equal(shortcut,"merge");
  assert.equal(prevented,1);
  shortcut=null;
  handleRoiEditKeyDown({...key,target:{closest:()=>true}});
  assert.equal(shortcut,null);
  runRoiEdit = actualRun;

  // 区域 2、7、10 合并：8、9 减一，11 及之后减二。
  roi.regions=Array.from({length:12},(_,i)=>region(i+1,i*20));
  roi.groups=[{regions:[1,6,9]}, {regions:[7,8,10,11]}];
  replaceRoiRegions([1,6,9],[region("合并")]);
  assert.deepEqual(roi.regions.map(r=>r.tag),[1,"合并",3,4,5,6,8,9,11,12]);
  assert.deepEqual(roi.groups[0].regions,[1]);
  assert.deepEqual(roi.groups[1].regions,[6,7,8,9]);
  assert.deepEqual([...roiEdit.selected],[1]);
  replaceRoiRegions([1],[region("分割主区域"),region("新区域一"),region("新区域二")]);
  assert.equal(roi.regions.length,12);
  assert.equal(roi.regions[1].tag,"分割主区域");
  assert.equal(roi.regions[10].tag,"新区域一");
  assert.equal(roi.regions[11].tag,"新区域二");
  assert.deepEqual(roi.groups[0].regions,[1,10,11]);

  // 同一编号的多个独立部分都能选中，空洞不算命中。
  const multipart={...region("多部分"),parts:[region("一"),region("二",40)]};
  assert.equal(pointInRegion({x:44,y:4},multipart),true);
  const hollow=region("有洞");
  hollow.holes=[[{x:2,y:2},{x:6,y:2},{x:6,y:6},{x:2,y:6}]];
  assert.equal(pointInRegion({x:4,y:4},hollow),false);
  assert.equal(samExcluded.call(null).length,12);
`,context);

async function run() {
  vm.runInContext(`
    roi.regions=[region("目标")];
    roiEdit.selected=new Set([0]);
    beginRoiSplit();
    assert.equal(roiEdit.splitting,true);
    let payload=null, resolveOperation;
    api=async (url,body)=>{
      assert.equal(url,"/api/roi/edit"); payload=body;
      return new Promise(resolve=>{resolveOperation=resolve;});
    };
  `,context);
  const request=vm.runInContext("runRoiEdit('split')",context);
  vm.runInContext(`
    assert.equal(roiEdit.busy,true);
    assert.equal(payload.guide.length,0);
    resolveOperation({regions:[region("主区域"),region("追加")]});
  `,context);
  await request;
  vm.runInContext(`
    assert.equal(roiEdit.busy,false);
    assert.deepEqual(roi.regions.map(r=>r.tag),["主区域","追加"]);
    roiEdit.selected=new Set([0]); roiEdit.splitting=true;
    roiEdit.guide=[{x:2,y:1},{x:2,y:7}];
  `,context);
  const stale=vm.runInContext("runRoiEdit('split')",context);
  vm.runInContext(`
    assert.deepEqual(payload.guide,[[2,1],[2,7]]);
    roiEdit.version++; roiEdit.busy=false;
    roi.regions=[region("已换影像")];
    resolveOperation({regions:[region("过期结果"),region("不能加入")]});
  `,context);
  await stale;
  vm.runInContext(`assert.deepEqual(roi.regions.map(r=>r.tag),["已换影像"]);`,context);
  vm.runInContext(`
    roi.regions=[region("一"),region("二",20),region("保留",40)];
    roiEdit.selected=new Set([0,1]);
    document.getElementById("roi-merge-distance").value="5";
  `,context);
  const merge=vm.runInContext("runRoiEdit('merge')",context);
  vm.runInContext(`
    assert.equal(payload.merge_max_distance_m,5);
    assert.equal(payload.excluded_regions.length,1);
    assert.equal(payload.excluded_regions[0].parts[0].hull[0][0],40);
    resolveOperation({regions:[region("真正合并")]});
  `,context);
  await merge;
  vm.runInContext(`
    assert.deepEqual(roi.regions.map(r=>r.tag),["真正合并","保留"]);
    roiEdit.selected=new Set([0,1]);
    document.getElementById("roi-merge-distance").value="-1";
    api=async (url,body)=>{payload=body;throw new Error("距离过远无法合并");};
  `,context);
  await vm.runInContext("runRoiEdit('merge')",context);
  vm.runInContext(`
    assert.equal(payload.merge_max_distance_m,-1);
    assert.deepEqual(roi.regions.map(r=>r.tag),["真正合并","保留"]);
    assert.equal(roiEdit.busy,false);
    payload=null;
  `,context);
  for (const value of ["", "-0.5", "-2", "非法"]) {
    element("roi-merge-distance").value=value;
    await vm.runInContext("runRoiEdit('merge')",context);
    vm.runInContext("assert.equal(payload,null);",context);
  }
  console.log("区域多选、合并快捷键、编号重排、分割追加及过期操作保护验证通过。");
}
run().catch(error=>{console.error(error);process.exitCode=1;});
