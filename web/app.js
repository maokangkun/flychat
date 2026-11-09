/* FlyChat unified page: chat + live 3D brain (relative paths only — subpath-safe) */
import * as THREE from './js/three.module.min.js';
import { OrbitControls } from './js/OrbitControls.js';

/* ---------------- chat ---------------- */
const chatEl = document.getElementById('chat');
const txtEl = document.getElementById('txt');
const sendEl = document.getElementById('send');
const footEl = document.getElementById('foot');
let busy = false;

function add(role, text, meta) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta'; m.textContent = meta;
    d.appendChild(m);
  }
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
}
add('fly', '(输入一句话 —— 果蝇脑每 token 增量仿真，回复需要几秒；'
           + '生成时左侧大脑会实时高亮放电神经元)');

function switchPane(chat) {
  document.body.classList.toggle('show-chat', chat);
  document.getElementById('tab-chat').classList.toggle('on', chat);
  document.getElementById('tab-brain').classList.toggle('on', !chat);
}
window.switchPane = switchPane;

async function pollStatus() {
  try {
    const s = await (await fetch('api/status')).json();
    footEl.textContent = `就绪 · 已对话 ${s.turns} 轮 · 累计 ${s.n_tokens_total} token`
      + ` · 全脑 ${s.n_neurons.toLocaleString()} 神经元 · 状态维度 ${s.state_dim}`;
  } catch (e) { footEl.textContent = '状态：服务未连接'; }
}

async function go() {
  const text = txtEl.value.trim();
  if (!text || busy) return;
  busy = true; sendEl.disabled = true; txtEl.value = '';
  add('user', text);
  if (window.matchMedia('(max-width: 820px)').matches) switchPane(false);
  footEl.textContent = '全脑仿真 + 增量生成中 …';
  try {
    const r = await fetch('api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const j = await r.json();
    if (j.ok) {
      add('fly', j.reply || '(空回复)',
          `上下文 ${j.n_context_tokens} tok · 生成 ${j.n_generated_tokens} tok · ${j.wall_s} s`);
    } else {
      add('fly', '(错误) ' + (j.error || 'unknown'));
    }
  } catch (e) { add('fly', '(请求失败) ' + e); }
  busy = false; sendEl.disabled = false;
  pollStatus();
}
document.getElementById('f').addEventListener('submit', e => { e.preventDefault(); go(); });
document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch('api/reset', {method:'POST'}); add('fly', '(对话已重置)');
  pollStatus();
});
pollStatus(); setInterval(pollStatus, 5000);

/* ---------------- 3D ---------------- */
const canvas = document.getElementById('c3d');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
const camera = new THREE.PerspectiveCamera(50, 1, 10, 2e6);
camera.position.set(45000, 25000, 90000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
function resize() {
  const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

let meta = null, activePts = null, activeRates = null;
let wireMesh = null, wireLoading = false, activeEdges = null, baseActiveSize = 6;

// round glowing sprite: WebGL point sprites are squares by default — use a
// radial-gradient texture so active dots render as soft circles, not big blocks
function makeGlowTexture(hardness) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,' + hardness + ')');
  grad.addColorStop(0.45, 'rgba(255,255,255,' + hardness * 0.55 + ')');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(cv);
  tex.needsUpdate = true;
  return tex;
}

async function loadMeta() {
  try {
    const r = await fetch('data/meta.json');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    meta = await r.json();
  } catch (e) {
    document.getElementById('n-neurons').textContent = '⚠ ' + e.message;
    document.getElementById('n-neurons').style.color = '#ff8866';
    return;
  }
  document.getElementById('n-neurons').textContent = meta.nNeurons.toLocaleString();
  // soma cloud (base) — bigger/brighter for mobile visibility
  const n = meta.soma.length;
  const pos = new Float32Array(n * 3);
  let nValid = 0;
  for (let i = 0; i < n; i++) {
    const s = meta.soma[i];
    if (s) { pos[nValid*3] = s[0]; pos[nValid*3+1] = s[1]; pos[nValid*3+2] = s[2]; nValid++; }
  }
  const isSmall = Math.min(screen.width, screen.height) < 700;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos.slice(0, nValid * 3), 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({
    color: 0x7c8db5, size: isSmall ? 3.4 : 2.4, sizeAttenuation: false })));
  // active overlay (rebuilt from polling)
  const g2 = new THREE.BufferGeometry();
  g2.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g2.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g2.setDrawRange(0, 0);
  baseActiveSize = isSmall ? 7.5 : 6;
  const texCore = makeGlowTexture(1.0), texHalo = makeGlowTexture(0.9);
  const matCore = new THREE.PointsMaterial({
    size: baseActiveSize, sizeAttenuation: false, vertexColors: true,
    map: texCore, transparent: true, opacity: 1.0, depthWrite: false,
    alphaTest: 0.02 });
  activePts = new THREE.Points(g2, matCore);
  activePts.frustumCulled = false;
  scene.add(activePts);
  // additive halo pass (same buffer, slightly bigger, faint) for glow visibility
  const halo = new THREE.Points(g2, new THREE.PointsMaterial({
    size: baseActiveSize * 1.7, sizeAttenuation: false, vertexColors: true,
    map: texHalo, transparent: true, opacity: 0.22, depthWrite: false,
    blending: THREE.AdditiveBlending }));
  halo.frustumCulled = false;
  scene.add(halo);
  activePts.userData.halo = halo;

  // pulsing synapse edges between active neurons (fill from /api/activity)
  const EDGE_MAX = 1600;
  const ge = new THREE.BufferGeometry();
  ge.setAttribute('position', new THREE.BufferAttribute(new Float32Array(EDGE_MAX * 6), 3));
  ge.setAttribute('color', new THREE.BufferAttribute(new Float32Array(EDGE_MAX * 6), 3));
  ge.setDrawRange(0, 0);
  activeEdges = new THREE.LineSegments(ge, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: isSmall ? 0.7 : 0.6,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  activeEdges.frustumCulled = false;
  scene.add(activeEdges);
}

async function pollActivity() {
  if (!meta || !activePts) return;
  try {
    const a = await (await fetch('api/activity')).json();
    if (!a.active) return;
    const pos = activePts.geometry.attributes.position.array;
    const col = activePts.geometry.attributes.color.array;
    let k = 0;
    const rates = a.active.map(x => x[1]);
    const rmax = Math.max(...rates, 1e-9);
    for (const [vi, r] of a.active) {
      const s = meta.soma[vi];
      if (!s) continue;
      pos[k*3] = s[0]; pos[k*3+1] = s[1]; pos[k*3+2] = s[2];
      const t = Math.pow(r / rmax, 0.3);
      col[k*3]   = 0.55 + 0.45 * t;
      col[k*3+1] = 0.30 * (1 - t) + 0.06 * t;
      col[k*3+2] = 0.12 * (1 - t) + 0.02 * t;
      k++;
    }
    activePts.geometry.setDrawRange(0, k);
    activePts.geometry.attributes.position.needsUpdate = true;
    activePts.geometry.attributes.color.needsUpdate = true;
    // edges among active neurons
    if (activeEdges && a.edges && a.edges.length) {
      const ep = activeEdges.geometry.attributes.position.array;
      const ec = activeEdges.geometry.attributes.color.array;
      const soma = meta.soma;
      let e = 0;
      for (const [va, vb, w] of a.edges) {
        const sa = soma[va], sb = soma[vb];
        if (!sa || !sb || e >= 1600) continue;
        ep[e*6]   = sa[0]; ep[e*6+1] = sa[1]; ep[e*6+2] = sa[2];
        ep[e*6+3] = sb[0]; ep[e*6+4] = sb[1]; ep[e*6+5] = sb[2];
        // fire gradient: hot core -> dim ember by weight
        const t = Math.pow(w, 0.5);
        for (const off of [0, 3]) {
          ec[e*6+off]   = 0.55 + 0.45 * t;
          ec[e*6+off+1] = 0.25 + 0.30 * t;
          ec[e*6+off+2] = 0.15 * t;
        }
        e++;
      }
      activeEdges.geometry.setDrawRange(0, e * 2);
      activeEdges.geometry.attributes.position.needsUpdate = true;
      activeEdges.geometry.attributes.color.needsUpdate = true;
    } else if (activeEdges) {
      activeEdges.geometry.setDrawRange(0, 0);
    }
    document.getElementById('n-act').textContent = a.n_active_brain.toLocaleString();
    window.__viz = {ts: performance.now(), seq: a.seq, nActive: k,
                    nEdges: (activeEdges ? activeEdges.geometry.drawRange.count / 2 : 0)};
  } catch (e) { /* transient */
    window.__viz = {ts: performance.now(), seq: -1, nActive: 0, nEdges: 0, liveErr: String(e)};
  }
}setInterval(pollActivity, 450);

/* optional LOD2 wireframe */
document.getElementById('btn-wire').addEventListener('click', async (ev) => {
  if (wireMesh || wireLoading) return;
  wireLoading = true;
  ev.target.textContent = '线框加载中 …';
  try {
    const chunks = [];
    for (let c = 0; ; c++) {
      const id = String(c).padStart(3, '0');
      let buf;
      try {
        const r = await fetch(`data/lod2/chunk_${id}.bin`);
        if (!r.ok) break;
        buf = await r.arrayBuffer();
      } catch (e) { break; }
      const dv = new DataView(buf);
      const n = dv.getUint32(0, true);
      const nNodes = dv.getUint32(4, true);
      const nSegs = dv.getUint32(8, true);
      const segOff = new Uint32Array(buf, 12 + 4 * (n + 1), n + 1);
      const posOff = 12 + 8 * (n + 1);
      const positions = new Float32Array(buf, posOff, nNodes * 3);
      const segments = new Uint32Array(buf, posOff + nNodes * 12, nSegs * 2);
      chunks.push({segOff, positions, segments, n});
    }
    let total = 0;
    for (const ch of chunks) total += ch.segOff[ch.n];
    const pos = new Float32Array(total * 6);
    let o = 0;
    for (const ch of chunks) {
      for (let i = 0; i < ch.n; i++) {
        const s0 = ch.segOff[i] * 2, s1 = ch.segOff[i + 1] * 2;
        for (let s = s0; s < s1; s += 2) {
          const a = ch.segments[s] * 3, b = ch.segments[s + 1] * 3;
          pos.set(ch.positions.subarray(a, a + 3), o); o += 3;
          pos.set(ch.positions.subarray(b, b + 3), o); o += 3;
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    wireMesh = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      color: 0x2b3a5c, transparent: true, opacity: 0.5 }));
    scene.add(wireMesh);
    ev.target.textContent = '全脑线框已加载';
    ev.target.classList.add('on');
  } catch (e) {
    ev.target.textContent = '线框加载失败';
  }
  wireLoading = false;
});

loadMeta().catch(e => { footEl.textContent = '3D 数据加载失败: ' + e; });
(function renderLoop() {
  requestAnimationFrame(renderLoop);
  controls.update();
  // heartbeat pulse on active neurons
  if (activePts && activePts.material) {
    const pulse = 1 + 0.5 * Math.sin(performance.now() / 160);
    activePts.material.size = baseActiveSize * pulse;
    const halo = activePts.userData.halo;
    if (halo) halo.material.size = baseActiveSize * 1.7 * pulse;
  }
  renderer.render(scene, camera);
})();

