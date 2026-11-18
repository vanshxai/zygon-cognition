/* 10 x 10 x 10 wireframe box: every token's exponent sits in its own cell. Only a picture for people; nothing in ZYGON computes with it.
   makeCube(canvas, tipEl, infoEl) -> {setData({cells:[{x,y,z,c,piece,id,ex,n,next:[[cell,share,piece]]}], tokens_known, shown})}  */
function makeCube(cv, tip, info) {
  const cx = cv.getContext('2d'); let W = 600, H = 480, DPR = 1, yaw = .65, pitch = .45, zoom = 1, auto = true, sel = -1, hov = -1, cells = [], byCell = {}, maxn = 1, meta = {}, showRel = true, drag = null, screen = [];
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'), vis = s => s.replace(/\n/g, '\\n').replace(/ /g, '\u2423');
  function fit() { const r = cv.getBoundingClientRect(); DPR = window.devicePixelRatio || 1; W = r.width; H = r.height; cv.width = W * DPR; cv.height = H * DPR; cx.setTransform(DPR, 0, 0, DPR, 0, 0); }
  addEventListener('resize', fit);
  const P = (x, y, z) => { const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), x1 = x * cy + z * sy, z1 = -x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp, f = 3.4 / (3.4 + z2), s = Math.min(W, H) * .3 * zoom; return [W / 2 + x1 * f * s, H / 2 - y1 * f * s, z2, f]; };
  const C = (i) => (i + .5) / 10 * 2 - 1, E = i => i / 10 * 2 - 1;
  function seg(a, b, col, lw, al) { const p = P(...a), q = P(...b); cx.beginPath(); cx.moveTo(p[0], p[1]); cx.lineTo(q[0], q[1]); cx.strokeStyle = col; cx.lineWidth = lw; cx.globalAlpha = al; cx.stroke(); cx.globalAlpha = 1; }
  function heat(n) { const t = Math.log(n + 1) / Math.log(maxn + 1); return `hsl(${200 - 190 * t},85%,${52 + 8 * t}%)`; }
  function draw() {
    if (auto) yaw += .004;
    cx.clearRect(0, 0, W, H);
    for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) {                    // the wireframe: every cell border
      const edge = (i === 0 || i === 10) && (j === 0 || j === 10), a = edge ? .22 : .07, c = '#9ad8ff', lw = 1;          // light blue, almost transparent: the exponents are what you look at
      seg([E(0), E(i), E(j)], [E(10), E(i), E(j)], c, lw, a); seg([E(i), E(0), E(j)], [E(i), E(10), E(j)], c, lw, a); seg([E(i), E(j), E(0)], [E(i), E(j), E(10)], c, lw, a);
    }
    if (showRel && sel >= 0 && byCell[sel]) { const s = byCell[sel]; s.next.forEach(([c, sh, pc], k) => { const t = byCell[c]; if (!t) return; const A = [C(s.x), C(s.y), C(s.z)], B = [C(t.x), C(t.y), C(t.z)], m = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2 + .18, (A[2] + B[2]) / 2];
      cx.beginPath(); for (let u = 0; u <= 20; u++) { const w = u / 20, q = [(1 - w) * (1 - w) * A[0] + 2 * w * (1 - w) * m[0] + w * w * B[0], (1 - w) * (1 - w) * A[1] + 2 * w * (1 - w) * m[1] + w * w * B[1], (1 - w) * (1 - w) * A[2] + 2 * w * (1 - w) * m[2] + w * w * B[2]], p = P(...q); u ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1]); }
      cx.strokeStyle = '#FFD166'; cx.lineWidth = 1 + 4 * sh; cx.globalAlpha = .85; cx.stroke(); cx.globalAlpha = 1; }); }
    screen = cells.map(c => { const p = P(C(c.x), C(c.y), C(c.z)); return {c, x: p[0], y: p[1], z: p[2], f: p[3]}; }).sort((a, b) => b.z - a.z);
    screen.forEach(({c, x, y, f}) => {
      const col = heat(c.n), on = c.c === sel, h = .1;                                   // the cell's own cube outline + a dot at its centre
      const o = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]].map(d => P(C(c.x) + d[0] * h, C(c.y) + d[1] * h, C(c.z) + d[2] * h));
      cx.strokeStyle = col; cx.lineWidth = on ? 2.4 : 1; cx.globalAlpha = on ? 1 : .8; cx.beginPath();
      [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => { cx.moveTo(o[a][0], o[a][1]); cx.lineTo(o[b][0], o[b][1]); }); cx.stroke(); cx.globalAlpha = 1;
      cx.beginPath(); cx.arc(x, y, Math.max(1.5, (2 + 3 * Math.log(c.n + 1) / Math.log(maxn + 1)) * f * zoom), 0, 7); cx.fillStyle = col; cx.fill();
      if (c.c === hov || on) { cx.strokeStyle = '#fff'; cx.lineWidth = 1.5; cx.stroke(); }
      if (on) { cx.fillStyle = '#FFD166'; cx.font = '12px Consolas'; cx.fillText(vis(c.piece), x + 8, y - 8); }
    });
    if (showRel && sel >= 0 && byCell[sel]) byCell[sel].next.forEach(([c, sh, pc]) => { const t = byCell[c]; if (!t) return; const p = P(C(t.x), C(t.y), C(t.z)); cx.fillStyle = '#FFD166'; cx.font = '11px Consolas'; cx.fillText(vis(pc) + ' ' + Math.round(sh * 100) + '%', p[0] + 6, p[1] - 6); });
    requestAnimationFrame(draw);
  }
  function near(e) { const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top; let b = -1, bd = 12; for (let k = screen.length - 1; k >= 0; k--) { const d = Math.hypot(screen[k].x - mx, screen[k].y - my); if (d < bd) { bd = d; b = screen[k].c.c; } } return b; }
  cv.addEventListener('mousedown', e => { drag = {x: e.clientX, y: e.clientY, moved: false}; auto = false; });
  addEventListener('mouseup', e => { if (drag && !drag.moved) { const b = near(e); sel = b; if (b >= 0 && byCell[b]) show(byCell[b]); } drag = null; });
  addEventListener('mousemove', e => { if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true; yaw += dx * .008; pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * .008)); drag.x = e.clientX; drag.y = e.clientY; }
    const r = cv.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) { hov = -1; tip.style.display = 'none'; return; }
    hov = near(e); if (hov >= 0 && byCell[hov]) { const c = byCell[hov]; tip.style.display = 'block'; tip.style.left = Math.min(e.clientX - r.left + 12, r.width - 230) + 'px'; tip.style.top = (e.clientY - r.top + 12) + 'px'; tip.innerHTML = `<b>${esc(vis(c.piece))}</b><br>${c.ex}<br>cell (${c.x}, ${c.y}, ${c.z}) | seen ${c.n}x`; } else tip.style.display = 'none'; });
  cv.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.max(.5, Math.min(3.5, zoom * (e.deltaY < 0 ? 1.1 : .91))); }, {passive: false});
  function show(c) { info.innerHTML = `<b>${esc(vis(c.piece))}</b> &nbsp; identity <b>${c.ex}</b> (id ${c.id}) &nbsp; cell (${c.x}, ${c.y}, ${c.z}) &nbsp; seen ${c.n} times<br>next words (the wheel after it): ` + (c.next.length ? c.next.map(([cc, sh, pc]) => `${esc(vis(pc))} ${Math.round(sh * 100)}%`).join(' | ') : 'none yet'); }
  fit(); requestAnimationFrame(draw);
  return {
    setData(d) { cells = d.cells; byCell = {}; maxn = 1; cells.forEach(c => { byCell[c.c] = c; maxn = Math.max(maxn, c.n); }); meta = d; },
    toggleSpin() { auto = !auto; return auto; }, toggleRel() { showRel = !showRel; return showRel; }, reset() { yaw = .65; pitch = .45; zoom = 1; }, fit,
    select(piece) { const c = cells.find(c => c.piece === piece); if (c) { sel = c.c; show(c); } }
  };
}
