import { Fragment, useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/ws'
const DEFAULT = { id: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5-1.5B' }
const VIEWS = ['output', 'attention', 'activations', 'logitlens', 'spot'] as const
type View = typeof VIEWS[number]
const DEFAULT_DATASET = 'def add(a, b):\n    return a + b\nfor i in range(10):\n    print(i)\nx = [3, 1, 2]\nx.sort()'

type Logit = { token: string; prob: number }
type Spot = { layers: number; modules: string[]; grid: number[][] }
type ModelData = {
  output: string; frames: number[][][]; act: number[][] | null; logit: Logit[] | null;
  spot: Spot | null; perhead: { layer: number; data: number[][] } | null; count: number; busy: boolean; loading: boolean
}
const empty = (): ModelData => ({ output: '', frames: [], act: null, logit: null, spot: null, perhead: null, count: 0, busy: false, loading: false })

type Tile = { id: number; model: string; tabs: View[]; active: number; h: number }
type Col = { id: number; w: number; tiles: Tile[] }

function cellColor(v: number, max: number) {
  const t = max > 0 ? v / max : 0
  return `rgb(${Math.round(0x20 + (0x00 - 0x20) * t)},${Math.round(0x1d + (0x7a - 0x1d) * t)},${Math.round(0x1d + (0xff - 0x1d) * t)})`
}
function ampColor(v: number, max: number) {
  const t = max > 0 ? v / max : 0
  return `rgb(${Math.round(0x20 + (0xe0 - 0x20) * t)},${Math.round(0x1d + (0xa8 - 0x1d) * t)},${Math.round(0x1d + (0x5e - 0x1d) * t)})`
}
function Grid({ rows, cols, rowH, onRow }: { rows: number[][]; cols: number; rowH: number; onRow?: (i: number) => void }) {
  const max = Math.max(...rows.flat())
  return (
    <div style={{ display: 'grid', gridTemplateRows: `repeat(${rows.length}, ${rowH}px)`, gap: 1 }}>
      {rows.map((row, i) => (
        <div key={i} onClick={onRow ? () => onRow(i) : undefined} style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 1, cursor: onRow ? 'pointer' : 'default' }}>
          {Array.from({ length: cols }, (_, k) => <div key={k} style={{ background: k < row.length ? cellColor(row[k], max) : 'var(--bg-2)' }} />)}
        </div>
      ))}
    </div>
  )
}
function SpotGrid({ grid, modules }: { grid: number[][]; modules: string[] }) {
  const flat = grid.flat(); const max = Math.max(...flat)
  const sorted = [...flat].sort((a, b) => b - a)
  const thr = sorted[Math.max(0, Math.floor(sorted.length * 0.05) - 1)] ?? Infinity
  return (
    <div style={{ display: 'grid', gridTemplateRows: `repeat(${grid.length}, 9px)`, gap: 1 }}>
      {grid.map((row, l) => (
        <div key={l} style={{ display: 'grid', gridTemplateColumns: `repeat(${modules.length}, 1fr)`, gap: 1 }}>
          {row.map((v, c) => <div key={c} style={{ background: ampColor(v, max), outline: v >= thr ? '1px solid #FDFCFC' : 'none' }} />)}
        </div>
      ))}
    </div>
  )
}

const hint = { color: 'var(--text-2)' as const }
const iconBtn = { background: 'transparent', border: 'none', color: 'var(--text-2)', cursor: 'pointer', padding: '0 5px', fontSize: 11 }

export default function App() {
  const [prompt, setPrompt] = useState('write a quicksort in python')
  const [sync, setSync] = useState(true)
  const [explorerOpen, setExplorerOpen] = useState(true)
  const [focusModel, setFocusModel] = useState(DEFAULT.id)
  const [open, setOpen] = useState<{ id: string; label: string }[]>([DEFAULT])
  const [catalog, setCatalog] = useState<{ id: string; label: string }[]>([])
  const [data, setData] = useState<Record<string, ModelData>>({ [DEFAULT.id]: empty() })
  const [cols, setCols] = useState<Col[]>([{ id: 1, w: 1, tiles: [{ id: 1, model: DEFAULT.id, tabs: ['output', 'attention'], active: 0, h: 1 }] }])
  const [ds, setDs] = useState(DEFAULT_DATASET)
  const [layer, setLayer] = useState(0)
  const [dragging, setDragging] = useState(false)
  const [overTile, setOverTile] = useState<number | null>(null)
  const nextId = useRef(2)
  const drag = useRef<{ tid: number; idx: number } | null>(null)
  const sockets = useRef<Record<string, WebSocket>>({})
  const rowRef = useRef<HTMLDivElement>(null)

  function patch(mid: string, f: (d: ModelData) => ModelData) { setData((all) => ({ ...all, [mid]: f(all[mid] ?? empty()) })) }
  function handleMessage(e: MessageEvent) {
    const m = JSON.parse(e.data)
    if (m.type === 'catalog') { setCatalog(m.models); return }
    const mid = m.model
    if (m.type === 'loading') { patch(mid, (d) => ({ ...d, loading: true })); return }
    if (m.type === 'opened') { patch(mid, (d) => ({ ...d, loading: false })); return }
    if (m.type === 'token') patch(mid, (d) => ({ ...d, output: d.output + m.text, count: d.count + 1 }))
    else if (m.type === 'attention') patch(mid, (d) => ({ ...d, frames: [...d.frames, m.data] }))
    else if (m.type === 'activation') patch(mid, (d) => ({ ...d, act: m.data }))
    else if (m.type === 'logitlens') patch(mid, (d) => ({ ...d, logit: m.layers }))
    else if (m.type === 'spotmap') patch(mid, (d) => ({ ...d, spot: { layers: m.layers, modules: m.modules, grid: m.grid } }))
    else if (m.type === 'perhead') patch(mid, (d) => ({ ...d, perhead: { layer: m.layer, data: m.data } }))
    else if (m.type === 'done') patch(mid, (d) => ({ ...d, busy: false }))
  }
  function socket(mid: string) {
    let s = sockets.current[mid]
    if (!s || s.readyState > WebSocket.OPEN) { s = new WebSocket(WS_URL); sockets.current[mid] = s; s.onmessage = handleMessage }
    return s
  }
  function sendTo(mid: string, msg: object) {
    const s = socket(mid); const m = { ...msg, model: mid }
    if (s.readyState === WebSocket.OPEN) s.send(JSON.stringify(m))
    else s.addEventListener('open', () => s.send(JSON.stringify(m)), { once: true })
  }
  useEffect(() => { sendTo(DEFAULT.id, { type: 'catalog' }) }, [])

  const focused = () => (open.some((m) => m.id === focusModel) ? focusModel : (open[0]?.id ?? DEFAULT.id))
  const targets = () => (sync ? open.map((m) => m.id) : [focused()])
  function send() {
    if (!prompt.trim()) return
    for (const mid of targets()) { patch(mid, () => ({ ...empty(), busy: true })); sendTo(mid, { type: 'generate', prompt, max_tokens: 64, probes: ['attention', 'activation', 'logitlens'] }) }
  }
  function stop() { for (const mid of targets()) sockets.current[mid]?.send(JSON.stringify({ type: 'stop', model: mid })) }
  function openModel(id: string, label: string) { if (open.some((m) => m.id === id)) return; setOpen((o) => [...o, { id, label }]); patch(id, () => ({ ...empty(), loading: true })); sendTo(id, { type: 'open' }) }
  const anyBusy = Object.values(data).some((d) => d.busy)

  // ---- 2-level layout ops (cols × tiles) ----
  function tileCount() { return cols.reduce((n, c) => n + c.tiles.length, 0) }
  function updateTile(tid: number, f: (t: Tile) => Tile) { setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.id === tid ? f(t) : t)) }))) }
  function setActive(tid: number, idx: number) { updateTile(tid, (t) => ({ ...t, active: idx })) }
  function setTileModel(tid: number, model: string) { updateTile(tid, (t) => ({ ...t, model })) }
  function addTab(tid: number) { updateTile(tid, (t) => ({ ...t, tabs: [...t.tabs, VIEWS.find((v) => !t.tabs.includes(v)) ?? 'output'], active: t.tabs.length })) }
  function newTile(model: string, view: View): Tile { return { id: nextId.current++, model, tabs: [view], active: 0, h: 1 } }
  function splitRight(tid: number) {
    setCols((cs) => {
      const ci = cs.findIndex((c) => c.tiles.some((t) => t.id === tid)); const tile = cs[ci].tiles.find((t) => t.id === tid)!
      return [...cs.slice(0, ci + 1), { id: nextId.current++, w: 1, tiles: [newTile(tile.model, tile.tabs[tile.active])] }, ...cs.slice(ci + 1)]
    })
  }
  function splitDown(tid: number) {
    setCols((cs) => cs.map((c) => {
      const ti = c.tiles.findIndex((t) => t.id === tid); if (ti < 0) return c
      const tile = c.tiles[ti]; const nt = newTile(tile.model, tile.tabs[tile.active])
      return { ...c, tiles: [...c.tiles.slice(0, ti + 1), nt, ...c.tiles.slice(ti + 1)] }
    }))
  }
  function pruneClose(tid: number) {
    if (tileCount() <= 1) return
    setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.filter((t) => t.id !== tid) })).filter((c) => c.tiles.length > 0))
  }
  function closeTab(tid: number, idx: number) {
    let emptied = false
    setCols((cs) => cs.map((c) => ({
      ...c, tiles: c.tiles.map((t) => {
        if (t.id !== tid) return t
        const tabs = t.tabs.filter((_, i) => i !== idx)
        if (tabs.length === 0) emptied = true
        return { ...t, tabs, active: Math.max(0, idx <= t.active ? t.active - 1 : t.active) }
      }).filter((t) => t.tabs.length > 0),
    })).filter((c) => c.tiles.length > 0))
    void emptied
  }
  function moveTab(toTid: number, toIdx?: number) {
    const d = drag.current; drag.current = null; setDragging(false); setOverTile(null)
    if (!d || (d.tid === toTid && toIdx === undefined)) return
    setCols((cs) => {
      let view: View | undefined
      cs.forEach((c) => c.tiles.forEach((t) => { if (t.id === d.tid) view = t.tabs[d.idx] }))
      if (view == null) return cs
      let next = cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.id === d.tid ? { ...t, tabs: t.tabs.filter((_, i) => i !== d.idx), active: Math.max(0, d.idx <= t.active ? t.active - 1 : t.active) } : t)) }))
      next = next.map((c) => ({ ...c, tiles: c.tiles.map((t) => {
        if (t.id !== toTid || t.tabs.includes(view!)) return t
        const tabs = [...t.tabs]; const at = toIdx ?? tabs.length; tabs.splice(at, 0, view!); return { ...t, tabs, active: Math.min(at, tabs.length - 1) }
      }) }))
      next = next.map((c) => ({ ...c, tiles: c.tiles.filter((t) => t.tabs.length > 0) })).filter((c) => c.tiles.length > 0)
      return next.length ? next : cs
    })
  }
  function resizeCols(ci: number, e: React.MouseEvent) {
    e.preventDefault(); const x0 = e.clientX; const w = rowRef.current?.clientWidth ?? 1
    const total = cols.reduce((s, c) => s + c.w, 0); const a0 = cols[ci].w, b0 = cols[ci + 1].w
    const move = (ev: MouseEvent) => { const dd = ((ev.clientX - x0) / w) * total; setCols((cs) => cs.map((c, i) => i === ci ? { ...c, w: Math.max(0.15, a0 + dd) } : i === ci + 1 ? { ...c, w: Math.max(0.15, b0 - dd) } : c)) }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
  function resizeTiles(ci: number, ti: number, e: React.MouseEvent) {
    e.preventDefault(); const y0 = e.clientY; const H = (e.currentTarget.parentElement as HTMLElement)?.clientHeight ?? 1
    const tiles = cols[ci].tiles; const total = tiles.reduce((s, t) => s + t.h, 0); const a0 = tiles[ti].h, b0 = tiles[ti + 1].h
    const move = (ev: MouseEvent) => { const dd = ((ev.clientY - y0) / H) * total; setCols((cs) => cs.map((c, i) => i !== ci ? c : { ...c, tiles: c.tiles.map((t, j) => j === ti ? { ...t, h: Math.max(0.12, a0 + dd) } : j === ti + 1 ? { ...t, h: Math.max(0.12, b0 - dd) } : t) })) }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }

  function viewBody(mid: string, view: View) {
    const d = data[mid] ?? empty()
    if (d.loading) return <span style={hint}>[ loading {open.find((m) => m.id === mid)?.label ?? 'model'}… ]</span>
    if (view === 'output') return <div style={{ whiteSpace: 'pre-wrap' }}>{d.output || <span style={hint}>[ run a prompt below ]</span>}{d.busy && <span style={{ color: 'var(--accent)' }}>▌</span>}</div>
    if (view === 'attention') {
      const last = d.frames.at(-1) ?? null
      if (!last) return <span style={hint}>[ attention while generating ]</span>
      const tri = d.frames.map((f) => f[layer] ?? []); const triC = tri.at(-1)?.length ?? 0
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 8 }}>all layers · current token · {last.length}×{last[0].length}<span style={hint}> · click a layer ↓</span></div>
        <Grid rows={last} cols={last[0].length} rowH={6} onRow={(i) => setLayer(i)} />
        <div style={{ color: 'var(--text-1)', margin: '14px 0 8px' }}>layer {layer} · query × kv · causal<button onClick={() => sendTo(mid, { type: 'drilldown', layer })} style={{ marginLeft: 10, background: 'transparent', border: '1px solid var(--line-strong)', borderRadius: 4, color: 'var(--text-2)', cursor: 'pointer', padding: '1px 8px' }}>[heads]</button></div>
        <Grid rows={tri} cols={triC} rowH={Math.max(2, Math.min(8, Math.floor(200 / Math.max(1, tri.length))))} />
        {d.perhead && <div style={{ marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 10 }}><div style={{ color: 'var(--text-1)', marginBottom: 6 }}>layer {d.perhead.layer} · per-head</div><Grid rows={d.perhead.data} cols={d.perhead.data[0].length} rowH={13} /></div>}
      </>)
    }
    if (view === 'activations') return d.act ? (<><div style={{ color: 'var(--text-1)', marginBottom: 8 }}>layer × module · output norm<span style={hint}> (self_attn · mlp)</span></div><Grid rows={d.act} cols={d.act[0].length} rowH={7} /></>) : <span style={hint}>[ activations while generating ]</span>
    if (view === 'logitlens') return d.logit ? (<><div style={{ color: 'var(--text-1)', marginBottom: 8 }}>layer → top-1 · {d.logit.length} layers</div><div style={{ display: 'grid', gap: 3 }}>{d.logit.map((r, l) => <div key={l} style={{ display: 'grid', gridTemplateColumns: '34px 76px 1fr', gap: 8, alignItems: 'center' }}><span style={hint}>L{l}</span><span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.token}</span><div style={{ background: 'var(--bg-2)', borderRadius: 2, height: 9 }}><div style={{ height: 9, width: `${Math.round(r.prob * 100)}%`, background: 'var(--accent)', borderRadius: 2 }} /></div></div>)}</div></>) : <span style={hint}>[ logit lens while generating ]</span>
    return (<>
      <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>dataset → grad×param<span style={hint}> · top cells = spot</span></div>
      <textarea value={ds} onChange={(e) => setDs(e.target.value)} rows={3} spellCheck={false} style={{ width: '100%', background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '6px 8px', outline: 'none', resize: 'vertical', marginBottom: 6 }} />
      <button onClick={() => sendTo(mid, { type: 'spot', examples: ds.split('\n').map((s) => s.trim()).filter(Boolean) })} style={{ background: 'transparent', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '4px 12px', color: 'var(--accent)', cursor: 'pointer', marginBottom: 10 }}>[compute spot]</button>
      {d.spot && <><div style={hint}>{d.spot.layers} × {d.spot.modules.length} · |grad×param|</div><div style={{ marginTop: 6 }}><SpotGrid grid={d.spot.grid} modules={d.spot.modules} /></div></>}
    </>)
  }

  const closed = catalog.filter((c) => !open.some((o) => o.id === c.id))
  const multi = tileCount() > 1

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 12px', borderBottom: '1px solid var(--line)', background: 'var(--bg-1)' }}>
        <button onClick={() => setExplorerOpen((v) => !v)} title="explorer" style={{ background: 'transparent', border: 'none', color: 'var(--text-1)', cursor: 'pointer', padding: 0 }}>[≡]</button>
        <strong>parametic-studio</strong>
        <div style={{ display: 'flex', gap: 6 }}>
          {open.map((m) => <span key={m.id} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'var(--bg-2)', border: '1px solid var(--line)' }}>{data[m.id]?.loading ? <span style={hint}>⟳ </span> : data[m.id]?.busy ? <span style={{ color: 'var(--live)' }}>● </span> : ''}{m.label}</span>)}
          {closed.length > 0 && (
            <select value="" onChange={(e) => { const c = catalog.find((x) => x.id === e.target.value); if (c) openModel(c.id, c.label) }} style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' }}>
              <option value="">+ model</option>
              {closed.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
            </select>
          )}
        </div>
        <button onClick={() => setSync((v) => !v)} title="broadcast input to all open models" style={{ marginLeft: 'auto', background: sync ? 'var(--bg-2)' : 'transparent', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 8px', color: sync ? 'var(--accent)' : 'var(--text-2)', cursor: 'pointer', fontSize: 11 }}>[{sync ? '●' : '○'}] sync</button>
        <span style={{ color: anyBusy ? 'var(--live)' : 'var(--text-2)', fontSize: 11 }}>{anyBusy ? '[●] live' : '[ idle ]'}</span>
      </div>

      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {explorerOpen && (
          <div style={{ width: 150, flexShrink: 0, padding: '8px 10px', overflow: 'auto', background: 'var(--bg-1)', borderRight: '1px solid var(--line)' }}>
            <div style={{ color: 'var(--text-2)', fontWeight: 500, marginBottom: 6 }}>[ explorer ]</div>
            {open.map((m) => <div key={m.id} onClick={() => setFocusModel(m.id)} style={{ cursor: 'pointer', padding: '2px 0', color: m.id === focused() ? 'var(--accent)' : 'var(--text-2)' }}>{m.id === focused() ? '▸ ' : '  '}{m.label}</div>)}
            <div style={{ color: 'var(--text-2)', marginTop: 8, fontSize: 11 }}>[+] tensors — soon</div>
          </div>
        )}
        <div ref={rowRef} style={{ flex: 1, display: 'flex', minWidth: 0 }}>
        {cols.map((col, ci) => (
          <Fragment key={col.id}>
            <div style={{ flexGrow: col.w, flexBasis: 0, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
              {col.tiles.map((tile, ti) => (
                <Fragment key={tile.id}>
                  <div onMouseDown={() => setFocusModel(tile.model)} onDragOver={(e) => { e.preventDefault(); setOverTile(tile.id) }} onDrop={() => moveTab(tile.id)}
                    style={{ position: 'relative', flexGrow: tile.h, flexBasis: 0, minHeight: 0, display: 'flex', flexDirection: 'column', borderTop: ti > 0 ? '1px solid var(--line)' : 'none' }}>
                    <div style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line-strong)', background: 'var(--bg-1)', overflow: 'hidden' }}>
                      <select value={tile.model} onChange={(e) => setTileModel(tile.id, e.target.value)} style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--accent)', border: 'none', borderRight: '1px solid var(--line)', padding: '4px 2px', maxWidth: 96 }}>
                        {open.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                      </select>
                      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                        {tile.tabs.map((v, idx) => (
                          <span key={v} draggable onDragStart={() => { drag.current = { tid: tile.id, idx }; setDragging(true) }} onDragEnd={() => { drag.current = null; setDragging(false); setOverTile(null) }}
                            onDragOver={(e) => { e.preventDefault(); e.stopPropagation() }} onDrop={(e) => { e.stopPropagation(); moveTab(tile.id, idx) }} onClick={() => setActive(tile.id, idx)}
                            style={{ display: 'flex', alignItems: 'center', gap: 3, padding: '5px 7px', cursor: 'grab', fontSize: 11, whiteSpace: 'nowrap', color: idx === tile.active ? 'var(--text-0)' : 'var(--text-2)', background: idx === tile.active ? 'var(--bg-2)' : 'transparent', borderTop: idx === tile.active ? '2px solid var(--accent)' : '2px solid transparent', borderRight: '1px solid var(--line)' }}>
                            {v}<span onClick={(e) => { e.stopPropagation(); closeTab(tile.id, idx) }} style={hint}>×</span>
                          </span>
                        ))}
                        <button onClick={() => addTab(tile.id)} title="add view" style={iconBtn}>+</button>
                      </div>
                      <div style={{ display: 'flex', flexShrink: 0 }}>
                        <button onClick={() => splitRight(tile.id)} title="split right" style={iconBtn}>[|]</button>
                        <button onClick={() => splitDown(tile.id)} title="split down" style={iconBtn}>[-]</button>
                        {multi && <button onClick={() => pruneClose(tile.id)} title="close pane" style={iconBtn}>[x]</button>}
                      </div>
                    </div>
                    <div style={{ flex: 1, overflow: 'auto', padding: '12px 14px' }}>{viewBody(tile.model, tile.tabs[tile.active])}</div>
                    {dragging && overTile === tile.id && <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,122,255,0.16)', border: '1px solid var(--accent)', pointerEvents: 'none' }} />}
                  </div>
                  {ti < col.tiles.length - 1 && <div onMouseDown={(e) => resizeTiles(ci, ti, e)} style={{ height: 5, flexShrink: 0, cursor: 'row-resize', background: 'var(--line)' }} />}
                </Fragment>
              ))}
            </div>
            {ci < cols.length - 1 && <div onMouseDown={(e) => resizeCols(ci, e)} style={{ width: 5, flexShrink: 0, cursor: 'col-resize', background: 'var(--line)' }} />}
          </Fragment>
        ))}
        </div>
        {/* chat dock — bound to the focused model */}
        <div style={{ width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--line)', background: 'var(--bg-1)' }}>
          <div style={{ padding: '6px 12px', color: 'var(--text-2)', fontWeight: 500, borderBottom: '1px solid var(--line)' }}>[ chat · {open.find((m) => m.id === focused())?.label ?? ''} ]{sync && <span style={hint}> · broadcast</span>}</div>
          <div style={{ flex: 1, overflow: 'auto', padding: '10px 12px', whiteSpace: 'pre-wrap' }}>
            {(data[focused()]?.output) || <span style={hint}>[ ask below ]</span>}
            {data[focused()]?.busy && <span style={{ color: 'var(--accent)' }}>▌</span>}
          </div>
          <div style={{ display: 'flex', gap: 8, padding: 10, borderTop: '1px solid var(--line)' }}>
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder={sync ? 'ask all open models…' : 'ask focused…'} style={{ flex: 1, background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '6px 10px', outline: 'none' }} />
            <button onClick={anyBusy ? stop : send} style={{ background: 'transparent', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '6px 12px', color: anyBusy ? 'var(--danger)' : 'var(--text-0)', cursor: 'pointer' }}>{anyBusy ? '[x]' : '↑'}</button>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 12px', borderTop: '1px solid var(--line)', background: 'var(--bg-1)', color: 'var(--text-2)', fontSize: 11 }}>
        <span>kernel :8000 · mps · bf16 · {open.length} model{open.length > 1 ? 's' : ''}</span>
        <span>{sync ? 'sync: broadcast' : 'sync: focused'} · {tileCount()} pane{tileCount() > 1 ? 's' : ''}</span>
      </div>
    </div>
  )
}
