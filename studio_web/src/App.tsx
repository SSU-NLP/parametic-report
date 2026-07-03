import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import CodeMirror, { EditorView, keymap } from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { json } from '@codemirror/lang-json'
import { javascript } from '@codemirror/lang-javascript'
import { java } from '@codemirror/lang-java'
import { cpp } from '@codemirror/lang-cpp'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags } from '@lezer/highlight'

// Tauri bridge — no-ops in the browser (non-Tauri) so the studio runs either way.
const inTauri = () => typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
async function tauriInvoke(cmd: string) {
  if (!inTauri()) return
  try { const { invoke } = await import('@tauri-apps/api/core'); await invoke(cmd) } catch { /* not in tauri */ }
}
// like tauriInvoke but takes args and surfaces success/failure — for commands the caller must await (ssh_connect/disconnect).
async function tauriInvokeResult(cmd: string, args?: Record<string, unknown>): Promise<void> {
  if (!inTauri()) throw new Error('not running in the desktop app')
  const { invoke } = await import('@tauri-apps/api/core')
  await invoke(cmd, args)
}
async function tauriListen(event: string, cb: (payload: string) => void): Promise<() => void> {
  if (!inTauri()) return () => {}
  try {
    const { listen } = await import('@tauri-apps/api/event')
    return await listen<string>(event, (e) => cb(e.payload))
  } catch { return () => {} }
}

const DEFAULT_WS = 'ws://localhost:8000/ws'
const SSH_TUNNEL_WS = 'ws://localhost:8422/ws'  // P7: local end of the Rust-owned SSH tunnel to a remote kernel
const WS_URL = localStorage.getItem('ps_kernel_url') || DEFAULT_WS
const WS_TOKEN = localStorage.getItem('ps_kernel_token') || ''
const isRemoteConnected = () => WS_URL === SSH_TUNNEL_WS
const DEFAULT = { id: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5-1.5B' }
const VIEWS = ['output', 'attention', 'activations', 'logitlens', 'spot', 'train', 'log'] as const
type View = typeof VIEWS[number]
const DEFAULT_DATASET = 'def add(a, b):\n    return a + b\nfor i in range(10):\n    print(i)\nx = [3, 1, 2]\nx.sort()'
const GENERAL_SET = 'The weather was pleasant and the streets were quiet.\nShe walked to the market to buy fresh vegetables.\nHistory teaches us patience and perspective.\nThe orchestra played beautifully through the evening.'
const PRESETS: Record<string, string> = {
  python: 'def add(a, b):\n    return a + b\nfor i in range(10):\n    print(i)\nx = [3, 1, 2]\nx.sort()\nwith open("f") as fp:\n    data = fp.read()\nresult = [n * n for n in nums if n > 0]\ntry:\n    v = int(s)\nexcept ValueError:\n    v = 0',
  java: 'public int add(int a, int b) {\n    return a + b;\n}\nfor (int i = 0; i < 10; i++) {\n    System.out.println(i);\n}\nList<Integer> x = new ArrayList<>();\nx.sort(Comparator.naturalOrder());\ntry {\n    int v = Integer.parseInt(s);\n} catch (NumberFormatException e) {\n    v = 0;\n}',
  cpp: 'int add(int a, int b) {\n    return a + b;\n}\nfor (int i = 0; i < 10; ++i) {\n    std::cout << i << std::endl;\n}\nstd::vector<int> x = {3, 1, 2};\nstd::sort(x.begin(), x.end());\nstd::ifstream fp("f.txt");\nstd::string data((std::istreambuf_iterator<char>(fp)), {});\ntry {\n    int v = std::stoi(s);\n} catch (const std::invalid_argument& e) {\n    v = 0;\n}',
  javascript: 'function add(a, b) {\n    return a + b;\n}\nfor (let i = 0; i < 10; i++) {\n    console.log(i);\n}\nconst x = [3, 1, 2];\nx.sort((a, b) => a - b);\nconst data = await fs.readFile("f.txt", "utf8");\nconst result = nums.filter(n => n > 0).map(n => n * n);\ntry {\n    v = parseInt(s, 10);\n} catch (e) {\n    v = 0;\n}',
}

// CodeMirror: theme wired to the app's CSS variables. Syntax palette lives in index.css as
// --syn-* tokens (VS Code Dark+/Light+ colors), so dark/light follow the app theme automatically.
const cmTheme = EditorView.theme({
  '&': { backgroundColor: 'var(--bg-0)', color: 'var(--text-0)', height: '100%', fontSize: '13px' },
  '.cm-content': { fontFamily: 'var(--mono)', caretColor: 'var(--text-0)' },
  '.cm-gutters': { backgroundColor: 'var(--bg-0)', color: 'var(--text-2)', border: 'none', borderRight: '1px solid var(--line)' },
  '.cm-activeLine': { backgroundColor: 'color-mix(in srgb, var(--text-0) 4%, transparent)' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--text-0)' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { backgroundColor: 'color-mix(in srgb, var(--accent) 25%, transparent)' },
  '.cm-cursor': { borderLeftColor: 'var(--text-0)' },
  '&.cm-focused': { outline: 'none' },
})
const cmHighlight = syntaxHighlighting(HighlightStyle.define([
  { tag: [tags.keyword, tags.controlKeyword, tags.moduleKeyword, tags.operatorKeyword], color: 'var(--syn-kw)' },
  { tag: [tags.definitionKeyword, tags.bool, tags.null, tags.atom, tags.self], color: 'var(--syn-def)' },
  { tag: [tags.string, tags.special(tags.string), tags.regexp], color: 'var(--syn-str)' },
  { tag: tags.number, color: 'var(--syn-num)' },
  { tag: [tags.comment, tags.lineComment, tags.blockComment], color: 'var(--syn-com)', fontStyle: 'italic' },
  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: 'var(--syn-fn)' },
  { tag: [tags.typeName, tags.className, tags.namespace, tags.standard(tags.variableName)], color: 'var(--syn-type)' },
  { tag: [tags.variableName, tags.propertyName, tags.attributeName, tags.definition(tags.variableName)], color: 'var(--syn-var)' },
]))
function cmLangExt(name: string) {
  const n = name.toLowerCase()
  if (n.endsWith('.py') || n === 'python') return [python()]
  if (n.endsWith('.js') || n.endsWith('.ts') || n === 'javascript') return [javascript()]
  if (n.endsWith('.java') || n === 'java') return [java()]
  if (/\.(cpp|cc|cxx|h|hpp)$/.test(n) || n === 'cpp') return [cpp()]
  if (n.endsWith('.json') || n.endsWith('.jsonl')) return [json()]
  return []
}

type Logit = { token: string; prob: number }
type Spot = { layers: number; modules: string[]; grid: number[][] }
type KnobRow = { key: string; kind: 'cell' | 'spot'; layer?: number; module?: string; topk?: number; op: string; alpha: number }
type ModelData = {
  output: string; frames: number[][][]; act: number[][] | null; logit: Logit[] | null;
  spot: Spot | null; spotProg: { i: number; total: number } | null; perhead: { layer: number; data: number[][] } | null
  knobs: KnobRow[]; kppl: { base: number | null; inter: number | null }; ab: { base: string | null; inter: string | null }
  regions: { name: string; count: number }[]; evals: { code: number | null; general: number | null }
  train: { losses: number[]; total: number; running: boolean; trained: boolean; before: { code: number | null; general: number | null } | null; error: string | null }
  tensors: { name: string; shape: number[]; dtype: string }[] | null
  count: number; busy: boolean; loading: boolean
  lastRunId: string | null; framesFlushed: boolean
  download: { pct: number; done_mb: number; total_mb: number } | null  // HF download stream (open only)
}
const emptyTrain = (): ModelData['train'] => ({ losses: [], total: 0, running: false, trained: false, before: null, error: null })
const empty = (): ModelData => ({ output: '', frames: [], act: null, logit: null, spot: null, spotProg: null, perhead: null, knobs: [], kppl: { base: null, inter: null }, ab: { base: null, inter: null }, regions: [], evals: { code: null, general: null }, train: emptyTrain(), tensors: null, count: 0, busy: false, loading: false, lastRunId: null, framesFlushed: false, download: null })

type Tile = { id: number; model: string; tabs: string[]; active: number; h: number }  // tabs: View | `data:<name>`
type Col = { id: number; w: number; tiles: Tile[] }

// theme-aware colormaps: dark base → bright ramp in dark mode, light base → deep ramp in light mode
const isLight = () => document.documentElement.dataset.theme === 'light'
function ramp(t: number, from: [number, number, number], to: [number, number, number]) {
  return `rgb(${Math.round(from[0] + (to[0] - from[0]) * t)},${Math.round(from[1] + (to[1] - from[1]) * t)},${Math.round(from[2] + (to[2] - from[2]) * t)})`
}
function cellColor(v: number, max: number) {  // blue ramp (attention/activations)
  const t = max > 0 ? v / max : 0
  return isLight() ? ramp(t, [0xE9, 0xE7, 0xE4], [0x00, 0x4F, 0xB0]) : ramp(t, [0x20, 0x1d, 0x1d], [0x00, 0x7a, 0xff])
}
function ampColor(v: number, max: number) {   // amber ramp (spot importance)
  const t = max > 0 ? v / max : 0
  return isLight() ? ramp(t, [0xE9, 0xE7, 0xE4], [0xB0, 0x6A, 0x00]) : ramp(t, [0x20, 0x1d, 0x1d], [0xe0, 0xa8, 0x5e])
}
// per-region hues for spot comparison — each saved region renders in its own color
const REGION_HUES: [number, number, number][] = [[0x00, 0x7a, 0xff], [0xe0, 0xa8, 0x5e], [0x30, 0xd1, 0x58], [0xff, 0x64, 0x82]]
const hueRamp = (rgb: [number, number, number]) => (v: number, max: number) => {
  const t = max > 0 ? v / max : 0
  return ramp(t, isLight() ? [0xE9, 0xE7, 0xE4] : [0x20, 0x1d, 0x1d], rgb)
}
const hueCss = (rgb: [number, number, number]) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
function ScaleBar({ max, color, label }: { max: number; color: (v: number, max: number) => string; label?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: 'var(--text-2)', margin: '4px 0' }}>
      {label && <span>{label}</span>}
      <span>0</span>
      <div style={{ width: 90, height: 8, borderRadius: 2, background: `linear-gradient(to right, ${color(0, 1)}, ${color(0.5, 1)}, ${color(1, 1)})` }} />
      <span>{Number.isFinite(max) ? max.toPrecision(3) : '—'}</span>
    </div>
  )
}
function Grid({ rows, cols, rowH, onRow, onRowEnter, onLeave, cellTitle }: { rows: number[][]; cols: number; rowH: number; onRow?: (i: number) => void; onRowEnter?: (i: number) => void; onLeave?: () => void; cellTitle?: (i: number, k: number, v: number) => string }) {
  const max = Math.max(...rows.flat())
  return (
    <div onMouseLeave={onLeave} style={{ display: 'grid', gridTemplateRows: `repeat(${rows.length}, ${rowH}px)`, gap: 1 }}>
      {rows.map((row, i) => (
        <div key={i} onClick={onRow ? () => onRow(i) : undefined} onMouseEnter={onRowEnter ? () => onRowEnter(i) : undefined} className={onRow ? 'hover-row' : undefined} title={onRow ? `layer ${i} — hover to preview · click to pin` : undefined}
          style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 1, cursor: onRow ? 'pointer' : 'default' }}>
          {Array.from({ length: cols }, (_, k) => <div key={k} title={cellTitle && k < row.length ? cellTitle(i, k, row[k]) : undefined} style={{ background: k < row.length ? cellColor(row[k], max) : 'var(--bg-2)' }} />)}
        </div>
      ))}
    </div>
  )
}
function SpotGrid({ grid, modules, onCell, selected, color = ampColor, cellTitle, onHover, hovered }: { grid: number[][]; modules: string[]; onCell?: (l: number, module: string) => void; selected?: Set<string>; color?: (v: number, max: number) => string; cellTitle?: (l: number, module: string, v: number) => string; onHover?: (cell: string | null) => void; hovered?: string | null }) {
  const flat = grid.flat(); const max = Math.max(...flat)
  const sorted = [...flat].sort((a, b) => b - a)
  const thr = sorted[Math.max(0, Math.floor(sorted.length * 0.05) - 1)] ?? Infinity
  return (
    <div style={{ display: 'grid', gridTemplateRows: `repeat(${grid.length}, 9px)`, gap: 1 }}
      onMouseLeave={onHover ? () => onHover(null) : undefined}>
      {grid.map((row, l) => (
        <div key={l} style={{ display: 'grid', gridTemplateColumns: `repeat(${modules.length}, 1fr)`, gap: 1 }}>
          {row.map((v, c) => {
            const key = `${l}.${modules[c]}`
            const sel = selected?.has(key)
            const hov = hovered === key
            return <div key={c} onClick={onCell ? () => onCell(l, modules[c]) : undefined}
              onMouseEnter={onHover ? () => onHover(key) : undefined}
              title={cellTitle ? cellTitle(l, modules[c], v) : `L${l} · ${modules[c]} · ${v.toExponential(2)}${onCell ? ' — click → knob' : ''}`}
              style={{ background: color(v, max), cursor: onCell ? 'pointer' : 'default', outline: hov ? '1.5px solid var(--accent)' : sel ? '1.5px solid var(--accent)' : v >= thr ? `1px solid ${isLight() ? '#1A1717' : '#FDFCFC'}` : 'none', outlineOffset: (hov || sel) ? -1 : 0 }} />
          })}
        </div>
      ))}
    </div>
  )
}

const hint = { color: 'var(--text-2)' as const }
const iconBtn = { background: 'transparent', border: 'none', color: 'var(--text-2)', cursor: 'pointer', padding: '0 5px', fontSize: 11 }
// common bordered action button — mechanical replacement target for the ~30 inline `[...]` buttons
function Btn({ onClick, children, color = 'var(--text-1)', title, disabled, style }: { onClick?: () => void; children: React.ReactNode; color?: string; title?: string; disabled?: boolean; style?: React.CSSProperties }) {
  return (
    <button className="btn" onClick={onClick} title={title} disabled={disabled}
      style={{ padding: '3px 12px', cursor: disabled ? 'default' : 'pointer', fontSize: 11, color, ...style }}>
      {children}
    </button>
  )
}
// inline 14px stroke icons — name→path map, no icon dependency. color inherits from .tree-icon.
const ICON_PATHS: Record<string, React.ReactNode> = {
  cube: <><path d="M12 2 21 7v10l-9 5-9-5V7z" /><path d="M12 2v10m0 0 9-5m-9 5-9-5" /></>,
  folder: <path d="M3 6a1 1 0 0 1 1-1h5l2 2h8a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z" />,
  file: <><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v4h4M9 12h6M9 16h6" /></>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
  link: <><path d="M9 15l6-6" /><path d="M10.5 6.5 13 4a4 4 0 0 1 6 6l-2.5 2.5M13.5 17.5 11 20a4 4 0 0 1-6-6l2.5-2.5" /></>,
  diamond: <path d="M12 2 22 12 12 22 2 12z" />,
  tensor: <rect x="6" y="6" width="12" height="12" rx="1.5" />,
}
function Icon({ name, size = 14 }: { name: string; size?: number }) {
  return (
    <svg className="tree-icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      {ICON_PATHS[name] ?? ICON_PATHS.file}
    </svg>
  )
}
function Chevron({ open }: { open: boolean }) {
  return (
    <svg className={`tree-chevron${open ? ' open' : ''}`} width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 6l6 6-6 6" />
    </svg>
  )
}
type MenuItem = { label: string; onClick: () => void; key?: string; danger?: boolean; disabled?: boolean } | 'sep'
function ContextMenu({ x, y, items, onClose }: { x: number; y: number; items: MenuItem[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ x, y })
  useEffect(() => {  // clamp inside the viewport once measured
    const el = ref.current; if (!el) return
    const r = el.getBoundingClientRect()
    setPos({ x: Math.min(x, window.innerWidth - r.width - 6), y: Math.min(y, window.innerHeight - r.height - 6) })
  }, [x, y])
  useEffect(() => {
    const close = () => onClose()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    // defer so the opening right-click / click doesn't immediately close it
    const t = window.setTimeout(() => {
      window.addEventListener('click', close)
      window.addEventListener('contextmenu', close)
      window.addEventListener('scroll', close, true)
    }, 0)
    window.addEventListener('keydown', onKey)
    return () => { clearTimeout(t); window.removeEventListener('click', close); window.removeEventListener('contextmenu', close); window.removeEventListener('scroll', close, true); window.removeEventListener('keydown', onKey) }
  }, [onClose])
  return (
    <div ref={ref} className="ctx-menu" style={{ left: pos.x, top: pos.y }} onClick={(e) => e.stopPropagation()} onContextMenu={(e) => { e.preventDefault(); e.stopPropagation() }}>
      {items.map((it, i) => it === 'sep'
        ? <div key={i} className="ctx-sep" />
        : <div key={i} className={`ctx-item${it.danger ? ' danger' : ''}${it.disabled ? ' disabled' : ''}`} onClick={() => { if (!it.disabled) it.onClick() }}>
            <span>{it.label}</span>{it.key && <span className="ctx-key">{it.key}</span>}
          </div>)}
    </div>
  )
}

// PENDING_OPS: request ops that get a ⟳-pending badge until their matching response (or error) arrives
const PENDING_OPS = new Set(['ppl', 'intervene', 'save_region', 'drilldown', 'region_compare', 'region_info'])

// Dataset content → training/spot examples. Understands JSON arrays / JSONL (records), with an
// optional per-dataset field selection (e.g. context+question for benchmark files); plain text
// falls back to one-example-per-line.
function parseRecords(text: string): unknown[] | null {
  const t = text.trim()
  if (t.startsWith('[')) {
    try { const j = JSON.parse(t); if (Array.isArray(j)) return j } catch { /* not a JSON array */ }
  }
  const lines = t.split('\n').map((s) => s.trim()).filter(Boolean)
  if (lines.length && lines.every((l) => l.startsWith('{') || (l.startsWith('"') && l.endsWith('"')))) {
    try { return lines.map((l) => JSON.parse(l)) } catch { /* not JSONL */ }
  }
  return null
}
function recordText(r: unknown, fields?: string[]): string | null {
  if (typeof r === 'string') return r
  if (!r || typeof r !== 'object') return null
  const obj = r as Record<string, unknown>
  const str = (v: unknown) => (typeof v === 'string' ? v : v == null ? null : JSON.stringify(v))
  // ponytail: multi-field examples join with a space so they stay one-per-line when materialized.
  if (fields?.length) return fields.map((f) => str(obj[f])).filter(Boolean).join(' ')
  for (const k of ['text', 'content', 'code', 'prompt', 'input', 'question']) if (typeof obj[k] === 'string') return obj[k] as string
  let best: string | null = null  // auto fallback: the longest string field
  for (const v of Object.values(obj)) if (typeof v === 'string' && (best == null || v.length > best.length)) best = v
  return best
}
function toExamples(text: string, fields?: string[]): string[] {
  const recs = parseRecords(text)
  if (recs) return recs.map((r) => recordText(r, fields)).filter((x): x is string => !!x && !!x.trim())
  return text.split('\n').map((s) => s.trim()).filter(Boolean)
}

type LogEntry = { ts: string; model: string; kind: 'action' | 'result'; text: string; py: string }
function regionPy(r: any): string {
  if (!r) return 'None'
  if (r.kind === 'cell') return `s.locate_cell(${r.layer}, ${JSON.stringify(r.module)})`
  if (r.kind === 'spot') return `s.locate_spot(examples, topk=${r.topk ?? 0.05})`
  return `s.regions[${JSON.stringify(r.name)}]`
}
// ponytail: examples arrays can hit ~1MB (2000 lines) — truncate the log's python column, not the actual WS payload.
function pyList(arr: string[]): string {
  const full = JSON.stringify(arr)
  return full.length <= 2000 ? full : `examples  # ${arr.length} examples (elided — full data in datasets store)`
}
// ponytail: the python equivalent of each WS action — copy the py column to reproduce a session outside the studio.
function actionPy(m: any): string | null {
  switch (m.type) {
    case 'generate': return `list(s.generate_text(${JSON.stringify(m.prompt)}, max_tokens=${m.max_tokens ?? 64}))`
    case 'spot': return `s.compute_spot(${pyList(m.examples)})`
    case 'intervene': return `s.intervene(${regionPy(m.region)}, ${JSON.stringify(m.op ?? 'scale')}, alpha=${m.alpha ?? 0}, key=${JSON.stringify(m.key ?? 'default')})`
    case 'clear': return m.key ? `s.clear(${JSON.stringify(m.key)})` : 's.clear()'
    case 'suspend': return 's.suspend()'
    case 'resume': return 's.resume()'
    case 'ppl': return `s.ppl(${pyList(m.examples)})${m.tag ? `  # ${m.tag}` : ''}`
    case 'save_region': return `s.save_region(${JSON.stringify(m.name)}, ${regionPy(m.region)})`
    case 'train': return `list(s.train_steps(${pyList(m.examples)}, mode=${JSON.stringify(m.mode ?? 'full')}, steps=${m.steps ?? 50}, lr=${m.lr ?? 1e-4}${m.region ? `, region=${regionPy(m.region)}` : ''}))`
    case 'stop_train': return 's.stop()'
    case 'reset_train': return 's.reset_training()'
    default: return null  // catalog/open/close/drilldown/stop — not experiment actions
  }
}
function download(name: string, text: string, type: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type }))
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

export default function App() {
  const [prompt, setPrompt] = useState('write a quicksort in python')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => (localStorage.getItem('parametic-theme') === 'light' ? 'light' : 'dark'))
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('parametic-theme', theme) }, [theme])
  const [sync, setSync] = useState(true)
  const [explorerOpen, setExplorerOpen] = useState(true)
  const [explorerW, setExplorerW] = useState(() => Number(localStorage.getItem('parametic-explorer-w')) || 230)
  const [chatW, setChatW] = useState(() => Number(localStorage.getItem('parametic-chat-w')) || 320)
  useEffect(() => { localStorage.setItem('parametic-explorer-w', String(explorerW)); localStorage.setItem('parametic-chat-w', String(chatW)) }, [explorerW, chatW])
  function dragSidebar(e: React.MouseEvent, side: 'left' | 'right') {
    e.preventDefault()
    const x0 = e.clientX, w0 = side === 'left' ? explorerW : chatW
    const move = (ev: MouseEvent) => {
      const d = ev.clientX - x0
      if (side === 'left') setExplorerW(Math.min(480, Math.max(150, w0 + d)))
      else setChatW(Math.min(600, Math.max(200, w0 - d)))
    }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
  const [focusModel, setFocusModel] = useState(DEFAULT.id)
  const [open, setOpen] = useState<{ id: string; label: string }[]>([DEFAULT])
  const [catalog, setCatalog] = useState<{ id: string; label: string; installed?: boolean; size_mb?: number | null }[]>([])
  const [data, setData] = useState<Record<string, ModelData>>({ [DEFAULT.id]: empty() })
  const [cols, setCols] = useState<Col[]>([{ id: 1, w: 1, tiles: [{ id: 1, model: DEFAULT.id, tabs: ['output', 'attention'], active: 0, h: 1 }] }])
  const [ds, setDs] = useState(DEFAULT_DATASET)
  const [layer, setLayer] = useState(0)
  const [hoverLayer, setHoverLayer] = useState<number | null>(null)  // attention: hover=preview, click=pin
  const [trainMode, setTrainMode] = useState('spot-freeze')
  const [trainRegion, setTrainRegion] = useState('')
  const [trainSteps, setTrainSteps] = useState(30)
  const [trainLr, setTrainLr] = useState(1e-4)
  const [trainDs, setTrainDs] = useState(PRESETS.java)
  // presets are client-side seeds; server entries live in $PARAMETIC_STUDIO_HOME/datasets (content lazy-fetched)
  const [datasets, setDatasets] = useState<{ name: string; content: string | null; fields?: string[]; server?: boolean; size?: number; link?: string | null }[]>([
    { name: 'python', content: PRESETS.python }, { name: 'java', content: PRESETS.java },
    { name: 'cpp', content: PRESETS.cpp }, { name: 'javascript', content: PRESETS.javascript },
    { name: 'general', content: GENERAL_SET },
  ])
  const [spotN, setSpotN] = useState('')                 // '' = all examples
  const [spotPick, setSpotPick] = useState<'first' | 'random'>('first')
  const [evalDsName, setEvalDsName] = useState('')       // '' = eval kppl on spot data (dsExamples); else a loaded dataset name
  const sample = (ex: string[]) => {
    const n = Number(spotN)
    if (!n || n >= ex.length) return ex
    if (spotPick === 'first') return ex.slice(0, n)
    const pool = [...ex]                                  // partial Fisher–Yates: n random picks
    for (let i = 0; i < n; i++) { const j = i + Math.floor(Math.random() * (pool.length - i)); [pool[i], pool[j]] = [pool[j], pool[i]] }
    return pool.slice(0, n)
  }
  const [regionInfo, setRegionInfo] = useState<Record<string, { layers: number; modules: string[]; grid: number[][]; importance: number[][] | null; count: number }>>({})
  const [compareSel, setCompareSel] = useState<string[]>([])  // region names in the compare tab (order = hue)
  const [compareHover, setCompareHover] = useState<string | null>(null)  // shared "L.module" cell — cross-highlights every compare grid
  const [interMetric, setInterMetric] = useState<'shared' | 'lift' | 'fraction'>('shared')  // intersection grid coloring
  const [compareData, setCompareData] = useState<{ names: string[]; layers: number; modules: string[]; grids: Record<string, number[][]>; kinds: Record<string, string>; intersection: number[][]; intersectionLift: number[][] | null; jaccard: Record<string, number> } | null>(null)
  // parse each dataset ONCE per change — parsing in render paths re-chewed megabytes of JSONL on
  // every token-stream re-render (GB-scale GC churn).
  const dsMeta = useMemo(() => {
    const meta: Record<string, { count: number; examples: string[] }> = {}
    for (const d of datasets) if (d.content != null) { const ex = toExamples(d.content, d.fields); meta[d.name] = { count: ex.length, examples: ex } }
    return meta
  }, [datasets])
  const dsExamples = useMemo(() => toExamples(ds), [ds])  // spot-view editor content, parsed once per edit
  const [expModels, setExpModels] = useState<Set<string>>(new Set())  // models with their tensor tree expanded
  const [expPaths, setExpPaths] = useState<Set<string>>(new Set())    // expanded folder paths (model-id prefixed)
  // explorer selection: `${section}:${name}` — target of Del key + context menu ('model:'|'data:'|'region:')
  const [selected, setSelected] = useState<string | null>(null)
  const selectedRef = useRef(selected); selectedRef.current = selected
  // open context menu: screen coords + which row it targets. items are rebuilt each render so the
  // armed-delete label ("sure? — click again") stays live while the menu is open.
  const [menu, setMenu] = useState<{ x: number; y: number; target: string } | null>(null)
  const [dragging, setDragging] = useState(false)
  const [overTile, setOverTile] = useState<number | null>(null)
  const nextId = useRef(2)
  const drag = useRef<{ tid: number; idx: number } | null>(null)
  const sockets = useRef<Record<string, WebSocket>>({})
  const rowRef = useRef<HTMLDivElement>(null)
  // log panel autoscroll: pin to bottom on new entries, unless the user has scrolled up
  const logScrollRef = useRef<HTMLDivElement>(null)
  const logPinned = useRef(true)
  const abPhase = useRef<Record<string, 'base' | 'inter'>>({})  // A/B sequencing (refs: onmessage closure is created once)
  const promptRef = useRef(prompt); promptRef.current = prompt
  const [genOpen, setGenOpen] = useState(false)
  const [maxTokens, setMaxTokens] = useState(256)
  const [temperature, setTemperature] = useState(0)
  const [probesOn, setProbesOn] = useState<Record<string, boolean>>({ attention: true, activation: true, logitlens: true })
  const genRef = useRef({ maxTokens, temperature }); genRef.current = { maxTokens, temperature }
  // kernel liveness: null=connecting, true=up, false=down (reconnecting with backoff)
  const [kernelUp, setKernelUp] = useState<boolean | null>(null)
  const [kernelStats, setKernelStats] = useState<{ rss_mb: number } | null>(null)
  // settings panel (menu-driven; browser gets a small status-bar button) — config + cached-model management
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [config, setConfig] = useState<Record<string, string>>({})
  // remote kernel: URL/token editable in settings, applied via reload (see WS_URL/WS_TOKEN module consts)
  const [kernelUrlInput, setKernelUrlInput] = useState(WS_URL)
  const [kernelTokenInput, setKernelTokenInput] = useState(WS_TOKEN)
  // P7 remote (SSH) kernel: Local|Remote mode toggle + connect form. Boots into 'remote' if already
  // tunneled (ps_kernel_url points at the SSH tunnel port) so reload keeps showing Disconnect.
  const [kernelMode, setKernelMode] = useState<'local' | 'remote'>(isRemoteConnected() ? 'remote' : 'local')
  const [sshHost, setSshHost] = useState('')
  const [sshPort, setSshPort] = useState('22')
  const [sshUser, setSshUser] = useState('')
  const [sshAuth, setSshAuth] = useState<'password' | 'key'>('password')
  const [sshPassword, setSshPassword] = useState('')  // state only — never persisted
  const [sshKeyPath, setSshKeyPath] = useState('')  // e.g. ~/.ssh/gpu.pem — not sensitive, ok to persist
  const [sshKeyPassphrase, setSshKeyPassphrase] = useState('')  // state only — never persisted
  const [sshRepoDir, setSshRepoDir] = useState('')
  const [sshModel, setSshModel] = useState('')
  const [sshPythonPath, setSshPythonPath] = useState('')  // e.g. /opt/conda/bin/python — GPU boxes keep torch in a non-default python
  const [sshConnecting, setSshConnecting] = useState(false)
  const [sshStatus, setSshStatus] = useState<{ state: string; detail?: string } | null>(null)
  const [installed, setInstalled] = useState<{ id: string; size_mb: number | null }[] | null>(null)
  // error toasts: stack of {id, text}, 5s auto-dismiss + click-to-dismiss, capped at 4 (oldest dropped)
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([])
  const toastId = useRef(0)
  const toast = (text: string) => setToasts((ts) => [...ts.slice(-3), { id: toastId.current++, text }])
  const dismissToast = (id: number) => setToasts((ts) => ts.filter((t) => t.id !== id))
  // pending registry: `${op}:${mid}` in flight since sendTo, cleared on matching response or error
  const [pending, setPending] = useState<Set<string>>(new Set())
  const pendingRef = useRef<Set<string>>(new Set())
  const setPendingKey = (key: string, on: boolean) => {
    const next = new Set(pendingRef.current)
    on ? next.add(key) : next.delete(key)
    pendingRef.current = next
    setPending(next)
  }
  // locate_progress: op → {i, total}, latest value while in flight, cleared when the op's response lands
  const [locateProg, setLocateProg] = useState<Record<string, { i: number; total: number }>>({})
  const reconnectAttempts = useRef(0)
  const reconnectTimer = useRef<number | null>(null)
  const splashClosed = useRef(false)  // fire close_splash exactly once (first kernelUp or 8s timeout)
  const authFailToasted = useRef(false)  // show the auth-failed toast once, not on every reconnect retry
  const openRef = useRef(open); openRef.current = open
  function scheduleReconnect() {
    if (reconnectTimer.current != null) return
    const delay = Math.min(30000, 1000 * 2 ** reconnectAttempts.current)
    reconnectTimer.current = window.setTimeout(() => {
      reconnectTimer.current = null
      reconnectAttempts.current++
      const probe = new WebSocket(WS_URL)
      probe.onopen = () => {
        probe.close()
        setKernelUp(true)
        reconnectAttempts.current = 0
        // resync: fresh kernel has no sessions/state for us — reload models, catalog re-pulls regions+datasets
        sendTo(openRef.current[0]?.id ?? DEFAULT.id, { type: 'catalog' })
        for (const m of openRef.current) { patch(m.id, (d) => ({ ...d, loading: true })); sendTo(m.id, { type: 'open' }) }
      }
      probe.onerror = () => { try { probe.close() } catch { /* already closed */ } scheduleReconnect() }
    }, delay)
  }
  // webview-safe dialogs: inline inputs replace window.prompt, two-step "sure?" replaces window.confirm
  const [asking, setAsking] = useState<'hf' | 'editor' | 'path' | 'hf-dataset' | null>(null)
  const [askValue, setAskValue] = useState('')
  const [askSplit, setAskSplit] = useState('')
  const [askConfig, setAskConfig] = useState('')          // HF dataset config (e.g. humanevalpack language)
  const [askFilter, setAskFilter] = useState('')           // 'col=value' row filter (e.g. tiny-codes programming_language=Python)
  const [hfLoading, setHfLoading] = useState<string | null>(null)  // repo id currently loading, for the Data section hint
  const [armed, setArmed] = useState<string | null>(null)
  const armedTimer = useRef<number | null>(null)
  function confirmClick(key: string, action: () => void) {
    if (armed === key) { setArmed(null); if (armedTimer.current) clearTimeout(armedTimer.current); action(); return }
    setArmed(key)
    if (armedTimer.current) clearTimeout(armedTimer.current)
    armedTimer.current = window.setTimeout(() => setArmed(null), 3000)  // arm expires
  }
  const loadStart = useRef<Record<string, number>>({})   // model id → load start ts (for the chip progress bar)
  const [tick, setTick] = useState(0)
  const anyLoading = open.some((m) => data[m.id]?.loading)
  useEffect(() => {
    if (!anyLoading) return
    const iv = setInterval(() => setTick((t) => t + 1), 1000)  // re-render each second while loading (elapsed counter)
    return () => clearInterval(iv)
  }, [anyLoading])
  void tick
  const [expLog, setExpLog] = useState<LogEntry[]>([])
  const [logPy, setLogPy] = useState(false)
  const logEntry = (model: string, kind: LogEntry['kind'], text: string, py = '') =>
    setExpLog((l) => [...l, { ts: new Date().toISOString(), model, kind, text, py }])
  useEffect(() => {
    const el = logScrollRef.current
    if (el && logPinned.current) el.scrollTop = el.scrollHeight
  }, [expLog, logPy])

  function patch(mid: string, f: (d: ModelData) => ModelData) { setData((all) => ({ ...all, [mid]: f(all[mid] ?? empty()) })) }
  function handleMessage(e: MessageEvent) {
    const m = JSON.parse(e.data)
    if (m.type === 'catalog') {
      setCatalog(m.models)
      const def = m.models.find((x: { id: string; label: string }) => x.id === m.default)
      if (def && def.id !== DEFAULT.id) {  // kernel booted with a different model — follow it if the UI is untouched
        setOpen((o) => (o.length === 1 && o[0].id === DEFAULT.id ? [def] : o))
        setData((all) => (all[DEFAULT.id] && !all[DEFAULT.id].count ? { [def.id]: empty() } : all))
        setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.model === DEFAULT.id ? { ...t, model: def.id } : t)) })))
        setFocusModel((f) => (f === DEFAULT.id ? def.id : f))
      }
      sendTo(def?.id ?? DEFAULT.id, { type: 'regions' })   // disk-persisted regions appear on startup
      sendTo(def?.id ?? DEFAULT.id, { type: 'datasets' })  // …and the kernel-side dataset store
      return
    }
    if (m.type === 'datasets') {
      setDatasets((dd) => {
        const server = (m.items as { name: string; size: number; link: string | null }[]).map((it) => {
          const prev = dd.find((x) => x.name === it.name)
          return { name: it.name, size: it.size, link: it.link, server: true, content: prev?.content ?? null, fields: prev?.fields }
        })
        const names = new Set(server.map((s) => s.name))
        return [...dd.filter((x) => !x.server && !names.has(x.name)), ...server]  // client presets + server store
      })
      setHfLoading(null)
      return
    }
    if (m.type === 'dataset_content') { setDatasets((dd) => dd.map((x) => (x.name === m.name ? { ...x, content: m.content } : x))); return }
    if (m.type === 'dataset_saved') { sendTo(m.model, { type: 'datasets' }); return }
    if (m.type === 'loading_dataset') { setHfLoading(m.repo); return }
    if (m.type === 'stats') { setKernelStats({ rss_mb: m.rss_mb }); return }
    if (m.type === 'config') { setConfig(m.config ?? {}); return }
    if (m.type === 'installed_models') { setInstalled(m.items ?? []); return }
    if (m.type === 'cached_deleted') { sendTo(focused(), { type: 'installed_models' }); return }
    const mid = m.model
    if (m.type === 'opened') { delete loadStart.current[mid]; patch(mid, (d) => ({ ...d, loading: false, download: null })); sendTo(mid, { type: 'regions' }); return }
    if (m.type === 'loading') { loadStart.current[mid] ??= Date.now(); patch(mid, (d) => ({ ...d, loading: true })); return }
    if (m.type === 'download_progress') { patch(mid, (d) => ({ ...d, download: { pct: m.pct, done_mb: m.done_mb, total_mb: m.total_mb } })); return }
    if (m.type === 'load_failed') { delete loadStart.current[mid]; toast(`[open] ${openRef.current.find((o) => o.id === mid)?.label ?? mid} failed to load`); closeModel(mid); return }
    if (m.type === 'token') patch(mid, (d) => ({ ...d, output: d.output + m.text, count: d.count + 1 }))
    else if (m.type === 'attention') patch(mid, (d) => ({ ...d, frames: [...d.frames, m.data] }))
    else if (m.type === 'activation') patch(mid, (d) => ({ ...d, act: m.data }))
    else if (m.type === 'logitlens') patch(mid, (d) => ({ ...d, logit: m.layers }))
    else if (m.type === 'spot_progress') patch(mid, (d) => ({ ...d, spot: { layers: m.layers, modules: m.modules, grid: m.grid }, spotProg: { i: m.i, total: m.total } }))
    else if (m.type === 'spotmap') patch(mid, (d) => ({ ...d, spot: { layers: m.layers, modules: m.modules, grid: m.grid }, spotProg: null }))
    else if (m.type === 'ppl') {
      setPendingKey(`ppl:${mid}`, false)
      logEntry(mid, 'result', `ppl${m.tag ? `[${m.tag}]` : ''} = ${Number(m.value).toPrecision(5)}`)
      if (m.tag === 'code' || m.tag === 'general') patch(mid, (d) => ({ ...d, evals: { ...d.evals, [m.tag]: m.value } }))
      else patch(mid, (d) => ({ ...d, kppl: { ...d.kppl, [m.tag === 'base' ? 'base' : 'inter']: m.value } }))
    }
    else if (m.type === 'intervened') { setPendingKey(`intervene:${mid}`, false); setLocateProg((p) => { const n = { ...p }; delete n.intervene; return n }) }
    else if (m.type === 'region_saved') {
      setPendingKey(`save_region:${mid}`, false); setLocateProg((p) => { const n = { ...p }; delete n.save_region; return n })
      logEntry(mid, 'result', `region "${m.name}" saved (${m.count} weights)`); sendTo(mid, { type: 'regions' })
    }
    else if (m.type === 'region_info') { setPendingKey(`region_info:${mid}`, false); setRegionInfo((ri) => ({ ...ri, [m.name]: { layers: m.layers, modules: m.modules, grid: m.grid, importance: m.importance ?? null, count: m.count } })) }
    else if (m.type === 'region_comparison') { setPendingKey(`region_compare:${mid}`, false); setCompareData({ names: m.names, layers: m.layers, modules: m.modules, grids: m.grids, kinds: m.kinds ?? {}, intersection: m.intersection, intersectionLift: m.intersection_lift ?? null, jaccard: m.jaccard }) }
    else if (m.type === 'regions') patch(mid, (d) => ({ ...d, regions: m.regions }))
    else if (m.type === 'train_step') patch(mid, (d) => ({ ...d, train: { ...d.train, losses: [...d.train.losses, m.loss], total: m.total, running: true } }))
    else if (m.type === 'trained') {
      logEntry(mid, 'result', `trained mode=${m.mode} steps=${m.steps} (${m.reason})`)
      // snapshot the pre-train evals as "before", then re-measure → evals become "after"
      patch(mid, (d) => ({ ...d, train: { ...d.train, running: false, trained: true, before: d.train.before ?? { ...d.evals } } }))
      sendTo(mid, { type: 'ppl', examples: PRESETS.python.split('\n').filter(Boolean), tag: 'code' })
      sendTo(mid, { type: 'ppl', examples: GENERAL_SET.split('\n').filter(Boolean), tag: 'general' })
    }
    else if (m.type === 'train_reset') patch(mid, (d) => ({ ...d, train: emptyTrain() }))
    else if (m.type === 'tensors') patch(mid, (d) => ({ ...d, tensors: m.tensors }))
    else if (m.type === 'locate_progress') setLocateProg((p) => ({ ...p, [m.op]: { i: m.i, total: m.total } }))
    else if (m.type === 'error') {
      toast(`[${m.op ?? 'kernel'}] ${m.reason}`)
      if (m.op === 'load_hf_dataset') setHfLoading(null)
      if (m.op) { setPendingKey(`${m.op}:${mid}`, false); setLocateProg((p) => { const n = { ...p }; delete n[m.op]; return n }) }
      if (m.op === 'train') patch(mid, (d) => ({ ...d, train: { ...d.train, running: false, error: m.reason } }))
    }
    else if (m.type === 'perhead') { setPendingKey(`drilldown:${mid}`, false); patch(mid, (d) => ({ ...d, perhead: { layer: m.layer, data: m.data } })) }
    else if (m.type === 'done') {
      const ph = abPhase.current[mid]
      if (ph === 'base') {        // A/B: baseline run finished → stash it, re-apply knobs, run intervened
        abPhase.current[mid] = 'inter'
        patch(mid, (d) => ({ ...d, ab: { ...d.ab, base: d.output }, output: '' }))
        sendTo(mid, { type: 'resume' })
        sendTo(mid, { type: 'generate', prompt: promptRef.current, max_tokens: genRef.current.maxTokens, temperature: genRef.current.temperature, probes: [] })
        return
      }
      if (ph === 'inter') {       // A/B: intervened run finished
        delete abPhase.current[mid]
        patch(mid, (d) => ({ ...d, ab: { ...d.ab, inter: d.output }, busy: false }))
        return
      }
      patch(mid, (d) => {
        if (d.frames.length > 32) sendTo(mid, { type: 'save_run', run: { prompt: promptRef.current, output: d.output, frames: d.frames, act: d.act, logit: d.logit, settings: { maxTokens: genRef.current.maxTokens, temperature: genRef.current.temperature }, reason: m.reason, ts: Date.now() } })
        return { ...d, busy: false }
      })
    }
    else if (m.type === 'run_saved') patch(mid, (d) => ({ ...d, lastRunId: m.id, framesFlushed: true, frames: d.frames.slice(-1) }))
    else if (m.type === 'run_data') patch(mid, (d) => ({ ...d, frames: m.run.frames ?? d.frames, framesFlushed: false }))
  }
  function socket(mid: string) {
    let s = sockets.current[mid]
    if (!s || s.readyState > WebSocket.OPEN) {
      s = new WebSocket(WS_URL); sockets.current[mid] = s; s.onmessage = handleMessage
      const sock = s as WebSocket & { _intentional?: boolean }
      if (WS_TOKEN) s.addEventListener('open', () => s.send(JSON.stringify({ type: 'auth', token: WS_TOKEN })), { once: true })
      sock.onopen = () => { setKernelUp(true); reconnectAttempts.current = 0 }
      sock.onclose = (e) => {
        if (sock._intentional) return  // unload closes are intentional
        if (e.code === 4401 && !authFailToasted.current) { authFailToasted.current = true; toast('[kernel] auth failed — check token in settings') }
        setKernelUp(false); scheduleReconnect()
      }
    }
    return s
  }
  function sendTo(mid: string, msg: object) {
    const s = socket(mid); const m = { ...msg, model: mid }
    const type = (m as unknown as { type: string }).type
    const py = actionPy(m)
    if (py) logEntry(mid, 'action', type, py)  // every experiment action lands in the log
    if (PENDING_OPS.has(type)) setPendingKey(`${type}:${mid}`, true)
    if (s.readyState === WebSocket.OPEN) s.send(JSON.stringify(m))
    else s.addEventListener('open', () => s.send(JSON.stringify(m)), { once: true })
  }
  useEffect(() => { sendTo(DEFAULT.id, { type: 'catalog' }) }, [])
  // P7: subscribe once to SSH tunnel progress from the Rust side (no-op outside Tauri).
  useEffect(() => {
    let unlisten: (() => void) | undefined
    let cancelled = false
    tauriListen('ssh-status', (payload) => {
      let parsed: { state: string; detail?: string }
      try { parsed = JSON.parse(payload) } catch { return }
      setSshStatus(parsed)
      if (parsed.state === 'error') toast(`[ssh] ${parsed.detail ?? 'connection failed'}`)
    }).then((fn) => { if (cancelled) fn(); else unlisten = fn })
    return () => { cancelled = true; unlisten?.() }
  }, [])
  useEffect(() => {
    if (!kernelUp) return
    const iv = setInterval(() => sendTo(focused(), { type: 'stats' }), 15000)
    return () => clearInterval(iv)
  }, [kernelUp])
  // splash → main: hand off when the kernel first comes up, or after 8s regardless (Tauri only; no-op in browser)
  const closeSplash = () => { if (splashClosed.current) return; splashClosed.current = true; tauriInvoke('close_splash') }
  useEffect(() => { if (kernelUp) closeSplash() }, [kernelUp])
  useEffect(() => { const t = window.setTimeout(closeSplash, 8000); return () => clearTimeout(t) }, [])
  const anyBusy = Object.values(data).some((d) => d.busy)
  // Esc = stop generation, except while typing in an input/textarea
  useEffect(() => {
    if (!anyBusy) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      stop()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [anyBusy])
  // explorer keyboard: Del/Backspace = delete selected row (via armed confirm) · Cmd/Ctrl+S = save active data editor
  const colsRef = useRef(cols); colsRef.current = cols
  const datasetsRef = useRef(datasets); datasetsRef.current = datasets
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) return
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedRef.current) { e.preventDefault(); deleteSelected() }
      else if ((e.metaKey || e.ctrlKey) && (e.key === 's' || e.key === 'S')) {
        let name: string | null = null
        for (const c of colsRef.current) for (const t of c.tiles) { const v = t.tabs[t.active]; if (v?.startsWith('data:')) name = v.slice(5) }
        if (!name) return  // no data editor active → let the browser handle it
        e.preventDefault()
        const dset = datasetsRef.current.find((x) => x.name === name)
        if (dset?.content != null) sendTo(focused(), { type: 'save_dataset', name, content: dset.content })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])  // deps via refs — handler reads latest cols/datasets/selected without re-binding
  // native menubar → UI actions (Tauri only)
  useEffect(() => {
    let off = () => {}
    tauriListen('menu', (id) => {
      if (id === 'settings') setSettingsOpen(true)
      else if (id === 'load-model') { setAsking('hf'); setAskValue('') }
      else if (id === 'unload-all') { for (const m of openRef.current) closeModel(m.id) }
      else if (id === 'toggle-theme') setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
      else if (id === 'toggle-explorer') setExplorerOpen((v) => !v)
    }).then((fn) => { off = fn })
    return () => off()
  }, [])
  // fetch config + installed list whenever the settings panel opens
  useEffect(() => { if (settingsOpen) { sendTo(focused(), { type: 'get_config' }); sendTo(focused(), { type: 'installed_models' }) } }, [settingsOpen])
  // toasts: 5s auto-dismiss each, independently
  useEffect(() => {
    if (toasts.length === 0) return
    const timers = toasts.map((t) => window.setTimeout(() => dismissToast(t.id), 5000))
    return () => timers.forEach(clearTimeout)
  }, [toasts])

  const focused = () => (open.some((m) => m.id === focusModel) ? focusModel : (open[0]?.id ?? DEFAULT.id))
  const targets = () => (sync ? open.map((m) => m.id) : [focused()])
  // P9: native file picker for the SSH key path — desktop app only (dynamic import keeps the
  // plugin out of the browser bundle's critical path).
  async function browseSshKeyPath() {
    if (!inTauri()) return
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({
        multiple: false, directory: false,
        filters: [{ name: 'SSH key', extensions: ['pem', 'key'] }, { name: 'All files', extensions: ['*'] }],
      })
      if (typeof picked === 'string') setSshKeyPath(picked)
    } catch { /* not in tauri / plugin unavailable */ }
  }
  // P7: SSH remote kernel connect/disconnect. Rust owns the tunnel + kernel lifecycle; we just
  // point WS_URL at the local tunnel port and reload once it reports success.
  async function sshConnect() {
    if (!sshHost.trim() || !sshUser.trim()) { toast('[ssh] host and username are required'); return }
    if (sshAuth === 'key' ? !sshKeyPath.trim() : !sshPassword) { toast(`[ssh] ${sshAuth === 'key' ? 'key path' : 'password'} is required`); return }
    setSshConnecting(true)
    try {
      await tauriInvokeResult('ssh_connect', {
        host: sshHost.trim(), port: Number(sshPort) || 22, username: sshUser.trim(),
        password: sshAuth === 'key' ? '' : sshPassword,
        keyPath: sshAuth === 'key' ? sshKeyPath.trim() : '',
        keyPassphrase: sshAuth === 'key' ? sshKeyPassphrase : '',
        repoDir: sshRepoDir.trim(), pythonPath: sshPythonPath.trim(), model: sshModel.trim(),
      })
      localStorage.setItem('ps_kernel_url', SSH_TUNNEL_WS)
      window.location.reload()
    } catch (err) {
      setSshConnecting(false)
      toast(`[ssh] ${err instanceof Error ? err.message : String(err)}`)
    }
  }
  async function sshDisconnect() {
    try { await tauriInvokeResult('ssh_disconnect') } catch (err) { toast(`[ssh] ${err instanceof Error ? err.message : String(err)}`) }
    localStorage.removeItem('ps_kernel_url')
    localStorage.removeItem('ps_kernel_token')
    window.location.reload()
  }
  function send() {
    if (!prompt.trim()) return
    for (const mid of targets()) { patch(mid, (d) => ({ ...empty(), spot: d.spot, knobs: d.knobs, kppl: d.kppl, regions: d.regions, evals: d.evals, train: d.train, tensors: d.tensors, busy: true })); sendTo(mid, { type: 'generate', prompt, max_tokens: maxTokens, temperature, probes: ['attention', 'activation', 'logitlens'].filter((p) => probesOn[p]) }) }  // keep workspace (spot/knobs/regions/train) across re-runs
  }
  function stop() {
    for (const mid of targets()) {
      delete abPhase.current[mid]  // stopping mid-A/B must not chain into the next generation
      sockets.current[mid]?.send(JSON.stringify({ type: 'stop', model: mid }))
    }
  }
  function openModel(id: string, label: string) {
    if (open.some((m) => m.id === id)) return
    const wasEmpty = open.length === 0
    setOpen((o) => [...o, { id, label }]); patch(id, () => ({ ...empty(), loading: true })); sendTo(id, { type: 'open' })
    if (wasEmpty) setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => ({ ...t, model: id })) })))  // adopt the first model
  }
  function closeModel(id: string) {
    const rest = open.find((m) => m.id !== id)?.id ?? null  // null → studio goes empty
    const s = sockets.current[id] as (WebSocket & { _intentional?: boolean }) | undefined
    if (s) { s._intentional = true; s.send(JSON.stringify({ type: 'close', model: id })); s.close() }
    delete sockets.current[id]
    setOpen((o) => o.filter((m) => m.id !== id))
    setData((d) => { const n = { ...d }; delete n[id]; return n })
    if (rest) setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.model === id ? { ...t, model: rest } : t)) })))
    setFocusModel((f) => (f === id ? (rest ?? '') : f))
  }
  // ---- 2-level layout ops (cols × tiles) ----
  function tileCount() { return cols.reduce((n, c) => n + c.tiles.length, 0) }
  function updateTile(tid: number, f: (t: Tile) => Tile) { setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.id === tid ? f(t) : t)) }))) }
  function setActive(tid: number, idx: number) { updateTile(tid, (t) => ({ ...t, active: idx })) }
  function setTileModel(tid: number, model: string) { updateTile(tid, (t) => ({ ...t, model })) }
  function addTab(tid: number, view: View) { updateTile(tid, (t) => t.tabs.includes(view) ? t : { ...t, tabs: [...t.tabs, view], active: t.tabs.length }) }  // choose which view; no dup tabs
  function newTile(model: string, view: string): Tile { return { id: nextId.current++, model, tabs: [view], active: 0, h: 1 } }
  function openTabId(tabId: string) {  // editor-style: open/focus a dynamic tab in the first tile
    setCols((cs) => cs.map((c, ci) => ci !== 0 ? c : { ...c, tiles: c.tiles.map((t, ti) => {
      if (ti !== 0) return t
      const idx = t.tabs.indexOf(tabId)
      return idx >= 0 ? { ...t, active: idx } : { ...t, tabs: [...t.tabs, tabId], active: t.tabs.length }
    }) }))
  }
  function openDataTab(name: string) {
    const dset = datasets.find((x) => x.name === name)
    if (dset?.server && dset.content == null) sendTo(focused(), { type: 'read_dataset', name })  // lazy fetch from disk
    openTabId(`data:${name}`)
  }
  function openRegionTab(name: string) {
    sendTo(focused(), { type: 'region_info', name })  // refresh the grid every open (cheap)
    openTabId(`region:${name}`)
  }
  function toggleCompare(name: string) {
    setCompareSel((sel) => {
      const next = sel.includes(name) ? sel.filter((x) => x !== name) : [...sel, name].slice(-REGION_HUES.length)  // cap at palette size
      if (next.length >= 2) sendTo(focused(), { type: 'region_compare', names: next })
      else setCompareData(null)
      return next
    })
  }
  // ---- explorer delete actions (shared by inline ×, context menu, and Del key) — each keeps its confirm ----
  function deleteDataset(dset: { name: string; server?: boolean; link?: string | null }) {
    if (!dset.server) { setDatasets((dd) => dd.filter((x) => x.name !== dset.name)); return }  // session preset — reload restores
    confirmClick(`data:${dset.name}`, () => { sendTo(focused(), { type: 'delete_dataset', name: dset.link ?? dset.name }); if (selectedRef.current === `data:${dset.name}`) setSelected(null) })
  }
  function deleteRegion(name: string) {
    confirmClick(`region:${name}`, () => {
      sendTo(focused(), { type: 'delete_region', name })
      setCompareSel((sel) => sel.filter((x) => x !== name)); setCompareData(null)
      if (selectedRef.current === `region:${name}`) setSelected(null)
    })
  }
  // Del key: dispatch on the selected row. datasets/regions run their armed delete; models unload.
  const deleteSelectedRef = useRef<() => void>(() => {})
  deleteSelectedRef.current = () => {
    const s = selectedRef.current; if (!s) return
    const [kind, ...rest] = s.split(':'); const name = rest.join(':')
    if (kind === 'data') { const dset = datasets.find((x) => x.name === name); if (dset) deleteDataset(dset) }
    else if (kind === 'region') deleteRegion(name)
    else if (kind === 'model') { closeModel(name); setSelected(null) }
  }
  const deleteSelected = () => deleteSelectedRef.current()
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
      let view: string | undefined
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

  function viewBody(mid: string, view: string) {
    const d = data[mid] ?? empty()
    if (view.startsWith('data:')) {  // dataset editor tab — model-independent
      const name = view.slice(5)
      const dset = datasets.find((x) => x.name === name)
      if (!dset) return <span style={hint}>dataset removed</span>
      if (dset.content == null) return <span style={hint}>loading {name}…</span>
      const examples = dsMeta[name]?.examples ?? []
      const recs = parseRecords(dset.content)
      const keys = recs?.length && recs[0] && typeof recs[0] === 'object' ? Object.keys(recs[0] as object) : null
      const toggleField = (k: string) => setDatasets((dd) => dd.map((x) => {
        if (x.name !== name) return x
        const f = x.fields ?? []
        return { ...x, fields: f.includes(k) ? f.filter((y) => y !== k) : [...f, k] }
      }))
      return (<>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
          <span style={{ color: 'var(--text-1)' }}><span className="mono">▤ {name}</span><span style={hint}> · {examples.length} examples</span></span>
          <Btn onClick={() => { const ex = sample(examples); setDs(ex.join('\n')); sendTo(focused(), { type: 'spot', examples: ex }) }} title="compute spot on the focused model with the parsed examples (sampling applies)" color="var(--accent)" style={{ padding: '2px 10px' }}>Spot</Btn>
          <Btn onClick={() => setTrainDs(examples.join('\n'))} title="use the parsed examples as the training dataset" style={{ padding: '2px 10px' }}>→ Train data</Btn>
          <Btn onClick={() => sendTo(focused(), { type: 'save_dataset', name, content: dset.content })} title="write to ~/.parametic_studio/datasets (persists across restarts)" style={{ padding: '2px 10px' }}>{dset.server ? 'Save' : 'Save to disk'}</Btn>
        </div>
        {keys && (
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
            <span style={{ ...hint, fontSize: 11 }}>fields:</span>
            {keys.map((k) => {
              const on = dset.fields?.includes(k)
              return <span key={k} onClick={() => toggleField(k)} style={{ fontSize: 11, padding: '1px 8px', borderRadius: 4, cursor: 'pointer', border: '1px solid var(--line-strong)', color: on ? 'var(--accent)' : 'var(--text-2)', background: on ? 'var(--bg-2)' : 'transparent' }}>{on ? '● ' : ''}{k}</span>
            })}
            <span style={{ ...hint, fontSize: 11 }}>{dset.fields?.length ? '(joined per record)' : '(none = auto: text-ish or longest field)'}</span>
          </div>
        )}
        {keys && <div title={examples[0]} className="mono" style={{ ...hint, fontSize: 11, marginBottom: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>parsed[0] → {examples[0] ?? '—'}</div>}
        <div style={{ width: '100%', height: keys ? 'calc(100% - 96px)' : 'calc(100% - 34px)', minHeight: 120, border: '1px solid var(--line-strong)', borderRadius: 4, overflow: 'hidden', background: 'var(--bg-0)' }}>
          <CodeMirror value={dset.content} height="100%" theme="none"
            onChange={(v) => setDatasets((dd) => dd.map((x) => (x.name === name ? { ...x, content: v } : x)))}
            extensions={[
              cmTheme, cmHighlight, ...cmLangExt(name),
              keymap.of([{ key: 'Mod-s', run: () => { sendTo(focused(), { type: 'save_dataset', name, content: dset.content }); return true } }]),
            ]} />
        </div>
      </>)
    }
    if (view === 'compare') {  // cross-dataset spot comparison (saved regions, one hue each)
      const regs = data[focused()]?.regions ?? []
      const cd = compareData
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 4 }}>region compare<span style={hint}> · pick 2–{REGION_HUES.length} saved regions (e.g. per-language spots)</span></div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
          {regs.map((r) => {
            const idx = compareSel.indexOf(r.name)
            const on = idx >= 0
            return <span key={r.name} onClick={() => toggleCompare(r.name)} style={{ fontSize: 11, padding: '1px 8px', borderRadius: 4, cursor: 'pointer', border: `1px solid ${on ? hueCss(REGION_HUES[idx]) : 'var(--line-strong)'}`, color: on ? hueCss(REGION_HUES[idx]) : 'var(--text-2)' }}>{on ? '● ' : ''}◈ {r.name}</span>
          })}
          {regs.length < 2 && <span style={{ ...hint, fontSize: 11 }}>save regions in the spot view first (one per dataset)</span>}
        </div>
        {compareSel.length >= 2 && !cd && <span style={hint}>comparing…</span>}
        {cd && (<>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(cd.names.length, 2)}, 1fr)`, gap: 12, marginBottom: 10 }}>
            {cd.names.map((n, i) => (
              <div key={n}>
                <div style={{ fontSize: 11, color: hueCss(REGION_HUES[i]), marginBottom: 3 }}>◈ {n}<span style={hint}> · {cd.kinds[n] === 'importance' ? '|g×w| importance' : 'selection fraction — legacy region, re-save to get the importance map'}</span></div>
                <SpotGrid grid={cd.grids[n]} modules={cd.modules} color={hueRamp(REGION_HUES[i])} onHover={setCompareHover} hovered={compareHover} />
              </div>
            ))}
          </div>
          {(() => {
            // intersection panel, 3 selectable metrics over the weights ALL regions selected:
            //  · shared  — geo-mean of per-region importance (where all agree the important weights live)
            //  · lift    — observed ∩ / chance; explodes on tiny data-independent cells (biases/norms)
            //  · fraction— raw shared-weight density
            // 'shared' is the honest default: lift/fraction saturate on biases and bury the real signal.
            const allImp = cd.names.every((n) => cd.kinds[n] === 'importance')
            const metric = interMetric === 'shared' && !allImp ? 'fraction' : interMetric
            const grid = metric === 'shared'
              ? cd.grids[cd.names[0]].map((row, l) => row.map((_, c) =>
                  cd.intersection[l][c] > 0 ? Math.exp(cd.names.reduce((s, n) => s + Math.log(cd.grids[n][l][c] || 1e-30), 0) / cd.names.length) : 0))
              : metric === 'lift' ? (cd.intersectionLift ?? cd.intersection) : cd.intersection
            const flat = grid.flat(); const lo = Math.min(...flat), hi = Math.max(...flat)
            const label = {
              shared: <>color = <b>importance all {cd.names.length} agree on</b> (geo-mean |g×w|) — biases/norms don't saturate it</>,
              lift: <>color = <b>enrichment vs chance</b> (lift; independent top-k ⇒ 1×) — ⚠ tiny data-independent cells (biases) blow up</>,
              fraction: <>color = <b>shared-weight density</b> (∩ fraction) — ~uniform by construction</>,
            }
            const Seg = ({ id, txt }: { id: 'shared' | 'lift' | 'fraction'; txt: string }) => (
              <Btn onClick={() => setInterMetric(id)} title={`color the intersection by ${id}`}
                color={interMetric === id ? 'var(--accent)' : 'var(--text-2)'}
                style={{ borderColor: interMetric === id ? 'var(--accent)' : 'var(--line-strong)', padding: '1px 8px', fontSize: 11 }}>{txt}</Btn>
            )
            return (<>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3, flexWrap: 'wrap' }}>
                <span style={{ color: 'var(--text-1)' }}>intersection<span style={hint}> · weights selected by ALL {cd.names.length} regions</span></span>
                <div style={{ display: 'flex', gap: 4 }}><Seg id="shared" txt="shared imp" /><Seg id="lift" txt="lift" /><Seg id="fraction" txt="∩ fraction" /></div>
              </div>
              <div style={{ ...hint, marginBottom: 3 }}>{label[metric]}{interMetric === 'shared' && !allImp && ' · (legacy region → fell back to fraction; re-save for importance)'}</div>
              <SpotGrid grid={grid} modules={cd.modules}
                color={(v: number) => ampColor(hi > lo ? (v - lo) / (hi - lo) : 0, 1)}
                cellTitle={(l, mod) => { const c = cd.modules.indexOf(mod); const lift = cd.intersectionLift?.[l][c]; const g = grid[l][c]
                  return `L${l} · ${mod}${metric === 'shared' ? ` · shared imp ${g.toExponential(1)}` : ''} · ∩ ${(cd.intersection[l][c] * 100).toFixed(2)}% of weights${lift != null ? ` · ${lift.toFixed(0)}× vs chance` : ''}` }}
                onHover={setCompareHover} hovered={compareHover} />
            </>)
          })()}
          <div style={{ marginTop: 10, fontSize: 11 }}>
            <span style={hint}>pairwise Jaccard (|A∩B| / |A∪B|):</span>
            {Object.entries(cd.jaccard).map(([k, v]) => {
              const [a, b] = k.split('|')
              return <div key={k} style={{ color: 'var(--text-1)' }}><span style={{ color: hueCss(REGION_HUES[cd.names.indexOf(a)]) }}>{a}</span> ∩ <span style={{ color: hueCss(REGION_HUES[cd.names.indexOf(b)]) }}>{b}</span> = <span style={{ color: 'var(--text-0)' }}>{(v * 100).toFixed(1)}%</span></div>
            })}
          </div>
        </>)}
      </>)
    }
    if (view.startsWith('region:')) {  // saved-region viewer tab
      const name = view.slice(7)
      if (!(data[focused()]?.regions ?? []).some((r) => r.name === name)) return <span style={hint}>region deleted</span>
      const info = regionInfo[name]
      if (!info) return <span style={hint}>loading region {name}…</span>
      const g = info.importance ?? info.grid
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 4 }}>◈ {name}<span style={hint}> · {info.count.toLocaleString()} weights selected</span></div>
        <div style={{ ...hint, fontSize: 11, marginBottom: 4 }}>{info.importance
          ? '|grad×param| importance captured when this spot was computed — same heatmap as the spot view. rows=layers (0↑), cols=modules.'
          : 'selection fraction per (layer, module) — legacy region without a saved importance map (re-save to get one). note: per-param top-k% makes this ~uniform by construction.'}</div>
        <ScaleBar max={Math.max(...g.flat())} color={ampColor} label={info.importance ? '|g×w|' : 'fraction'} />
        <SpotGrid grid={g} modules={info.modules} />
        <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>to use it: spot view → named-region knob · train view → region select</div>
      </>)
    }
    if (d.loading) return <span style={hint}>loading {open.find((m) => m.id === mid)?.label ?? 'model'}…</span>
    if (view === 'output') return <div className="mono" style={{ whiteSpace: 'pre-wrap' }}>{d.output || <span style={hint}>run a prompt below</span>}{d.busy && <span style={{ color: 'var(--accent)' }}>▌</span>}</div>
    if (view === 'attention') {
      const last = d.frames.at(-1) ?? null
      if (!last) return <span style={hint}>attention while generating</span>
      const shown = hoverLayer ?? layer  // hover previews a layer, click pins it
      const tri = d.frames.map((f) => f[shown] ?? []); const triC = tri.at(-1)?.length ?? 0
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 4 }}>all layers · current token · {last.length}×{last[0].length}<span style={hint}> · rows=layers, cols=kv tokens · brighter = stronger attention · hover to preview, click to pin ↓</span></div>
        <ScaleBar max={Math.max(...last.flat())} color={cellColor} label="attn" />
        <Grid rows={last} cols={last[0].length} rowH={6} onRow={(i) => setLayer(i)} onRowEnter={(i) => setHoverLayer(i)} onLeave={() => setHoverLayer(null)} cellTitle={(i, k, v) => `L${i} · kv ${k} · ${v.toFixed(3)}`} />
        <div style={{ color: 'var(--text-1)', margin: '14px 0 8px' }}>layer {shown}{hoverLayer != null && hoverLayer !== layer ? <span style={hint}> (preview · click to pin, pinned: L{layer})</span> : <span style={hint}> (pinned)</span>} · query × kv · causal<Btn onClick={() => sendTo(mid, { type: 'drilldown', layer: shown })} disabled={pending.has(`drilldown:${mid}`)} style={{ marginLeft: 10, padding: '1px 8px' }}>{pending.has(`drilldown:${mid}`) ? '⟳ ' : ''}Heads</Btn></div>
        {d.framesFlushed && d.frames.length <= 1
          ? <Btn onClick={() => sendTo(mid, { type: 'load_run', id: d.lastRunId })} color="var(--accent)">Load full history</Btn>
          : <Grid rows={tri} cols={triC} rowH={Math.max(2, Math.min(8, Math.floor(200 / Math.max(1, tri.length))))} cellTitle={(i, k, v) => `query ${i} · kv ${k} · ${v.toFixed(3)}`} />}
        {d.perhead && <div style={{ marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 10 }}><div style={{ color: 'var(--text-1)', marginBottom: 6 }}>layer {d.perhead.layer} · per-head</div><Grid rows={d.perhead.data} cols={d.perhead.data[0].length} rowH={13} cellTitle={(i, k, v) => `head ${i} · kv ${k} · ${v.toFixed(3)}`} /></div>}
      </>)
    }
    if (view === 'activations') {
      if (!d.act) return <span style={hint}>activations while generating</span>
      const mods = ['self_attn', 'mlp']
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 4 }}>layer × module · output norm</div>
        <div style={{ ...hint, fontSize: 11, marginBottom: 4 }}>L2 norm of each module's output vector at the last generated token — rows=layers (0↑), cols=self_attn | mlp. brighter = larger contribution to this token.</div>
        <ScaleBar max={Math.max(...d.act.flat())} color={cellColor} label="‖out‖" />
        <Grid rows={d.act} cols={d.act[0].length} rowH={7} cellTitle={(i, k, v) => `L${i} · ${mods[k] ?? k} · ‖out‖=${v.toFixed(2)}`} />
      </>)
    }
    if (view === 'logitlens') return d.logit ? (<><div style={{ color: 'var(--text-1)', marginBottom: 8 }}>layer → top-1 · {d.logit.length} layers</div><div style={{ display: 'grid', gap: 3 }}>{d.logit.map((r, l) => <div key={l} style={{ display: 'grid', gridTemplateColumns: '34px 76px 1fr', gap: 8, alignItems: 'center' }}><span className="mono" style={hint}>L{l}</span><span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.token}</span><div style={{ background: 'var(--bg-2)', borderRadius: 2, height: 9 }}><div style={{ height: 9, width: `${Math.round(r.prob * 100)}%`, background: 'var(--accent)', borderRadius: 2 }} /></div></div>)}</div></>) : <span style={hint}>logit lens while generating</span>
    if (view === 'log') {
      const short = (id: string) => open.find((m) => m.id === id)?.label ?? id
      const toCSV = () => ['ts,model,kind,text,py', ...expLog.map((e) => [e.ts, short(e.model), e.kind, e.text, e.py].map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','))].join('\n')
      const toScript = () => `# parametic-studio session replay — s = ModelSession.from_pretrained("<model>")\n# examples = your dataset lines\n` + expLog.filter((e) => e.py).map((e) => `${e.py}  # ${e.ts.slice(11, 19)} ${short(e.model)}`).join('\n')
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>experiment log<span style={hint}> · {expLog.length} entries · session-only (not persisted)</span></div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <Btn onClick={() => setLogPy((v) => !v)} color={logPy ? 'var(--accent)' : 'var(--text-1)'} title="show each action as runnable ModelSession python" style={{ padding: '2px 10px' }}>{logPy ? '●' : '○'} py</Btn>
          <Btn onClick={() => download('experiment_log.json', JSON.stringify(expLog, null, 2), 'application/json')} style={{ padding: '2px 10px' }}>JSON</Btn>
          <Btn onClick={() => download('experiment_log.csv', toCSV(), 'text/csv')} style={{ padding: '2px 10px' }}>CSV</Btn>
          {logPy && <Btn onClick={() => download('replay.py', toScript(), 'text/x-python')} style={{ padding: '2px 10px' }}>replay.py</Btn>}
          <Btn onClick={() => setExpLog([])} color="var(--text-2)" style={{ padding: '2px 10px' }}>Clear</Btn>
        </div>
        {expLog.length === 0 && <span style={hint}>actions and results will be recorded here</span>}
        <div ref={logScrollRef} onScroll={(e) => { const el = e.currentTarget; logPinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24 }}
          style={{ maxHeight: '100%', overflow: 'auto' }}>
          {logPy
            ? <pre className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', background: 'var(--bg-2)', borderRadius: 4, padding: 8 }}>{expLog.filter((e) => e.py).map((e) => e.py).join('\n') || '# no actions yet'}</pre>
            : <div className="mono" style={{ display: 'grid', gridTemplateColumns: '58px 74px 1fr', gap: '2px 8px', fontSize: 11 }}>
                {expLog.map((e, i) => (<Fragment key={i}>
                  <span style={hint}>{e.ts.slice(11, 19)}</span>
                  <span style={{ color: 'var(--text-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{short(e.model)}</span>
                  <span style={{ color: e.kind === 'result' ? 'var(--accent)' : 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={e.py || e.text}>{e.kind === 'result' ? '→ ' : ''}{e.text}</span>
                </Fragment>))}
              </div>}
        </div>
      </>)
    }
    if (view === 'train') {
      const tr = d.train
      const needsRegion = trainMode.startsWith('spot')
      const canTrain = !tr.running && !tr.trained && (!needsRegion || trainRegion !== '')
      const runTrain = () => {
        patch(mid, (dd) => ({ ...dd, train: { ...emptyTrain(), running: true }, knobs: [], kppl: { base: null, inter: null } }))  // kernel auto-clears knobs
        sendTo(mid, { type: 'ppl', examples: PRESETS.python.split('\n').filter(Boolean), tag: 'code' })     // pre-train evals → "before"
        sendTo(mid, { type: 'ppl', examples: GENERAL_SET.split('\n').filter(Boolean), tag: 'general' })
        sendTo(mid, {
          type: 'train', mode: trainMode, examples: toExamples(trainDs),
          steps: trainSteps, lr: trainLr, lora_dim: 8, ...(needsRegion ? { region: { kind: 'named', name: trainRegion } } : {}),
        })
      }
      const sel = { fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' } as const
      const L = tr.losses
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>region-aware fine-tuning<span style={hint}> · reversible via reset</span></div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <select value={trainMode} onChange={(e) => setTrainMode(e.target.value)} style={sel} title="spot-freeze: train everything EXCEPT the region · spot-only: train ONLY the region">
            {['full', 'spot-freeze', 'spot-only', 'lora'].map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          {needsRegion && (
            <select value={trainRegion} onChange={(e) => setTrainRegion(e.target.value)} style={sel}>
              <option value="">region…</option>
              {d.regions.map((r) => <option key={r.name} value={r.name}>◈ {r.name}</option>)}
            </select>
          )}
          <span style={hint}>steps</span><input type="number" min={1} value={trainSteps} onChange={(e) => setTrainSteps(Math.max(1, Math.round(Number(e.target.value)) || 1))} style={{ ...sel, width: 48 }} />
          <span style={hint}>lr</span><input type="number" step={1e-5} value={trainLr} onChange={(e) => { const v = Number(e.target.value); if (v > 0) setTrainLr(v) }} style={{ ...sel, width: 72 }} />
        </div>
        <textarea value={trainDs} onChange={(e) => setTrainDs(e.target.value)} rows={3} spellCheck={false} title="training examples (one per line)" style={{ width: '100%', background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '6px 8px', outline: 'none', resize: 'vertical', marginBottom: 6, fontFamily: 'var(--mono)' }} />
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <Btn onClick={runTrain} disabled={!canTrain} color={canTrain ? 'var(--accent)' : 'var(--text-2)'}>Train</Btn>
          {tr.running && <Btn onClick={() => sendTo(mid, { type: 'stop_train' })} color="var(--danger)">Stop</Btn>}
          {tr.trained && <Btn onClick={() => sendTo(mid, { type: 'reset_train' })} title="restore pre-training weights">Reset</Btn>}
          {tr.running && <span style={{ color: 'var(--live)', fontSize: 11, alignSelf: 'center' }}>● training {L.length}/{tr.total}</span>}
          {tr.trained && <span style={{ color: 'var(--accent)', fontSize: 11, alignSelf: 'center' }}>trained · weights modified</span>}
        </div>
        {tr.error && <div style={{ color: 'var(--danger)', fontSize: 11, marginBottom: 6 }}>{tr.error}</div>}
        {L.length > 0 && (() => {
          const mx = Math.max(...L), mn = Math.min(...L)
          const pts = L.map((v, i) => `${(i / Math.max(1, tr.total - 1)) * 260},${36 - ((v - mn) / (mx - mn || 1)) * 30}`).join(' ')
          return (<div style={{ marginBottom: 8 }}>
            <div style={hint}>loss · <span className="mono">{L[L.length - 1].toFixed(4)}</span></div>
            <svg width="100%" height={40} viewBox="0 0 260 40" preserveAspectRatio="none" style={{ background: 'var(--bg-2)', borderRadius: 4 }}><polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={1.2} /></svg>
          </div>)
        })()}
        {tr.before && (
          <div style={{ fontSize: 11 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 1fr', gap: 6 }}>
              <span />
              <span style={hint}>code PPL</span><span style={hint}>general PPL</span>
              <span style={hint}>before</span>
              <span className="mono">{tr.before.code?.toPrecision(4) ?? '—'}</span><span className="mono">{tr.before.general?.toPrecision(4) ?? '—'}</span>
              <span style={hint}>after</span>
              <span className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.code?.toPrecision(4) ?? '…'}</span><span className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.general?.toPrecision(4) ?? '…'}</span>
            </div>
          </div>
        )}
      </>)
    }
    const runSpot = (text: string) => { const ex = sample(toExamples(text)); setDs(ex.join('\n')); sendTo(mid, { type: 'spot', examples: ex }) }  // materialize the sampled set — what you see is what ran
    return (<>
      <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>dataset → grad×param<span style={hint}> · top cells = spot</span></div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
        <select value="" onChange={(e) => {
          const dset = datasets.find((x) => x.name === e.target.value)
          if (!dset) return
          if (dset.content == null) { sendTo(mid, { type: 'read_dataset', name: dset.name }) }  // fetch, then pick again
          else runSpot(toExamples(dset.content, dset.fields).join('\n'))
        }} title="pick a dataset from the explorer and compute its spot" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' }}>
          <option value="">▤ dataset…</option>
          {datasets.map((x) => <option key={x.name} value={x.name}>{x.name} ({dsMeta[x.name]?.count ?? '…'})</option>)}
        </select>
        <span style={hint}>sample</span>
        <input type="number" min={1} value={spotN} onChange={(e) => { const v = e.target.value; if (v === '' || Number(v) >= 1) setSpotN(v) }} placeholder="all" title="how many examples to use (empty = all)" style={{ width: 54, fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' }} />
        <select value={spotPick} onChange={(e) => setSpotPick(e.target.value as 'first' | 'random')} title="first-k or a random sample" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' }}>
          <option value="first">first-k</option>
          <option value="random">random</option>
        </select>
        <span style={hint}>→ loads &amp; computes</span>
      </div>
      <textarea value={ds} onChange={(e) => setDs(e.target.value)} rows={3} spellCheck={false} style={{ width: '100%', background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '6px 8px', outline: 'none', resize: 'vertical', marginBottom: 6, fontFamily: 'var(--mono)' }} />
      <Btn onClick={() => runSpot(ds)} color="var(--accent)" style={{ padding: '4px 12px', marginBottom: 10 }}>Compute spot</Btn>
      {d.spotProg && <div style={{ marginBottom: 8 }}><div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><span style={hint}>computing · {d.spotProg.i}/{d.spotProg.total}</span><Btn onClick={() => sendTo(mid, { type: 'stop_spot' })} color="var(--danger)" style={{ padding: '0 8px' }}>Stop</Btn></div><div style={{ background: 'var(--bg-2)', borderRadius: 2, height: 4, marginTop: 3 }}><div style={{ height: 4, width: `${Math.round((d.spotProg.i / d.spotProg.total) * 100)}%`, background: 'var(--accent)', borderRadius: 2 }} /></div></div>}
      {d.spot && (() => {
        const examples = dsExamples                      // spot data — always drives region()/mask selection
        const evalExamples = evalDsName ? (dsMeta[evalDsName]?.examples ?? examples) : examples  // kppl measurement set — may differ from spot data
        const selected = new Set(d.knobs.map((k) => k.key))
        const measure = () => sendTo(mid, { type: 'ppl', examples: evalExamples, tag: 'inter' })
        const region = (k: KnobRow) => k.kind === 'spot' ? { kind: 'spot', examples, topk: k.topk ?? 0.05 } : { kind: 'cell', layer: k.layer, module: k.module }
        const sendKnob = (k: KnobRow) => { sendTo(mid, { type: 'intervene', region: region(k), op: k.op, alpha: k.alpha, key: k.key }); measure() }
        const baselineOnce = () => { if (d.knobs.length === 0 && d.kppl.base == null) sendTo(mid, { type: 'ppl', examples: evalExamples, tag: 'base' }) }  // clean model baseline first
        const addKnob = (row: KnobRow) => { baselineOnce(); patch(mid, (dd) => ({ ...dd, knobs: [...dd.knobs, row] })); sendKnob(row) }
        const addCell = (l: number, module: string) => {
          const key = `${l}.${module}`
          if (selected.has(key)) return
          addKnob({ key, kind: 'cell', layer: l, module, op: 'scale', alpha: 0 })
        }
        const runAB = () => {
          abPhase.current[mid] = 'base'
          patch(mid, (dd) => ({ ...dd, ab: { base: null, inter: null }, output: '', busy: true }))
          sendTo(mid, { type: 'suspend' })            // baseline pass runs on original weights
          sendTo(mid, { type: 'generate', prompt: promptRef.current, max_tokens: genRef.current.maxTokens, temperature: genRef.current.temperature, probes: [] })
        }
        const adjust = (key: string, p: Partial<KnobRow>) => {
          patch(mid, (dd) => ({ ...dd, knobs: dd.knobs.map((k) => (k.key === key ? { ...k, ...p } : k)) }))
          const k = d.knobs.find((x) => x.key === key); if (k) sendKnob({ ...k, ...p })
        }
        const remove = (key: string) => { sendTo(mid, { type: 'clear', key }); patch(mid, (dd) => ({ ...dd, knobs: dd.knobs.filter((k) => k.key !== key) })); measure() }
        const clearAll = () => { sendTo(mid, { type: 'clear' }); patch(mid, (dd) => ({ ...dd, knobs: [], kppl: { base: null, inter: null } })) }
        const { base, inter } = d.kppl
        return (<>
          <div style={hint}>{d.spot.layers} × {d.spot.modules.length} · |grad×param|<span> · rows=layers, cols=modules · bright cells = important (spot) · click a cell → knob</span></div>
          <ScaleBar max={Math.max(...d.spot.grid.flat())} color={ampColor} label="|g×w|" />
          <div style={{ marginTop: 2 }}><SpotGrid grid={d.spot.grid} modules={d.spot.modules} onCell={addCell} selected={selected} /></div>
          <div style={{ marginTop: 14, borderTop: '1px solid var(--line)', paddingTop: 10 }}>
            <div style={{ color: 'var(--text-1)', marginBottom: 6, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
              knob board<span style={hint}> · {d.knobs.length} active · reversible</span>
              <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
                <span style={hint}>eval on</span>
                <select value={evalDsName} onChange={(e) => setEvalDsName(e.target.value)} title="dataset used to measure baseline/combined PPL (kppl) — separate from the spot data above" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' }}>
                  <option value="">spot data ({examples.length})</option>
                  {datasets.map((x) => <option key={x.name} value={x.name}>{x.name} ({dsMeta[x.name]?.count ?? '…'})</option>)}
                </select>
              </span>
            </div>
            <div style={{ marginBottom: 6 }}>
              {!selected.has('spot') && <Btn onClick={() => addKnob({ key: 'spot', kind: 'spot', topk: 0.05, op: 'scale', alpha: 0 })} style={{ padding: '2px 10px' }}>+ Top-k% spot</Btn>}
              {d.knobs.length === 0 && <span style={{ ...hint, marginLeft: 8 }}>or click a spot cell above ↑</span>}
              {locateProg.intervene && <span style={{ ...hint, fontSize: 11, marginLeft: 8 }}>locating… {locateProg.intervene.i}/{locateProg.intervene.total}</span>}
            </div>
            {d.knobs.map((k) => (
              <div key={k.key} style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                {k.kind === 'spot'
                  ? <span style={{ width: 96, display: 'flex', alignItems: 'center', gap: 2, color: 'var(--accent)', fontSize: 11 }}>spot top<input type="number" min={0.1} max={100} step={0.5} value={(k.topk ?? 0.05) * 100} onChange={(e) => adjust(k.key, { topk: Math.min(100, Math.max(0.1, Number(e.target.value) || 0.1)) / 100 })} style={{ width: 38, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 3, fontSize: 11, padding: '0 2px' }} />%</span>
                  : <span className="mono" style={{ width: 96, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-1)', fontSize: 11 }} title={`L${k.layer} · ${k.module}`}>L{k.layer}·{(k.module ?? '').replace('.weight', '').replace('_proj', '')}</span>}
                <select value={k.op} onChange={(e) => adjust(k.key, { op: e.target.value })} style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '1px 3px' }}>
                  {['scale', 'zero', 'mean', 'random'].map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                {k.op === 'scale' && <>
                  <input type="range" min={0} max={2} step={0.05} value={Math.min(k.alpha, 2)} onChange={(e) => adjust(k.key, { alpha: Number(e.target.value) })} style={{ flex: 1, minWidth: 40 }} />
                  <input type="number" step={0.01} value={k.alpha} onChange={(e) => adjust(k.key, { alpha: Number(e.target.value) })} style={{ width: 52, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 3, fontSize: 11, padding: '1px 3px' }} title="precise α (can exceed 2 to amplify)" />
                </>}
                <button onClick={() => remove(k.key)} style={{ ...iconBtn, color: 'var(--danger)' }}>×</button>
              </div>
            ))}
            {d.knobs.length > 0 && <>
              <div style={{ display: 'flex', gap: 6, margin: '4px 0 8px' }}>
                <Btn onClick={clearAll} color="var(--text-2)" style={{ padding: '2px 10px' }}>Clear all</Btn>
                <Btn onClick={runAB} disabled={d.busy} color="var(--accent)" style={{ padding: '2px 10px' }} title="run the prompt twice: knobs off (baseline) then on (intervened)">A/B compare</Btn>
              </div>
              {base != null && (
                <div style={{ fontSize: 11 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    <div><span style={hint}>baseline PPL</span><div className="mono" style={{ color: 'var(--text-0)' }}>{base.toPrecision(4)}</div></div>
                    <div><span style={hint}>combined PPL</span><div className="mono" style={{ color: inter != null && inter > base ? 'var(--danger)' : 'var(--text-0)' }}>{inter != null ? `${inter.toPrecision(4)}  (×${(inter / base).toPrecision(3)})` : '…'}</div></div>
                  </div>
                  <div style={{ ...hint, marginTop: 3 }}>measured on {evalDsName ? `${evalDsName} (${evalExamples.length})` : `spot data (${evalExamples.length})`}</div>
                </div>
              )}
              {(d.ab.base != null || d.ab.inter != null) && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, fontSize: 11 }}>
                  <div><span style={hint}>baseline output</span><div className="mono" style={{ whiteSpace: 'pre-wrap', background: 'var(--bg-2)', borderRadius: 4, padding: 6, marginTop: 3, maxHeight: 180, overflow: 'auto' }}>{d.ab.base ?? <span style={hint}>generating…</span>}</div></div>
                  <div><span style={hint}>intervened output</span><div className="mono" style={{ whiteSpace: 'pre-wrap', background: 'var(--bg-2)', borderRadius: 4, padding: 6, marginTop: 3, maxHeight: 180, overflow: 'auto', border: '1px solid var(--line-strong)' }}>{d.ab.inter ?? (d.ab.base != null ? d.output || <span style={hint}>generating…</span> : <span style={hint}>…</span>)}</div></div>
                </div>
              )}
              <div style={hint}>re-run the prompt to see the combined output · A/B compare runs it twice</div>
            </>}
          </div>
          <div style={{ marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 10 }}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
              <input id={`rn-${mid}`} placeholder="region name" style={{ width: 110, background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 6px', outline: 'none', fontSize: 11 }} />
              <Btn onClick={() => { const el = document.getElementById(`rn-${mid}`) as HTMLInputElement; const name = el?.value.trim(); if (name) { sendTo(mid, { type: 'save_region', name, region: { kind: 'spot', examples, topk: d.knobs.find((k) => k.kind === 'spot')?.topk ?? 0.05 } }); el.value = '' } }} disabled={pending.has(`save_region:${mid}`)} style={{ padding: '2px 10px' }} title="save the current top-k% spot mask to the workspace">{pending.has(`save_region:${mid}`) ? '⟳ ' : ''}Save region</Btn>
              <Btn onClick={() => { sendTo(mid, { type: 'ppl', examples: PRESETS.python.split('\n').filter(Boolean), tag: 'code' }); sendTo(mid, { type: 'ppl', examples: GENERAL_SET.split('\n').filter(Boolean), tag: 'general' }) }} disabled={pending.has(`ppl:${mid}`)} color="var(--accent)" style={{ padding: '2px 10px' }} title="PPL on code vs general text — selective damage shows here">{pending.has(`ppl:${mid}`) ? '⟳ ' : ''}Eval code｜general</Btn>
            </div>
            {locateProg.save_region && <div style={{ ...hint, fontSize: 11, marginTop: 4 }}>locating… {locateProg.save_region.i}/{locateProg.save_region.total}</div>}
            {(d.evals.code != null || d.evals.general != null) && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11 }}>
                <div><span style={hint}>code PPL</span><div className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.code?.toPrecision(4) ?? '…'}</div></div>
                <div><span style={hint}>general PPL</span><div className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.general?.toPrecision(4) ?? '…'}</div></div>
              </div>
            )}
          </div>
        </>)
      })()}
    </>)
  }

  const closed = catalog.filter((c) => !open.some((o) => o.id === c.id))
  const multi = tileCount() > 1

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 12px', borderBottom: '1px solid var(--line)', background: 'var(--bg-1)' }}>
        <button onClick={() => setExplorerOpen((v) => !v)} title="explorer" style={{ background: 'transparent', border: 'none', color: 'var(--text-1)', cursor: 'pointer', padding: 0 }}>≡</button>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <img src="/logo.png" alt="" style={{ height: 16 }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
          <strong>Parametic Studio</strong>
        </span>
        <div style={{ display: 'flex', gap: 6 }}>
          {open.map((m) => {
            const loading = data[m.id]?.loading
            const dl = data[m.id]?.download
            const elapsed = loading && loadStart.current[m.id] ? Math.floor((Date.now() - loadStart.current[m.id]) / 1000) : 0
            const gb = (mb: number) => (mb / 1024).toFixed(1)
            return (
              <span key={m.id} title={loading ? `downloading/loading… ${elapsed}s` : undefined} style={{ position: 'relative', overflow: 'hidden', fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'var(--bg-2)', border: '1px solid var(--line)' }}>
                {loading ? <span style={{ color: 'var(--accent)' }}>⟳ </span> : data[m.id]?.busy ? <span style={{ color: 'var(--live)' }}>● </span> : ''}
                <span className="mono">{m.label}</span>
                {dl ? <span className="mono" style={hint}> {Math.round(dl.pct)}% · {gb(dl.done_mb)}/{gb(dl.total_mb)}GB</span> : loading && <span className="mono" style={hint}> {elapsed}s</span>}
                <span onClick={() => closeModel(m.id)} title="unload model" style={{ marginLeft: 6, cursor: 'pointer', color: 'var(--text-2)' }}>×</span>
                {loading && (dl
                  ? <span style={{ position: 'absolute', bottom: 0, left: 0, height: 2, width: `${Math.max(0, Math.min(100, dl.pct))}%`, background: 'var(--accent)' }} />
                  : <span style={{ position: 'absolute', bottom: 0, left: 0, height: 2, width: '40%', background: 'var(--accent)', animation: 'loading-slide 1.2s linear infinite' }} />)}
              </span>
            )
          })}
          <select value="" onChange={(e) => {
            if (e.target.value === '__hf') { setAsking('hf'); setAskValue(''); return }  // inline input below
            const c = catalog.find((x) => x.id === e.target.value)
            if (c) openModel(c.id, c.label)
          }} style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '2px 4px' }}>
            <option value="">+ model</option>
            {closed.map((c) => {
              const gb = c.size_mb != null ? (c.size_mb / 1024).toFixed(1) : null
              const label = c.installed ? `✓ ${c.label}` : gb ? `${c.label} (${gb}GB ⬇)` : c.label
              return <option key={c.id} value={c.id}>{label}</option>
            })}
            <option value="__hf">custom (HF id)…</option>
          </select>
          {asking === 'hf' && (
            <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)} placeholder="org/model-id (Llama/Qwen-style) · Enter"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && askValue.trim()) { const id = askValue.trim(); openModel(id, id.split('/').pop() ?? id); setAsking(null) }
                if (e.key === 'Escape') setAsking(null)
              }} onBlur={() => setAsking(null)}
              style={{ fontSize: 11, width: 240, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 4, padding: '2px 6px', outline: 'none' }} />
          )}
        </div>
        <Btn onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} title="toggle light/dark" color="var(--text-2)" style={{ marginLeft: 'auto', padding: '2px 8px' }}>{theme === 'dark' ? '☾' : '☀'}</Btn>
        <Btn onClick={() => setSync((v) => !v)} title="broadcast input to all open models" color={sync ? 'var(--accent)' : 'var(--text-2)'} style={{ background: sync ? 'var(--bg-2)' : 'transparent', padding: '2px 8px' }}>{sync ? '●' : '○'} sync</Btn>
        <span style={{ color: anyBusy ? 'var(--live)' : 'var(--text-2)', fontSize: 11 }}>{anyBusy ? '● live' : 'idle'}</span>
      </div>

      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {explorerOpen && (() => {
          const fd = data[focused()]
          const fmtDtype = (d: string) => d.replace('bfloat16', 'bf16').replace('float32', 'f32').replace('float16', 'f16')
          // VS Code-style folder tree: every dotted name segment is a folder, params are files.
          type TNode = { children: Map<string, TNode>; leaf?: { shape: number[]; dtype: string }; leaves: number }
          const buildTree = (mid: string): TNode | null => {
            const ts = data[mid]?.tensors
            if (!ts) return null
            const root: TNode = { children: new Map(), leaves: 0 }
            for (const t of ts) {
              let node = root
              const segs = t.name.split('.')
              segs.forEach((seg, i) => {
                node.leaves++
                if (!node.children.has(seg)) node.children.set(seg, { children: new Map(), leaves: 0 })
                node = node.children.get(seg)!
                if (i === segs.length - 1) { node.leaf = { shape: t.shape, dtype: t.dtype }; node.leaves++ }
              })
            }
            return root
          }
          const toggle = (p: string) => setExpPaths((s) => { const n = new Set(s); n.has(p) ? n.delete(p) : n.add(p); return n })
          // indent guides: one 1px vertical rule per level, then the row content
          const IndentGuides = ({ depth }: { depth: number }) => (<>{Array.from({ length: depth }, (_, i) => <span key={i} className="tree-indent" style={{ width: 10, flexShrink: 0, alignSelf: 'stretch' }} />)}</>)
          const renderTree = (node: TNode, path: string, depth: number): React.ReactNode =>
            [...node.children.entries()].map(([seg, child]) => {
              const p = `${path}.${seg}`
              if (child.leaf) return (
                <div key={p} className="tree-row mono" title={`${p.split('#.')[1] ?? p} · [${child.leaf.shape.join('×')}] ${child.leaf.dtype}`} style={{ fontSize: 10, height: 20 }}>
                  <IndentGuides depth={depth} />
                  <span style={{ width: 12, flexShrink: 0 }} />
                  <Icon name="tensor" size={12} />
                  <span className="tree-label" style={{ color: 'var(--text-1)' }}>{seg}</span>
                  <span style={{ ...hint, flexShrink: 0 }}>[{child.leaf.shape.join('×')}] {fmtDtype(child.leaf.dtype)}</span>
                </div>
              )
              return (
                <Fragment key={p}>
                  <div onClick={() => toggle(p)} className="tree-row mono" style={{ fontSize: 10, height: 20 }}>
                    <IndentGuides depth={depth} />
                    <Chevron open={expPaths.has(p)} />
                    <Icon name="folder" size={12} />
                    <span className="tree-label" style={{ color: 'var(--text-1)' }}>{seg}</span>
                    <span style={{ ...hint, flexShrink: 0 }}>({child.leaves})</span>
                  </div>
                  {expPaths.has(p) && renderTree(child, p, depth + 1)}
                </Fragment>
              )
            })
          const toggleModel = (mid: string) => {
            setExpModels((s) => { const n = new Set(s); n.has(mid) ? n.delete(mid) : n.add(mid); return n })
            if (!expModels.has(mid) && data[mid]?.tensors == null) sendTo(mid, { type: 'tensors' })  // lazy fetch per model
          }
          // dataset context actions — reuse the editor-tab code paths (sample→spot, →train data)
          const useForSpot = (name: string) => {
            const dset = datasets.find((x) => x.name === name); if (!dset) return
            if (dset.content == null) { sendTo(focused(), { type: 'read_dataset', name }); return }  // fetch first; re-invoke after it lands
            const ex = sample(toExamples(dset.content, dset.fields)); setDs(ex.join('\n')); sendTo(focused(), { type: 'spot', examples: ex })
          }
          const useAsTrainData = (name: string) => {
            const dset = datasets.find((x) => x.name === name); if (!dset) return
            if (dset.content == null) { sendTo(focused(), { type: 'read_dataset', name }); return }
            setTrainDs(toExamples(dset.content, dset.fields).join('\n'))
          }
          // open a context menu at the event position, targeting `target` (items rebuilt each render)
          const openMenu = (e: React.MouseEvent, target: string) => { e.preventDefault(); e.stopPropagation(); setMenu({ x: e.clientX, y: e.clientY, target }) }
          // delete item: reuses the armed confirm. session presets delete outright (menu closes); server
          // datasets/regions arm on the 1st click (menu stays, label flips) and execute on the 2nd.
          const dsMenu = (dset: typeof datasets[number]): MenuItem[] => {
            const isArmed = armed === `data:${dset.name}`
            const label = !dset.server ? 'Remove from list' : dset.link ? (isArmed ? 'Unlink? — click again' : 'Unlink') : (isArmed ? 'Delete? — click again' : 'Delete')
            return [
              { label: 'Open in editor', onClick: () => { setMenu(null); openDataTab(dset.name) } },
              { label: 'Use for spot', onClick: () => { setMenu(null); useForSpot(dset.name) } },
              { label: 'Use as train data', onClick: () => { setMenu(null); useAsTrainData(dset.name) } },
              'sep',
              { label, danger: true, key: 'Del', onClick: () => { const willExecute = !dset.server || isArmed; deleteDataset(dset); if (willExecute) setMenu(null) } },
            ]
          }
          const regionMenu = (name: string): MenuItem[] => {
            const isArmed = armed === `region:${name}`
            return [
              { label: 'View', onClick: () => { setMenu(null); setSelected(`region:${name}`); openRegionTab(name) } },
              { label: 'Compare…', onClick: () => { setMenu(null); openTabId('compare') } },
              'sep',
              { label: isArmed ? 'Delete? — click again' : 'Delete', danger: true, key: 'Del', onClick: () => { deleteRegion(name); if (isArmed) setMenu(null) } },
            ]
          }
          const modelMenu = (mid: string): MenuItem[] => [
            { label: 'Unload', danger: true, key: 'Del', onClick: () => { setMenu(null); closeModel(mid); if (selected === `model:${mid}`) setSelected(null) } },
          ]
          // items for the currently open menu, keyed off its target (kind:name)
          const menuItems = (): MenuItem[] => {
            if (!menu) return []
            const [kind, ...rest] = menu.target.split(':'); const name = rest.join(':')
            if (kind === 'data') { const dset = datasets.find((x) => x.name === name); return dset ? dsMenu(dset) : [] }
            if (kind === 'region') return regionMenu(name)
            if (kind === 'model') return modelMenu(name)
            return []
          }
          const addFiles = (files: FileList | null) => {
            for (const f of Array.from(files ?? [])) {
              const name = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
              const reader = new FileReader()
              reader.onload = () => sendTo(focused(), { type: 'save_dataset', name, content: String(reader.result) })  // uploads persist to the kernel store
              reader.readAsText(f)
            }
          }
          const addBtn = { ...hint, fontSize: 11, cursor: 'pointer', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '1px 8px', background: 'transparent' } as const
          return (
          <div style={{ width: explorerW, flexShrink: 0, padding: '8px 10px', overflow: 'auto', background: 'var(--bg-1)' }}>
            <div className="section-h" style={{ marginBottom: 6 }}>Models</div>
            {open.map((m) => {
              const tree = expModels.has(m.id) ? buildTree(m.id) : null
              const isSel = selected === `model:${m.id}`
              return (
                <Fragment key={m.id}>
                  <div className={`tree-row mono${isSel ? ' sel' : ''}`} onClick={() => { setSelected(`model:${m.id}`); setFocusModel(m.id) }} onContextMenu={(e) => { setSelected(`model:${m.id}`); openMenu(e, `model:${m.id}`) }} title={m.id}>
                    <span onClick={(e) => { e.stopPropagation(); toggleModel(m.id) }}><Chevron open={expModels.has(m.id)} /></span>
                    <Icon name="cube" />
                    <span className="tree-label" style={{ color: m.id === focused() && !isSel ? 'var(--accent)' : 'var(--text-1)' }}>{m.label}</span>
                    <span className="tree-x" title="unload model" onClick={(e) => { e.stopPropagation(); closeModel(m.id); if (isSel) setSelected(null) }}>×</span>
                  </div>
                  {expModels.has(m.id) && (tree == null ? <div style={{ ...hint, fontSize: 10, paddingLeft: 22 }}>loading…</div> : renderTree(tree, `${m.id}#`, 1))}
                </Fragment>
              )
            })}

            <div className="section-h" style={{ margin: '10px 0 4px' }}>Data</div>
            {datasets.map((dset) => {
              const isSel = selected === `data:${dset.name}`
              const isArmed = armed === `data:${dset.name}`
              return (
              <div key={dset.name} className={`tree-row mono${isSel ? ' sel' : ''}`} onClick={() => { setSelected(`data:${dset.name}`); openDataTab(dset.name) }} onContextMenu={(e) => { setSelected(`data:${dset.name}`); openMenu(e, `data:${dset.name}`) }}
                title={dset.link ? `linked from outside the store (${dset.link}) · open in an editor tab` : dset.server ? 'on disk · open in an editor tab' : 'session-only · open in an editor tab'} style={{ fontSize: 11 }}>
                <Icon name={dset.link ? 'link' : dset.server ? 'database' : 'file'} />
                <span className="tree-label" style={{ color: 'var(--text-1)' }}>{dset.name} <span style={hint}>({dsMeta[dset.name]?.count ?? `${((dset.size ?? 0) / 1024).toFixed(1)}k`})</span></span>
                <span className={`tree-x${isArmed ? ' armed' : ''}`} onClick={(e) => { e.stopPropagation(); deleteDataset(dset) }}
                  title={dset.link ? `unlink '${dset.link}' — originals kept · click twice` : dset.server ? 'delete from store (cannot be undone) · click twice' : 'remove from list'}>{isArmed ? (dset.link ? 'unlink?' : 'sure?') : '×'}</span>
              </div>
              )
            })}
            <div style={{ display: 'flex', gap: 5, marginTop: 4, flexWrap: 'wrap' }}>
              <label style={addBtn}>+ File<input type="file" multiple style={{ display: 'none' }} onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} /></label>
              <label style={addBtn}>+ Folder<input type="file" multiple style={{ display: 'none' }} {...({ webkitdirectory: '', directory: '' } as object)} onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} /></label>
              <button onClick={() => { setAsking('path'); setAskValue('') }} title="symlink an external path into ~/.parametic_studio/datasets" style={addBtn}>+ Path</button>
              <button onClick={() => { setAsking('editor'); setAskValue('') }} title="save the spot-view editor content to the dataset store" style={addBtn}>+ Editor</button>
              <button onClick={() => { setAsking('hf-dataset'); setAskValue(''); setAskSplit(''); setAskConfig(''); setAskFilter('') }} title="load a dataset from the Hugging Face Hub by repo id" style={addBtn}>+ HF</button>
            </div>
            {(asking === 'path' || asking === 'editor') && (
              <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)}
                placeholder={asking === 'path' ? '~/data/my.jsonl or folder · Enter' : 'dataset name · Enter'}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && askValue.trim()) {
                    if (asking === 'path') sendTo(focused(), { type: 'link_path', path: askValue.trim() })
                    else if (ds.trim()) sendTo(focused(), { type: 'save_dataset', name: askValue.trim(), content: ds })
                    setAsking(null)
                  }
                  if (e.key === 'Escape') setAsking(null)
                }} onBlur={() => setAsking(null)}
                style={{ fontSize: 11, width: '100%', marginTop: 4, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 4, padding: '2px 6px', outline: 'none' }} />
            )}
            {asking === 'hf-dataset' && (() => {
              const submitHfDataset = () => {
                if (!askValue.trim()) return
                const [filterCol, ...rest] = askFilter.split('=')
                const filterVal = rest.join('=').trim()
                sendTo(focused(), {
                  type: 'load_hf_dataset', repo: askValue.trim(),
                  ...(askSplit.trim() ? { split: askSplit.trim() } : {}),
                  ...(askConfig.trim() ? { config: askConfig.trim() } : {}),
                  ...(filterCol.trim() && filterVal ? { filter_column: filterCol.trim(), filter_value: filterVal } : {}),
                })
                setAsking(null)
              }
              const onKey = (e: React.KeyboardEvent) => {
                if (e.key === 'Enter') submitHfDataset()
                if (e.key === 'Escape') setAsking(null)
              }
              const onBlur = (e: React.FocusEvent) => { if (!e.relatedTarget) setAsking(null) }
              const fieldStyle = { fontSize: 11, minWidth: 0, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 4, padding: '2px 6px', outline: 'none' } as const
              return (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                  <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)} placeholder="openai/gsm8k · Enter"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 2 }} />
                  <input value={askSplit} onChange={(e) => setAskSplit(e.target.value)} placeholder="train"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 1 }} />
                  <input value={askConfig} onChange={(e) => setAskConfig(e.target.value)} placeholder="python  (humanevalpack)"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 1 }} title="dataset config (e.g. humanevalpack language)" />
                  <input value={askFilter} onChange={(e) => setAskFilter(e.target.value)} placeholder="programming_language=Python  (tiny-codes)"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 2 }} title="row filter: col=value, applied before capping rows" />
                  <button onMouseDown={(e) => e.preventDefault()} onClick={submitHfDataset} disabled={!askValue.trim()}
                    title="download this dataset from the Hugging Face Hub"
                    style={{ ...addBtn, flexShrink: 0, borderColor: 'var(--accent)', color: 'var(--accent)', opacity: askValue.trim() ? 1 : 0.5 }}>Load</button>
                </div>
              )
            })()}
            {hfLoading && <div style={{ ...hint, fontSize: 11, marginTop: 4 }}>loading {hfLoading}…</div>}

            <div style={{ display: 'flex', alignItems: 'center', margin: '10px 0 4px' }}>
              <span className="section-h">Regions</span>
              {(fd?.regions ?? []).length >= 2 && <button onClick={() => openTabId('compare')} title="compare saved regions (per-dataset spots)" style={{ ...iconBtn, marginLeft: 'auto', color: 'var(--accent)', padding: 0 }}>Compare</button>}
            </div>
            {(fd?.regions ?? []).length === 0 && <div style={{ ...hint, fontSize: 11 }}>save one in the spot view</div>}
            {(fd?.regions ?? []).map((r) => {
              const isSel = selected === `region:${r.name}`
              const isArmed = armed === `region:${r.name}`
              return (
              <div key={r.name} className={`tree-row${isSel ? ' sel' : ''}`} onClick={() => { setSelected(`region:${r.name}`); openRegionTab(r.name) }} onContextMenu={(e) => { setSelected(`region:${r.name}`); openMenu(e, `region:${r.name}`) }}
                title={`${r.count.toLocaleString()} weights · open viewer`} style={{ fontSize: 11 }}>
                <Icon name="diamond" />
                <span className="tree-label" style={{ color: 'var(--text-1)' }}>{r.name} <span style={hint}>({r.count.toLocaleString()})</span></span>
                <span className={`tree-x${isArmed ? ' armed' : ''}`} onClick={(e) => { e.stopPropagation(); deleteRegion(r.name) }} title="delete region from disk — click twice to confirm">{isArmed ? 'sure?' : '×'}</span>
              </div>
              )
            })}
            {menu && <ContextMenu x={menu.x} y={menu.y} items={menuItems()} onClose={() => setMenu(null)} />}
          </div>
          )
        })()}
        {explorerOpen && <div onMouseDown={(e) => dragSidebar(e, 'left')} style={{ width: 5, flexShrink: 0, cursor: 'col-resize', background: 'var(--line)' }} />}
        {open.length === 0 ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: 'var(--text-2)' }}>
          <div>No model loaded</div>
          {catalog.length > 0 ? <div style={hint}>use + Model above to load one</div> : <div style={hint}>kernel offline — start it on :8000</div>}
        </div>
        ) : (
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
                            {v.startsWith('data:') ? `▤ ${v.slice(5)}` : v.startsWith('region:') ? `◈ ${v.slice(7)}` : v}<span onClick={(e) => { e.stopPropagation(); closeTab(tile.id, idx) }} style={hint}>×</span>
                          </span>
                        ))}
                        {tile.tabs.filter((t) => (VIEWS as readonly string[]).includes(t)).length < VIEWS.length && (
                          <select value="" onChange={(e) => { if (e.target.value) addTab(tile.id, e.target.value as View) }} title="add view" style={{ ...iconBtn, appearance: 'none', background: 'transparent' }}>
                            <option value="">+</option>
                            {VIEWS.filter((v) => !tile.tabs.includes(v)).map((v) => <option key={v} value={v}>{v}</option>)}
                          </select>
                        )}
                      </div>
                      <div style={{ display: 'flex', flexShrink: 0 }}>
                        <button onClick={() => splitRight(tile.id)} title="split right" style={iconBtn}>⊟</button>
                        <button onClick={() => splitDown(tile.id)} title="split down" style={iconBtn}>⊞</button>
                        {multi && <button onClick={() => pruneClose(tile.id)} title="close pane" style={iconBtn}>×</button>}
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
        )}
        {/* chat dock — bound to the focused model */}
        <div onMouseDown={(e) => dragSidebar(e, 'right')} style={{ width: 5, flexShrink: 0, cursor: 'col-resize', background: 'var(--line)' }} />
        <div style={{ width: chatW, flexShrink: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', padding: '6px 12px', color: 'var(--text-2)', fontWeight: 500, borderBottom: '1px solid var(--line)' }}>
            <span className="section-h">Chat · <span className="mono" style={{ textTransform: 'none', letterSpacing: 0 }}>{open.find((m) => m.id === focused())?.label ?? ''}</span>{sync && <span style={{ ...hint, textTransform: 'none', letterSpacing: 0 }}> · broadcast</span>}</span>
            <button onClick={() => setGenOpen((v) => !v)} title="generation options" style={{ ...iconBtn, marginLeft: 'auto', color: genOpen ? 'var(--accent)' : 'var(--text-2)' }}>⚙</button>
          </div>
          {genOpen && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', padding: '6px 12px', borderBottom: '1px solid var(--line)', fontSize: 11 }}>
              <span style={hint}>max_tokens</span>
              <input type="number" min={1} value={maxTokens} onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value)))} style={{ width: 60, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '1px 4px', fontSize: 11 }} />
              <span style={hint} title="0 = greedy (deterministic) · >0 = sampling">temp</span>
              <input type="number" min={0} step={0.1} value={temperature} onChange={(e) => setTemperature(Math.max(0, Number(e.target.value)))} style={{ width: 48, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '1px 4px', fontSize: 11 }} />
              <span style={hint} title="fewer probes = faster generation">probes</span>
              {(['attention', 'activation', 'logitlens'] as const).map((p) => (
                <span key={p} onClick={() => setProbesOn((o) => ({ ...o, [p]: !o[p] }))} style={{ cursor: 'pointer', padding: '0 6px', borderRadius: 4, border: '1px solid var(--line-strong)', color: probesOn[p] ? 'var(--accent)' : 'var(--text-2)' }}>{probesOn[p] ? '●' : '○'} {p.slice(0, 4)}</span>
              ))}
            </div>
          )}
          <div className="mono" style={{ flex: 1, overflow: 'auto', padding: '10px 12px', whiteSpace: 'pre-wrap' }}>
            {(data[focused()]?.output) || <span style={hint} className="mono">ask below</span>}
            {data[focused()]?.busy && <span style={{ color: 'var(--accent)' }}>▌</span>}
          </div>
          <div style={{ display: 'flex', gap: 8, padding: 10, borderTop: '1px solid var(--line)' }}>
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder={sync ? 'ask all open models…' : 'ask focused…'} style={{ flex: 1, background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '6px 10px', outline: 'none' }} />
            <Btn onClick={anyBusy ? stop : send} title={anyBusy ? 'stop generation' : 'send'} color={anyBusy ? 'var(--danger)' : 'var(--text-0)'} style={{ border: `1px solid ${anyBusy ? 'var(--danger)' : 'var(--line-strong)'}`, padding: '6px 12px', whiteSpace: 'nowrap' }}>{anyBusy ? '■ stop' : '↑'}</Btn>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 12px', borderTop: '1px solid var(--line)', background: 'var(--bg-1)', color: 'var(--text-2)', fontSize: 11 }}>
        <span>{kernelUp === false
          ? <span style={{ color: 'var(--danger)' }}>● kernel offline · reconnecting…</span>
          : kernelUp === null
            ? <span style={hint}>○ connecting to kernel :8000…</span>
            : <>kernel {isRemoteConnected() ? `${sshHost || 'remote'} (ssh)` : (() => { try { return new URL(WS_URL.replace(/^ws/, 'http')).host } catch { return WS_URL } })()} · mps · bf16 · {open.length} model{open.length > 1 ? 's' : ''}{kernelStats && ` · rss ${(kernelStats.rss_mb / 1024).toFixed(1)}G`}</>}</span>
        <span style={{ display: 'flex', gap: 10 }}>
          {!inTauri() && <button onClick={() => setSettingsOpen(true)} style={{ ...iconBtn, padding: 0 }}>Settings</button>}
          <span>{sync ? 'sync: broadcast' : 'sync: focused'} · {tileCount()} pane{tileCount() > 1 ? 's' : ''}</span>
        </span>
      </div>
      {settingsOpen && (() => {
        const inp = { width: '100%', background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--line-strong)', borderRadius: 4, padding: '3px 8px', outline: 'none', fontSize: 12 } as const
        const setC = (k: string, v: string) => setConfig((c) => ({ ...c, [k]: v }))
        const gb = (mb: number | null) => (mb != null ? (mb / 1024).toFixed(1) : '?')
        return (
          <div style={{ position: 'fixed', top: 44, right: 12, width: 360, maxHeight: 'calc(100vh - 60px)', overflow: 'auto', background: 'var(--bg-1)', border: '1px solid var(--line-strong)', borderRadius: 8, boxShadow: '0 8px 32px rgba(0,0,0,0.45)', padding: 14, zIndex: 50 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <strong style={{ color: 'var(--text-0)' }}>Settings</strong>
              <button onClick={() => setSettingsOpen(false)} style={{ ...iconBtn, marginLeft: 'auto', fontSize: 14 }}>×</button>
            </div>

            <div className="section-h" style={{ marginBottom: 6 }}>Kernel connection</div>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
              {(['local', 'remote'] as const).map((mode) => (
                <button key={mode} onClick={() => { if (mode === 'remote' && !inTauri()) return; setKernelMode(mode) }}
                  disabled={mode === 'remote' && !inTauri()}
                  title={mode === 'remote' && !inTauri() ? 'Remote (SSH) is only available in the desktop app' : undefined}
                  style={{ flex: 1, padding: '4px 8px', fontSize: 11, borderRadius: 4, cursor: mode === 'remote' && !inTauri() ? 'default' : 'pointer',
                    border: `1px solid ${kernelMode === mode ? 'var(--accent)' : 'var(--line-strong)'}`,
                    color: mode === 'remote' && !inTauri() ? 'var(--text-2)' : kernelMode === mode ? 'var(--accent)' : 'var(--text-1)',
                    background: 'var(--bg-2)', opacity: mode === 'remote' && !inTauri() ? 0.5 : 1 }}>
                  {mode === 'local' ? 'Local' : 'Remote (SSH)'}
                </button>
              ))}
            </div>
            {!inTauri() && kernelMode === 'remote' && (
              <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>Remote (SSH) is only available in the desktop app.</div>
            )}

            {kernelMode === 'local' ? (
              <>
                <div style={{ display: 'grid', gap: 6, marginBottom: 4 }}>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>url</span>
                    <input value={kernelUrlInput} onChange={(e) => setKernelUrlInput(e.target.value)} placeholder={DEFAULT_WS} spellCheck={false} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>token</span>
                    <input type="password" value={kernelTokenInput} onChange={(e) => setKernelTokenInput(e.target.value)} spellCheck={false} style={inp} />
                  </label>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 4px' }}>
                  <Btn onClick={() => {
                    const url = kernelUrlInput.trim()
                    url && url !== DEFAULT_WS ? localStorage.setItem('ps_kernel_url', url) : localStorage.removeItem('ps_kernel_url')
                    const token = kernelTokenInput.trim()
                    token ? localStorage.setItem('ps_kernel_token', token) : localStorage.removeItem('ps_kernel_token')
                    window.location.reload()
                  }} color="var(--accent)">Connect</Btn>
                  <span style={{ ...hint, fontSize: 11 }}>reloads the app</span>
                </div>
                <div style={{ ...hint, fontSize: 11, marginBottom: 14 }}>remote kernel: regions/datasets are stored on that machine, not here — e.g. ws(s)://host:port/ws</div>
              </>
            ) : (
              <>
                <div style={{ display: 'grid', gap: 6, marginBottom: 4 }}>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>host</span>
                    <input value={sshHost} onChange={(e) => setSshHost(e.target.value)} placeholder="gpu.lab.edu or 1.2.3.4" spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>port</span>
                    <input value={sshPort} onChange={(e) => setSshPort(e.target.value)} placeholder="22" spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>username</span>
                    <input value={sshUser} onChange={(e) => setSshUser(e.target.value)} spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                  </label>
                  <div style={{ display: 'flex', gap: 6, margin: '2px 0' }}>
                    {(['password', 'key'] as const).map((auth) => (
                      <button key={auth} onClick={() => setSshAuth(auth)} disabled={isRemoteConnected()}
                        style={{ flex: 1, padding: '4px 8px', fontSize: 11, borderRadius: 4, cursor: isRemoteConnected() ? 'default' : 'pointer',
                          border: `1px solid ${sshAuth === auth ? 'var(--accent)' : 'var(--line-strong)'}`,
                          color: sshAuth === auth ? 'var(--accent)' : 'var(--text-1)',
                          background: 'var(--bg-2)', opacity: isRemoteConnected() ? 0.5 : 1 }}>
                        {auth === 'password' ? 'Password' : 'Key (.pem)'}
                      </button>
                    ))}
                  </div>
                  {sshAuth === 'password' ? (
                    <label style={{ display: 'grid', gap: 2 }}>
                      <span style={{ ...hint, fontSize: 11 }}>password</span>
                      <input type="password" value={sshPassword} onChange={(e) => setSshPassword(e.target.value)} spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                    </label>
                  ) : (
                    <>
                      <label style={{ display: 'grid', gap: 2 }}>
                        <span style={{ ...hint, fontSize: 11 }}>key path</span>
                        <div style={{ display: 'flex', gap: 6 }}>
                          <input value={sshKeyPath} onChange={(e) => setSshKeyPath(e.target.value)} placeholder="~/.ssh/gpu.pem" spellCheck={false} disabled={isRemoteConnected()} style={{ ...inp, flex: 1 }} />
                          <Btn onClick={browseSshKeyPath} disabled={isRemoteConnected() || !inTauri()}
                            title={inTauri() ? undefined : 'file picker is only available in the desktop app'}
                            style={{ flexShrink: 0 }}>Browse…</Btn>
                        </div>
                      </label>
                      <label style={{ display: 'grid', gap: 2 }}>
                        <span style={{ ...hint, fontSize: 11 }}>key passphrase (optional)</span>
                        <input type="password" value={sshKeyPassphrase} onChange={(e) => setSshKeyPassphrase(e.target.value)} placeholder="passphrase (encrypted keys only)" spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                      </label>
                    </>
                  )}
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>remote repo dir</span>
                    <input value={sshRepoDir} onChange={(e) => setSshRepoDir(e.target.value)} placeholder="~/parametic-report" spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>python path (optional)</span>
                    <input value={sshPythonPath} onChange={(e) => setSshPythonPath(e.target.value)} placeholder="/opt/conda/bin/python (where torch lives)" spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>model (optional)</span>
                    <input value={sshModel} onChange={(e) => setSshModel(e.target.value)} placeholder="Qwen/Qwen2.5-1.5B-Instruct" spellCheck={false} disabled={isRemoteConnected()} style={inp} />
                  </label>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 4px' }}>
                  {isRemoteConnected() ? (
                    <Btn onClick={sshDisconnect} color="var(--danger)">Disconnect</Btn>
                  ) : (
                    <Btn onClick={sshConnect} disabled={sshConnecting} color="var(--accent)">{sshConnecting ? 'Connecting…' : 'Connect'}</Btn>
                  )}
                  {sshStatus && (
                    <span style={{ ...hint, fontSize: 11 }}>
                      {sshStatus.state === 'connecting' && 'authenticating…'}
                      {sshStatus.state === 'starting-kernel' && 'starting remote kernel…'}
                      {sshStatus.state === 'forwarding' && 'forwarding…'}
                      {sshStatus.state === 'connected' && 'connected'}
                      {sshStatus.state === 'disconnected' && 'disconnected'}
                      {sshStatus.state === 'error' && `error: ${sshStatus.detail ?? 'unknown'}`}
                    </span>
                  )}
                </div>
                <div style={{ ...hint, fontSize: 11, marginBottom: 14 }}>runs the studio kernel over SSH on a remote GPU box; the app tunnels to it at localhost:8422.</div>
              </>
            )}

            <div className="section-h" style={{ marginBottom: 6 }}>Kernel</div>
            <div style={{ display: 'grid', gap: 6, marginBottom: 4 }}>
              {(['python_path', 'kernel_dir', 'model'] as const).map((k) => (
                <label key={k} style={{ display: 'grid', gap: 2 }}>
                  <span style={{ ...hint, fontSize: 11 }}>{k}</span>
                  <input value={config[k] ?? ''} onChange={(e) => setC(k, e.target.value)} spellCheck={false} style={inp} />
                </label>
              ))}
              <label style={{ display: 'grid', gap: 2 }}>
                <span style={{ ...hint, fontSize: 11 }}>datasets_dir</span>
                <input value={config.datasets_dir ?? ''} onChange={(e) => setC('datasets_dir', e.target.value)} spellCheck={false}
                  placeholder="~/.parametic_studio/datasets (default)" style={inp} />
              </label>
              <label style={{ display: 'grid', gap: 2 }}>
                <span style={{ ...hint, fontSize: 11 }}>hf_token</span>
                <input type="password" value={config.hf_token ?? ''} onChange={(e) => setC('hf_token', e.target.value)} spellCheck={false}
                  autoComplete="off" placeholder="for gated datasets" style={inp} />
              </label>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 14px' }}>
              <Btn onClick={() => sendTo(focused(), { type: 'set_config', config })} color="var(--accent)">Save</Btn>
              <span style={{ ...hint, fontSize: 11 }}>applies on next app launch</span>
            </div>

            <div className="section-h" style={{ marginBottom: 6 }}>Models · cache</div>
            {installed == null && <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>loading…</div>}
            {installed?.length === 0 && <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>no cached models</div>}
            {installed?.map((it) => (
              <div key={it.id} style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11, marginBottom: 3 }}>
                <span className="mono" style={{ color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={it.id}>{it.id}</span>
                <span className="mono" style={hint}>{gb(it.size_mb)}GB</span>
                <span onClick={() => confirmClick(`cache:${it.id}`, () => sendTo(focused(), { type: 'delete_cached', model: it.id }))}
                  title="delete from HF cache — click twice" style={{ cursor: 'pointer', color: armed === `cache:${it.id}` ? 'var(--danger)' : 'var(--text-2)', whiteSpace: 'nowrap' }}>{armed === `cache:${it.id}` ? 'sure?' : '×'}</span>
              </div>
            ))}

            <div className="section-h" style={{ margin: '14px 0 6px' }}>Appearance</div>
            <Btn onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}>{theme === 'dark' ? '☾' : '☀'} theme: {theme}</Btn>
          </div>
        )
      })()}
      {toasts.length > 0 && (
        <div className="toast-stack">
          {toasts.map((t) => <div key={t.id} className="toast" onClick={() => dismissToast(t.id)} title="dismiss">{t.text}</div>)}
        </div>
      )}
    </div>
  )
}
