# parametic-studio — UI/기능 스펙 (결정 확정본)

> `HANDOFF.md`·`reports/parametic-studio-plan.md`의 상위 결정을 따르고, 여기서 **UI/IA와 기능 우선순위**를
> 확정한다. 구현 착수 전 사용자 합의용 문서. (2026-06-24)

## 0. 한 줄

기존 batch job 플랫폼을 대체하는 **인터랙티브 연구 툴** — 살아있는 커널(모델 상주) 위에
**Zed 풍 멀티패널 IDE**. 연구자용이지만 일반 사용자도 직관적으로(점진적 공개).

## 1. 확정 결정 (이번 세션)

| 항목 | 결정 |
|---|---|
| 레이아웃 | **Zed 도킹형** — 좌 Explorer dock · 중앙 탭 캔버스 · 우 Chat dock · 하단 Timeline · 상단 타이틀바 · 상태바 |
| MVP | **라이브 어텐션 채팅** (핸드오프 M1) — 커널 + 토큰 스트림 + 어텐션 히트맵 + 정지 |
| 테마 | **다크 기본** (opencode 풍 웜-뉴트럴 모노크롬), 라이트 토글 후속 |
| 뷰 탭 우선순위 | Attention(토대) → **Spot map(연구 헤드라인)** → Spot 개입(클라이맥스) → Tensor(보조) |
| 프론트 | React + Vite + TypeScript (핸드오프 확정) |
| 개발 | TDD, tiny eager 모델 CPU 테스트로 GPU 없이 CI |

### 데모 내러티브 vs 빌드 순서 (중요)
- **빌드 순서**는 어텐션 먼저 — 살아있는 커널·스트리밍이 모든 기능의 토대이고 spot 개입도 같은 커널의
  forward-hook을 재사용한다.
- **데모 내러티브**는 Spot map이 헤드라인 — "코딩 능력이 파라미터 어디에 사는가" → spot을 끄면 출력
  붕괴, 같은 크기 random은 멀쩡(diff)이 클라이맥스. 어텐션은 "내부를 실시간으로 본다"는 보조 훅.

## 2. 레이아웃 (Zed 도킹)

```
┌─ 타이틀바: 모델명 · eager·dtype · [● live · mps] · [stop] ───────────────┐
├──────────┬───────────────────────────────────┬───────────────────────────┤
│ Explorer │  Canvas (tabs)                     │  Chat                     │
│  (좌 dock)│  [Attention][Spot map][Tensor][+]  │  (우 dock)                │
│          │                                    │                           │
│ Metadata │   ← 활성 뷰 본문 →                  │  대화 스트림              │
│ Tensors  │                                    │                           │
│  layers… │                                    │  [ ask… ↑ ]               │
├──────────┴───────────────────────────────────┴───────────────────────────┤
│ Timeline: ▶ step ├──●────┤ 12 / 24   (리플레이 스크러버, M2)               │
├────────────────────────────────────────────────────────────────────────────┤
│ 상태바: session a3f9 · mps · fp16            18.4 tok/s · 12 tokens          │
└────────────────────────────────────────────────────────────────────────────┘
```

- **모든 dock은 토글로 접기/펼치기.** 기본 상태 = Chat + Attention만 노출(일반 사용자 모드).
  Explorer/Spot/Tensor는 펼쳐야 보임(연구자 모드). → 점진적 공개.
- **Explorer dock**: safetensors 트리(Metadata / Tensors → embed_tokens / layers(N) → layer.k →
  self_attn·mlp·norms). 노드 클릭 → 활성 캔버스 뷰를 해당 레이어/모듈로 포커스.
- **Canvas 탭**: 사진의 "모델 뷰 전환 탭". 탭은 Explorer 클릭 또는 `+`로 추가.
- **Chat dock**: 생성을 구동. 입력 → WS `generate` → 토큰 스트림. 정지 버튼은 타이틀바/상태바.
- **Timeline**: 토큰 스텝 스크러버. 긁으면 활성 뷰(어텐션)가 해당 스텝으로 갱신(리플레이, M2).
- **상태바**: device·dtype·session·tok/s.

## 3. 테마 토큰 (opencode TUI 다크 — `opencode.ai/DESIGN.md` 기반)

출처: `npx getdesign add opencode.ai` → `opencode.ai/DESIGN.md`(벤더링됨). opencode **마케팅 사이트는
라이트(크림 `#fdfcfc`)**, **다크는 제품 TUI 표면(`#201d1d`/`#302c2c`)에만** 사용. 우리는 IDE(=TUI류)이고
다크 기본이므로 **opencode TUI 다크 표면을 베이스로 채택**한다.

```css
/* surfaces — opencode TUI dark */
--bg-0: #1A1717;  /* app gutter (surface-dark보다 한 단 어둡게) */
--bg-1: #201D1D;  /* 패널/dock (opencode surface-dark) */
--bg-2: #302C2C;  /* 라이즈드/활성 탭/프롬프트 행 (surface-dark-elevated) */
/* borders — 플랫, 1px 헤어라인만 */
--line:        rgba(253,252,252,0.10); /* subtle */
--line-strong: #646262;                /* opencode hairline-strong */
/* text on dark */
--text-0: #FDFCFC; /* on-dark primary */
--text-1: #9A9898; /* ash — secondary */
--text-2: #6E6E73; /* stone — muted/hint */
/* semantic — opencode Apple HIG 램프 (TUI 용도) */
--accent:  #007AFF; /* 활성/선택/링크 */
--live:    #30D158; /* live/success */
--danger:  #FF3B30; /* stop/error */
--warning: #FF9F0A;
/* type — 단일 모노 서체가 정체성 (Berkeley Mono는 유료 → JetBrains Mono로 대체) */
--mono: "Berkeley Mono", "JetBrains Mono", "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
/* shape — 컨테이너 직각, 컨트롤만 4px */
--radius-ctrl: 4px; /* 버튼/입력/배지/스니펫만 */
--radius-box:  0px; /* 패널/dock/탭/트리행/캔버스 = 직각 */
```

**정체성 규칙 (opencode에서 그대로 가져옴):**
- **100% 모노스페이스.** sans 없음 — 트리·산문·버튼·수치·탭 라벨 전부 `--mono`. 가중치 400/500/700.
- **ASCII 브래킷 마커**가 아이콘. 트리 펼침 `[+]`/`[-]`, 상태 `[x]`/`[●]`, 섹션 라벨 `[ tensors ]`.
  아이콘 폰트/SVG 대신 브래킷 글리프. (Explorer 트리·상태바·섹션 헤더에 적용)
- **플랫.** 섀도·그라데이션·블러 0. 분리 신호는 1px `--line` 헤어라인뿐.
- **컨테이너 직각(0px), 컨트롤만 4px.** 패널/탭/트리는 샤프, 버튼/입력/배지만 둥근 4px.
- **accent 절제.** Apple 블루는 활성/선택/링크에만. live/stop은 시맨틱 램프.
- IDE 밀도용 사이즈: UI 베이스 13px/1.5, 캡션 12px, 섹션 라벨/헤딩 13–16px/700.
- 어텐션 히트맵 ramp: 다크 베이스 단일 시퀀셜 `#201D1D → #007AFF`(low→high), 1램프 원칙.
- 기존 `spot_viz_examples.html` 히트맵/depth/concentration/scatter는 이 다크 토큰으로 리스킨해 이식.
- 벤더링된 `opencode.ai/DESIGN.md`가 SSOT. 컴포넌트/토큰은 거기 이름을 직접 참조(패러프레이즈 금지).

## 4. 캔버스 뷰 (탭)

| 탭 | 내용 | 데이터 | 마일스톤 |
|---|---|---|---|
| **Attention** | 스텝별 `[L,Q,KV]` head-mean 히트맵, 레이어 클릭→per-head `[heads,Q,KV]` 드릴다운, 토큰-어텐션 오버레이 | WS `attention`/`drilldown` 프레임 | **M1 (MVP)** |
| **Spot map** | 레이어×모듈 중요도 히트맵 + top-k% spot 하이라이트, depth profile, concentration | 기존 batch artifact 로드(재계산 X) | M4 직전 (헤드라인) |
| **Spot 개입** | 마스크 로드 → suppress(×0/×α)·amplify(×α>1)·substitute 토글 → 재생성 diff, PPL/지표 패널 | forward-hook(비파괴·가역) | M4 (클라이맥스) |
| **Tensor** | 선택 텐서 shape/dtype/값 분포 통계 | Explorer 클릭 연동 | 보조 |

## 5. 커널 / 서버 / 전송 (요약 — 상세는 plan.md)

- **kernel**: `attn_implementation="eager"` 강제, device {cuda:bf16, **mps:bf16**, cpu:fp32}
  (mps fp16은 Qwen 가중치 오버플로 → NaN/`!!!!` 쓰레기 출력. 실측 확인됨),
  수동 디코드 루프(스텝마다 토큰 yield + 어텐션 집계 emit), 정지 지원.
- **server** `parametic_studio/api.py`: WS(`generate`/`stop`/`drilldown` ↔ `token`/`attention`/`done`)
  + REST(카탈로그·세션·리플레이). WS 계약은 `HANDOFF.md §5` 초안 사용, M1에서 프론트와 고정.
- **재사용**: `scripts/save_masked_model.py:apply_mask` (개입은 hook으로 비파괴화),
  `scripts/visualize_heatmaps.py:parse_param` (param→layer/module),
  `scripts/create_approx_spot_masks.py:resolve_device` (**mps 분기 추가**)·`stable_top_indices`,
  `parametic_platform/catalog.py` 패턴 이식.
  주의: 핸드오프가 적은 `arch_adapter.py`는 실제로는 `visualize_heatmaps.py`에 `parse_param`이 있고
  `is_target`는 없음(필터는 runner 레벨). `resolve_device`는 **mps 미지원** → 분기 추가 필수.

## 6. 마일스톤 (TDD, tests first)

- **M0** 커널 스켈레톤 + 로컬 mps 스트리밍 — eager 로딩, 수동 디코드 루프, device 선택,
  WS `generate`/`stop`→`token`/`done`. 테스트: tiny eager 모델(CPU)로 토큰 산출·정지.
- **M1 (MVP)** 어텐션 라이브 — head-mean 집계 `[L,Q,KV]`, per-head 드릴다운, 프론트 2D 히트맵 +
  토큰-어텐션 오버레이 + 정지. 테스트: 프레임 shape/집계 수치, drilldown shape.
- **M2** 리플레이(디스크 기록·스크러버) + 3D 어텐션 큐브(three.js).
- **M3** 백엔드 추상화 + VESSL workspace 커널(discover/attach, WS 프록시).
- **M4** Spot map + Spot 개입 GUI(hook, 재생성 diff, 지표).
- **M5** IA 마감, 구 `parametic_platform/` 제거(MVP 도달 후).

## 7. 검증

- 유닛(CPU, GPU 불필요): 디코드 루프 토큰·어텐션 프레임 shape·hook ×0/×α/치환 수치·기록 roundtrip. WS mock.
- 로컬 mps 수동: 사용자 Mac에서 3B 모델, chat → 라이브 어텐션.
- 프론트: Vitest + preview MCP로 패널/스트리밍 확인.

## 8. 스케일 정책 (모델 크기 가정)

**핵심 원칙: 맵은 모델 크기에 비례해 촘촘해지지 않는다.** 두 시각화 모두 집계이고, 밀도는
파라미터 수가 아니라 **고정 표시 예산(display budget)** 이 정한다. 스케일이 커지면 달라지는 건
*커널 상주 메모리*(백엔드 라우팅)이지 *픽셀*이 아니다.

### 모델 크기 ↔ 백엔드 티어
| 티어 | 모델 | 백엔드 | 메모(fp16) | 비고 |
|---|---|---|---|---|
| 로컬 데모 기본 | ≤3B | Local mps fp16 | ~6GB | 편안한 데모 디폴트 |
| **로컬 최대** | **~8B** | Local mps | ~16GB | 통합메모리 24GB+ 필요, MVP 상한 |
| 원격 | 13B–100B+ | **VESSL cuda (M3)** | 멀티-GPU/양자화 | UI 동일, 커널만 무거움 |

- **MVP/로컬 상한 = 8B.** 100B는 UI에서 배제하지 않음 — `LocalBackend`/`VesslBackend` 추상화가
  자연히 흡수(M3). 단 MVP 데모 타깃 아님.
- spot map은 모델-크기 텐서가 아니라 **작은 집계 요약(layer×module CSV)** 만 로드 → 크기 무관.
- 개입(M4)은 GPU 상주 모델에 boolean 마스크 적용 → 크기 무관하게 저렴.

### 고정 표시 예산 (스케일 불변 — 처음부터 적용)
- **어텐션 kv축**: ≤256열로 binning. 상위 attended 토큰은 정확히, 나머지 binned. (kv 밀도는 모델
  크기가 아니라 **컨텍스트 길이**가 정함 — 어차피 묶어야 함, 핸드오프 §5.2)
- **어텐션 layer축**: 전부 렌더, >48이면 가상 스크롤.
- **드릴다운** `[heads,Q,KV]`: 레이어 1개 온디맨드. heads 8B≈32 / 100B≈128 → 문제없음.
- **spot map**: layer×module 집계(상한 ~120×9). per-parameter는 **절대 풀 격자로 안 그림** —
  샘플 scatter + top-k% 하이라이트만(3B의 weight 하나도 1600만 셀이라 어느 크기든 동일 원칙).

## 9. 라이브 ↔ 리플레이 상태 모델 (충돌 방지)

채팅(라이브 생성)과 Timeline 스크러버는 **충돌하지 않는다** — 둘은 서로 다른 축을 제어하고,
"라이브"는 커서의 특수 위치일 뿐. 단일 진실원: `timeline`(스텝 시퀀스) + `cursor`(현재 스텝) + `follow`.
(핸드오프 §5.3 "스트리밍 또는 리플레이가 한 메커니즘"의 구현형.)

- **두 내비게이션 축:**
  - Chat 선택 = 어느 **run(턴)** 인가. 과거 어시스턴트 메시지 클릭 → 그 턴 기록 로드(리플레이, GPU 불필요).
  - Timeline `cursor` = 그 run 안의 어느 **스텝**인가.
- **follow-live 토글 (`tail -f` 패턴):**
  - 생성 중 + follow ON → cursor가 최신 스텝으로 자동 전진, 썸은 라이브 엣지에 붙음.
  - 생성 중 스크럽 → follow **자동 detach**, cursor 고정·캔버스 정지. 생성은 **백그라운드 계속**,
    스크러버 max는 계속 증가. `[●] jump to live` pill 노출 → 누르면 re-attach.
  - 생성 종료 → 순수 리플레이.
- **한 줄 규칙:** `live` ⇔ cursor가 active run의 `latest`에 pin된 상태. 그 외 전부 replay.
- 동시성: 라이브 턴이 도는 동안 과거 턴 기록 열람은 GPU-free라 독립. 커널은 한 번에 1개 run만 생성.

## 10. Settings & 환경(Environment)

Settings는 **캔버스 탭**(plan.md 패널 목록 Chat/Attention/Spot/**Settings** 중 하나).
진입점 3개: 타이틀바 우측 기어 `[≡]` · **상태바 env 칩 클릭** · 커맨드 팔레트.

**환경 가시성 2층:**
- 글랜스: 상태바가 항상 `device · dtype · tok/s` 노출(클릭 → 아래 Session/Env 섹션 점프).
- 풀: Settings → **Session / Environment** = opencode 풍 모노 key-value 리드아웃(manpage 스타일).

**Settings 섹션:**
1. **Session / Environment (읽기 전용 진단)** — "실제로 뭐가 돌고 있나" 신뢰용:
   backend(Local/VESSL + host) · 해석된 device(mps / cuda:0 / cpu) · `mps_available`/`cuda_available` 프로브 ·
   dtype · model(id · revision · params · layers · heads) · `attn_implementation=eager`(이유 주석) ·
   memory(모델 풋프린트 · free/used) · `HF_TOKEN` 존재여부(**값 절대 노출 X**) · 커널 uptime.
2. **Generation** — max_tokens · do_sample · temperature · top_p · seed.
3. **Capture/Recording** — **기록 품질 기본값**(aggregate / full, §11) · 열 binning 표시 임계 · layer 가상화 임계.
   - full은 GB급이라 **per-run 토글**(Chat 입력 옆)로도 켤 수 있고, 켤 때 **예상 크기 확인 모달**(§11)을 거침.
4. **Appearance** — theme(dark 기본 / light) · 폰트 크기 · 브래킷 마커 강도.
- 모델 로드/교체(catalog 선택)는 새 커널을 띄우는 별도 **launch 플로우** — Settings 아님.

## 11. Chat stream 추적 단위 & 기록 메커니즘

**핵심: 어텐션은 인과적(causal)이라 런당 고정 그리드 `[L, N, N]`(N=prompt+생성) 한 개로 표현.
윗삼각형 = 못 보는 미래(회색), 아랫삼각형만 값.** 행 t는 step t에 확정되고 이후 안 변하므로
최종 그리드가 모든 step에 대해 무손실 → 별도 per-step 파일 불필요, prefill 특별취급 불필요.

### 단위 / 주소
- **run(턴)** = `generate()` 1회. 채팅 user→assistant 쌍에 1:1.
- **step** = 디코드 반복 1회(토큰 1개) = 그리드의 행 하나.
- 주소: `session_id`(**UUID**, 커널 시작 시 생성·재시작 시 새 값) / `run_id`(세션 내 **0-base 정수**) / `step`(int).
- **prefill = 맨 앞 P줄**(질문 토큰), decode = 그 뒤 줄. 같은 그리드의 앞부분일 뿐(Q 붕괴 이슈 소멸).
- chat ↔ recording: assistant 메시지가 `run_id` 보유 → 클릭 시 그 run 로드. Chat=run 선택기·Timeline=step 선택기(§9).

### 기록 품질 모드 (저장장치 여유에 따라 선택)
| 모드 | 저장물 | 런당 크기 | per-head replay |
|---|---|---|---|
| **aggregate (기본)** | head 평균 그리드 `[L,N,N]`(아랫삼각만) | MB급 (데모 ~10MB · 8B 긴거 수백MB) | 본 것만 캐시, 미스 시 `{available:false}` |
| **full (옵트인)** | + raw per-head `[L,H,N,N]` | **GB급** (8B 2k ~4GB) | **어디서나 가능** |
- 기본 = aggregate. full은 **Settings 기본값** 또는 **per-run 토글**(다음 생성에만 적용)로 선택.
- **full 선택 시 확인 모달 필수**: 모델·컨텍스트(L·H·N) 기반 예상 크기 산출 →
  "이 설정 기준 런당 약 **~X GB**. 컨텍스트·max_tokens에 비례해 커집니다. 계속?" → 취소 시 aggregate로 진행.
  (추정식 ≈ `L×H×N²/2×2B`.)

### 스트리밍 = 기록 (dual-write, 한 루프)
```
forward(output_attentions=True)          # [L,heads,q,kv]
row = head_mean(attn)                     # 헤드 평균 → 이번 step 행 [L,kv]
grid[:, step, :kv] = row                  # 고정 causal 그리드 채움 (윗삼각=회색)
if mode == "full": heads_store[step] = attn   # raw per-head도 저장
tok = sample(logits)
ws.emit(token:     {run, step, token_id, text})
ws.emit(attention: {run, step, row})      # 라이브는 이번 행만 전송
```
라이브 = 방금 행 렌더(+이전 행 보임), 리플레이 = 그리드 행 슬라이스 → 동일 경로.
스크럽 = 커서까지 행 reveal(나머지 회색). 화면이 너무 넓으면 **표시할 때만** 열 binning(저장은 원본).

### 저장 레이아웃 (`recordings.py`)
```
recordings/<session_id(UUID)>/<run_id(int)>/
  index.json    # prompt·model(L,H)·N·prompt_len·n_steps·mode·status·created_at
  tokens.jsonl  # {step,token_id,text}
  attn.f16      # head 평균 causal 그리드 [L,N,N] (아랫삼각만 값)
  heads/        # per-head: aggregate=온디맨드 캐시 / full=전부 <step>_<layer>.f16=[H,kv]
```
- 위치 = **로컬 디스크**(기존 artifact 디스크). raw(GB)는 full 모드에서만 닿음.
- **drilldown 미스 계약(Codex #1 해소):** aggregate에서 캐시에 없는 (step,layer) 요청 →
  `{available:false}` + UI "이 스텝은 라이브 중 드릴다운 안 함" 안내. full 모드는 항상 히트.

### WS 프로토콜 보강
핸드오프 §5에 `run` 추가. `token`/`attention`에 `run`+`step` 필수(예: `"run":3,"step":12`),
`drilldown`=`{run,step,layer}` (replay에서도 동일, 미스는 위 계약). 최종 고정 M1에서 프론트와.

## 12. 남은 확정 대기

1. 추가 UI 예시 이미지(사용자 제공 예정) → 디테일 조정.
2. ~~리플레이 기록 보존 정책~~ → **확정(§11):** 기본 aggregate(MB), full(GB)은 Settings/per-run 옵트인 + 확인 모달.
3. Spot 정의 입력: 기존 batch artifact 재사용(현 전제) 확정 여부.
