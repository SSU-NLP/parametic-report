# parametic-studio — 핸드오프 (세션 재개용)

> 콜드 재개용. 이 문서 먼저 읽고, 설계는 `STUDIO_VIEW_ARCH.md`(뷰/probe/멀티모델 아키텍처)·
> `STUDIO_UI_SPEC.md`(셸/테마/기록)를 참조. 마지막 커밋: `dd53850` (브랜치 `ui/redesign`).

## 0. 무엇 / 어디에

기존 batch 플랫폼을 대체하는 **인터랙티브 멀티모델 추론 스튜디오**. 살아있는 커널 + VS Code풍 멀티뷰 웹.
- `parametic_studio/` — `device.py`(mps bf16), `kernel/model_session.py`(로딩·디코드·probe·spot),
  `api.py`(FastAPI WS·다중세션), `catalog.py`(로드 가능 모델).
- `studio_web/` — React+Vite+TS, opencode 다크. 단일 파일 `src/App.tsx`(상태·소켓·뷰·그리드 전부).
- `tests/studio/` — tiny eager 모델 CPU + WS TestClient. **`python -m pytest tests/ -q` → 39 passed.**

## 1. 실행법

```bash
# 커널(기본 Qwen2.5-1.5B-Instruct, mps bf16). env PARAMETIC_STUDIO_MODEL로 변경.
python -m parametic_studio.api            # :8000. 첫 모델 다운로드/로드까지 대기. /docs 200이면 up.
# 웹
npm --prefix studio_web run dev           # :5173 (또는 preview_start "studio_web", launch.json에 등록됨)
# 검증
python -m pytest tests/ -q                # 39 green
cd studio_web && npx tsc --noEmit -p .     # 0 (타입체크가 권위 — vite HMR "Failed to reload" 로그는 빠른 편집 잔재일 수 있음)
```
**함정: `api.py`/`kernel` 수정 후 커널을 반드시 재기동**(uvicorn.run에 reload 없음). 프론트는 HMR 자동.
프로세스 죽일 때 `ps ax | grep parametic_studio.api | grep -v grep | awk '{print $1}' | while read p; do kill $p; done` (pkill 금지 — 셸 죽음).

## 2. 완료된 것 (committed dd53850)

- **커널**: eager 로딩, 수동 그리디 디코드(stop/eos), **probe 구독**. probe:
  - `attention` — **전체 causal 정사각 N×N**(prefill 스텝이 P개 쿼리 행 emit + decode 1행/스텝 → `attn_rows`). head-mean `[L,kv]`. 드릴다운=마지막 스텝 per-head `[heads,kv]`.
  - `activation` — forward hook으로 모듈(self_attn·mlp) 출력 norm `[L,M]`.
  - `logitlens` — 레이어별 residual→final norm→lm_head top-1 `{token,prob}`.
  - `compute_spot(examples)` — |grad×param| 누적 → `{layers, modules, grid[L][M]}`. **현재 동기·원샷(루프 블로킹).**
- **api**: WS, **다중 세션 레지스트리**(model_id→ModelSession), **비블로킹 지연로드**(`asyncio.to_thread` + per-model `asyncio.Lock`), 모든 이벤트 `model` 태그, `loading`/`opened`. `catalog`.
- **device**: `resolve_device`(auto→mps/cuda/cpu, cuda 없으면 cpu fallback), `dtype_for`(cuda·**mps=bf16**, cpu=fp32). ⚠️ mps fp16은 Qwen 오버플로 → 쓰레기.
- **웹**: 5뷰(output/attention/activations/logitlens/spot) = **드래그 탭**, **2단계 그리드(열×타일)** — 우측`[|]`·하단`[-]` split, 2축 리사이즈, DnD reorder/move + 파란 드롭 프리뷰. 좌 Explorer, 우 **Chat dock(focus 모델 바인딩)**, 상단 모델 칩+`[+model]`+`[sync]`, 하단 status.
- **멀티모델**: **모델당 WS 소켓 1개**(브로드캐스트 시 stop이 다음 generate 삼키는 문제 회피), 모델별 상태(`data[modelId]`), **sync 브로드캐스트**(한 프롬프트→전 모델), 타일=(모델×뷰).

## 3. WS 계약 (모델당 소켓, 모든 server→client 이벤트에 `model`)

```
client→server: catalog · open{model} · close{model} · generate{model,prompt,max_tokens,probes:[attention,activation,logitlens]}
               · stop{model} · drilldown{model,layer} · spot{model,examples}
server→client: catalog{models} · loading{model} · opened{model} · closed{model}
               · token{model,step,token_id,text} · attention{model,step,shape,data:[L][kv]}  // prefill 다수행→정사각
               · activation{model,step,shape,data:[L][M]} · logitlens{model,step,layers:[{token,prob}]}
               · perhead{model,layer,shape,data:[heads,kv]} · spotmap{model,layers,modules,grid} · done{model,reason}

모델 load=open(지연로드,비블로킹) / unload=close(`ModelSession.close()` hook 제거 + SESSIONS pop + mps empty_cache로 메모리 해제). 프론트 칩 ×로 unload(최소 1개 유지, 닫은 모델 바인딩 타일은 남은 모델로 재배정).
```
재사용 자산: `scripts/save_masked_model.py:apply_mask`(→knob hook), `scripts/visualize_heatmaps.py:parse_param`,
`scripts/create_approx_spot_masks.py`(top-k/grad×param). torchvision는 **제거됨**(pyenv `_lzma` 누락 → transformers→torchvision→lzma 깨짐, 텍스트 LLM엔 불필요).

## 4. 다음 마일스톤 (합의됨 — 여기서 재개)

**A. 프리셋 데이터셋 + 스트리밍 spot (로컬, 클라우드 불필요)**
- python·java 코드 스니펫 작은 세트 번들 → `[python spot]`·`[java spot]` 버튼.
- 커널: `compute_spot`을 per-example `spot_step(example, acc)`로 쪼갬(running grid 반환) + **per-weight top-k% 마스크도 산출**(knob용).
- WS: 각 스텝 `asyncio.to_thread`(비블로킹) → `spot_progress{i,total,layers,modules,grid}` 스트림, 끝 `spotmap`.
- 프론트 spot 뷰: **progress bar + 라이브 grid 갱신 + 현재 top-k% 아웃라인**("지금 spot 여기"가 떠오름). TDD: spot_step 누적/마스크.

**B. knob (개입) — 믹싱보드 클라이맥스**
- 커널: `set_intervention(region, op, alpha)` / `clear` — **forward pre-hook(비파괴·가역·토글)** 으로 선택 영역 weight를 ×0(suppress)/×α(scale)/substitute(zero·mean·random). region = spot 마스크 또는 (layer,module) 셀. apply_mask 로직 재사용.
- WS `intervene`/`clear`. 프론트: knob 패널(spot/셀 선택 + op + α 슬라이더) + **baseline ｜ intervened diff**(재생성: output·PPL). 임의 마스크는 **코드 escape**(후속).
- 설계 합의: GUI = 흔한 영역(spot·모듈·레이어)+라이브 diff. 임의/정밀/스윕 = 코드. GUI의 값어치는 "노브→즉시 diff" 루프.

**C. 클라우드(VESSL, M3) — 큰 모델/실스케일 spot일 때만**
- `VesslBackend`: workspace(exposed port)에서 커널 실행 → api가 WS 프록시. vesslctl skill. **데모 spot/knob은 로컬로 가능하니 선행 아님.**

## 5. 보류 백로그
- M2 기록/리플레이(task #10–12, recordings.py + Timeline 스크러버) — causal 그리드 디스크 저장. 현재 전부 휘발성.
- 토큰↔어텐션 오버레이(BertViz식, 생성 토큰 클릭→입력 단어 하이라이트).
- logit lens 개선(초반 echo 착시 → "정답 토큰의 레이어별 확률" emergence-curve 또는 tuned-lens).
- 어텐션 정사각 **시각적 1:1 셀**(현재 데이터는 N×N이나 행 2px·열 넓어 납작하게 보임).
- 드릴다운은 **마지막 스텝만**(replay drilldown은 M2 + `{available:false}` 계약).
- 비-멀티태스커 UX(도그푸드): 점진적 공개(기본 단순 + "비교" 한 클릭), 동시갱신 부담 완화.

## 6. 도그푸드 발견 (했던 것)
- 🐞 모델 지연로드 이벤트루프 블로킹 → **고침**(to_thread + loading 표시).
- spot 계산도 같은 블로킹 패턴 → **A에서 to_thread+스트리밍으로 고칠 것**.
- UX: 시작 전 결정 과다·동시갱신·DnD-split 발견성 낮음 → 점진적 공개 권장.

## 7. 규약
- python = pyenv `python`(torch+mps). TDD(tiny eager 모델 CPU, fake tokenizer). ponytail(최소 코드).
- 커밋은 **요청 시에만**, 메시지 끝에 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- 생성 dir·node_modules·체크포인트 커밋 금지(gitignore). studio_web/.gitignore가 node_modules 제외.
