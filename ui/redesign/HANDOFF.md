# parametic-studio — 로컬 착수 핸드오프

> 이 문서는 **원격 세션에서 작성** → 브랜치 push 후 **로컬(Mac/mps)에서 시작**하기 위한 핸드오프다.
> 전체 설계는 `reports/parametic-studio-plan.md`를 먼저 읽어라. 이 문서는 *로컬에서 바로 손대는 데*
> 필요한 환경·계약·첫 스텝만 담는다.

## 0. 무엇을 만드는가 (한 줄)

기존 batch job 데모 플랫폼(`parametic_platform/`)을 **완전 대체**하는 인터랙티브 연구 툴.
**커널(모델 메모리 상주 추론 세션) + 스튜디오(멀티패널 IDE)** 구조. R Studio/Matlab 비유.

## 1. 확정된 결정 (2026-06-23)

- 기존 batch 플랫폼 **완전 대체** (단, studio MVP 도달 전까지 `parametic_platform/`는 **삭제 금지·유지**).
- 프론트 **React + Vite + TypeScript** (빌드 도입). 기존 buildless Preact 버림.
- 데모 1순위 **작은 모델(예: Llama-3.2-3B) + 로컬 mps**. cuda는 원격에 올린 뒤 검증.
- MVP = **기능 1+2 어텐션(라이브 스트림 + 리플레이)**. spot 제어(기능 3)는 후속.
- 개발 방식 **TDD(tests first)** — tiny eager 모델 CPU 테스트로 GPU 없이 CI.

## 2. 아키텍처 요약

```
[studio_web]  React+Vite+TS 멀티패널 IDE (Chat / Attention / Spot / Settings)
     │  REST(카탈로그·세션·리플레이)  +  WebSocket(토큰·어텐션 프레임·정지·개입)
[parametic_studio/api.py]  세션 서버(FastAPI): 세션 lifecycle·WS 프록시·리플레이 저장
     │  KernelProtocol (WS 메시지 계약) — 커널 위치 무관 동일
[kernel]  살아있는 추론 프로세스(모델 상주)
     ├─ LocalBackend  : in-process, device = mps / cuda / cpu
     └─ VesslBackend  : VESSL **workspace**(exposed port)에서 동일 커널 실행 → API가 WS 프록시
```

핵심: **커널 코드는 어디서 돌든 동일**. 백엔드는 *어디서 띄우고 어떻게 붙느냐*(런처+전송)만 다르다.
VESSL **batch job으로는 인터랙티브 불가** → 살아있는 커널은 VESSL **workspace**로만.

## 3. 신설 repo 레이아웃 (로컬에서 만들 것)

```
parametic_studio/
  api.py              세션 서버(FastAPI) — WS 프록시·세션·리플레이
  catalog.py          모델/디바이스 레지스트리 (parametic_platform/catalog.py 패턴 이식)
  recordings.py       어텐션·토큰 기록 read/write (리플레이)
  kernel/
    model_session.py  로딩·수동 스트리밍 디코드 루프
    attention.py      어텐션 캡처·집계
    intervention.py   spot forward-hook (극소/극대/치환) — M4
    streaming.py      WS 이벤트 emit
  backends/
    base.py           Backend 인터페이스(launch/discover/attach)
    local.py          mps/cuda/cpu in-process
    vessl.py          workspace 런치·attach (M3)
studio_web/           React+Vite+TS — 패널: Chat/Attention/Spot/Settings
tests/studio/         pytest (tiny eager 모델, CPU)
```

## 4. 재사용 자산 (새로 짜지 말 것)

| 용도 | 재사용 | 메모 |
|---|---|---|
| spot 마스크 적용(극소=×0) | `scripts/save_masked_model.py:apply_mask()` | studio에선 **forward hook으로 비파괴·가역화** |
| param명→(layer, module) | `scripts/arch_adapter.py:parse_param/is_target` | 시각화·레이어 선택 |
| top-k% 마스크 생성 | `scripts/create_approx_spot_masks.py` | spot 정의 입력으로 로드(재계산 X) |
| 디바이스 선택 | `resolve_device("auto")` (create_approx_spot_masks.py) | **mps 분기 추가 필요** |
| 모델 로딩/생성 | `scripts/generation_sanity.py` | **`attn_implementation="eager"`로 변경** 필수 |
| 카탈로그 패턴 | `parametic_platform/catalog.py` | 이식·확장 |

## 5. 어텐션 — 가장 중요한 기술 계약 (함정 주의)

1. `output_attentions=True`는 flash-attn을 끄므로 **`attn_implementation="eager"` 강제**. 작은 모델 mps OK.
2. raw 어텐션 = O(layers×heads×seq²). 8B면 한 스텝 수 GB → **그대로 스트리밍 금지**.
   - **GPU 위에서 집계 후 전송**: 기본 프레임 = 레이어별 head-mean → `[layers, q, kv]`. kv 길면 다운샘플/top-k.
   - 드릴다운: 선택 레이어 1개만 per-head `[heads, q, kv]` 온디맨드.
3. **리플레이 = 기록 파일 읽기(GPU 불필요).** 생성 중 토큰+어텐션을 디스크에 기록 →
   "스트리밍 또는 리플레이"가 한 메커니즘으로 해결.

### WS 메시지 계약 초안 (서버↔클라)
```jsonc
// 서버 → 클라
{ "type": "token",     "step": 12, "token_id": 1234, "text": " def" }
{ "type": "attention", "step": 12, "shape": [L, Q, KV], "agg": "head_mean", "data": [...] }
{ "type": "done",      "reason": "eos" | "stopped" | "max_tokens" }
{ "type": "error",     "message": "..." }
// 클라 → 서버
{ "type": "generate", "session": "...", "prompt": "...", "max_tokens": 256 }
{ "type": "stop" }
{ "type": "drilldown", "step": 12, "layer": 7 }   // per-head 요청
```
(확정 아님 — M1에서 프론트와 맞추며 고정. typed-array/binary 전환은 성능 보고 결정.)

## 6. 로컬 환경 (Mac / mps)

```bash
# Python (커널·API). conda 또는 venv 무방.
python -m pip install torch torchvision   # mps 지원 빌드(Apple Silicon)
python -m pip install transformers accelerate fastapi "uvicorn[standard]" websockets pytest
# mps 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# 프론트
cd studio_web && npm create vite@latest . -- --template react-ts && npm install
npm install three                 # 3D 어텐션 큐브(M2)
```
- 모델 가중치는 HF. gated 모델은 `HF_TOKEN` 필요(메모: 우리 계정 Llama-3.2-3B 접근 O).
- dtype: mps=fp16/fp32, cuda=bf16, cpu=fp32.

## 7. 마일스톤 — M0/M1만 로컬 착수분

- **M0 커널 스켈레톤 + mps 스트리밍**
  - `kernel/model_session.py`: eager 로딩, 수동 디코드 루프(토큰 1개씩 yield), 정지.
  - `backends/local.py`: device 선택(mps/cuda/cpu, `resolve_device` + mps 분기).
  - `api.py`: WS 엔드포인트 — `generate`/`stop` → `token`/`done` 이벤트.
  - **테스트 먼저**: tiny eager 모델(CPU)로 디코드 루프 토큰 산출 / 정지 동작.
- **M1 어텐션 라이브 (MVP, 기능 1+2)**
  - `kernel/attention.py`: 스텝별 head-mean 집계 `[L,Q,KV]`, 드릴다운 per-head.
  - `api.py`: `attention` 프레임 emit, `drilldown` 처리.
  - `studio_web`: 2D 히트맵(현재 스텝) + 토큰-어텐션 오버레이 + 정지 버튼.
  - **테스트 먼저**: 어텐션 프레임 shape/집계 수치, drilldown shape.

(M2 리플레이+3D / M3 VESSL workspace / M4 spot GUI / M5 마감·구플랫폼 제거 — `reports/parametic-studio-plan.md` 참조.)

## 8. 사용자가 줄 것 / 열린 질문
1. **UI 예시**(멀티패널 레이아웃) → `studio_web` 패널 IA 확정.
2. 리플레이 기록 보존 정책(전부 vs 집계만; 디스크 용량).
3. spot 정의 입력: 기존 batch artifact 재사용(현 전제) vs studio 내 재계산.

## 9. 참고
- 전체 설계: `reports/parametic-studio-plan.md`
- 기존 플랫폼 동작/제약: `handoff.md`, `CLAUDE.md`
- VESSL: `scripts/vessl/`, vesslctl skill
