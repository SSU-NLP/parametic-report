# parametic-studio — 설계 계획 (구현 전, 승인 대기)

> 상태: **계획만**. UI 예시 수령 + 착수 지시 전까지 구현 없음.
> 결정(2026-06-23): 기존 batch 플랫폼 **완전 대체** / 프론트 **React+Vite+TS** /
> 데모 1순위 **작은 모델 mps 로컬** / MVP **어텐션 스트림+리플레이(기능 1+2)**.

## Context (왜 새로 트는가)

기존 `parametic_platform/`은 전부 **배치 job(run-to-completion)** 구조다 — VESSL에 job을
던지고 종료까지 폴링한 뒤 결과 파일을 내려받는다(`worker.py`의 `vessl_submit/wait/fetch`,
worker↔runner 계약: exit 0 + `manifest.json`). 이 구조로는 **chat 실시간 어텐션 스트리밍**과
**GUI spot 실시간 제어**가 불가능하다. 둘 다 *모델을 메모리에 올려둔 채 살아있는 추론 세션*을
전제로 하기 때문이다.

그래서 parametic-studio는 **커널(살아있는 추론 세션) + 스튜디오(멀티패널 IDE)** 구조다.
R Studio / Matlab 비유가 정확하다: 커널이 모델을 들고 있고, 위에 패널 UI가 붙는다.

## 아키텍처 — 3계층

```
[studio_web]  React+Vite+TS 멀티패널 IDE  (Chat / Attention / Spot / Settings)
     │  REST(카탈로그·세션·백엔드 탐색·리플레이)  +  WebSocket(토큰·어텐션 프레임·정지·개입)
[parametic_studio/api.py]  세션 서버(FastAPI): 세션 lifecycle, WS 프록시, 리플레이 저장
     │  KernelProtocol (WS 메시지 계약) — 위치 무관 동일
[kernel]  살아있는 추론 프로세스(모델 메모리 상주)
     ├─ LocalBackend  : in-process, device = mps / cuda / cpu
     └─ VesslBackend  : VESSL **workspace**(exposed port) 안에서 동일 커널 실행 → API가 WS 프록시
```

**핵심 원칙:** 커널 코드는 어디서 돌든 동일한 파이썬 모듈. *어디서 띄우고 어떻게 붙느냐*(런처+전송)만
백엔드별로 다르다. → `KernelProtocol`(WS 계약)과 `Backend` 추상화로 분리.

> VESSL **batch job으로는 인터랙티브 불가** — 살아있는 커널은 반드시 VESSL **workspace**(SSH/포트
> 노출, vesslctl skill 사용)로 띄운다. 사용자의 "vessl을 job으로 찾는다"는 `Backend.discover()`가
> 실행 중인 VESSL workspace/job을 나열하고 attach하는 기능으로 해석.

## 재사용 자산 (탐색 확인 완료)

| 용도 | 재사용 대상 | 메모 |
|---|---|---|
| spot 마스크 적용(극소=×0) | `scripts/save_masked_model.py:apply_mask()` | 파괴적 → studio에선 **forward hook**으로 비파괴·가역화 |
| param명→(layer, module) 매핑 | `scripts/arch_adapter.py:parse_param/is_target` | GUI 레이어/모듈 선택·시각화에 필수 |
| top-k% 마스크 생성 | `scripts/create_approx_spot_masks.py` | spot 정의 입력으로 사용(재계산 X, artifact 로드) |
| 디바이스 선택 | `resolve_device("auto")` (위 파일) | **mps 분기 추가 필요**(현재 cuda/cpu뿐) |
| 모델 로딩/생성 패턴 | `scripts/generation_sanity.py` | `attn_implementation="eager"`로 바꿔야 어텐션 캡처 가능 |
| 카탈로그/세션 등록 | `parametic_platform/catalog.py` 패턴 | studio용으로 이식·확장 |

**새로 만들 것(어려운 순):** ① 어텐션 캡처+스트리밍 ② 토큰단위 스트리밍 디코드 루프
③ WebSocket 전송 ④ 컴퓨트 백엔드 추상화(local/vessl) ⑤ spot 개입 hook GUI ⑥ React 멀티패널+3D.

## 커널 설계 (가장 어렵고 핵심)

`parametic_studio/kernel/` 신설. 모듈: `model_session.py / attention.py / intervention.py / streaming.py`.

- **로딩:** `AutoModelForCausalLM`, **`attn_implementation="eager"`**(output_attentions 필수 조건),
  device/dtype = {cuda: bf16, mps: fp16/fp32, cpu: fp32}.
- **스트리밍 디코드:** `TextIteratorStreamer`가 아니라 **수동 디코드 루프**(스텝마다 어텐션·개입 제어가
  필요하므로). 루프: forward(`output_attentions=True`) → next token 샘플 → `token` 이벤트 +
  `attention` 프레임 이벤트를 WS로 emit → 반복. 정지(stop) 지원.
- **어텐션 캡처 전략(함정 주의):** raw 어텐션은 O(layers×heads×seq²)라 8B면 한 스텝에 수 GB →
  **GPU 위에서 집계 후 전송**.
  - 기본 프레임: 레이어별 head-mean → `[layers, q, kv]`. kv가 길면 다운샘플/top-k.
  - 드릴다운: 선택 레이어 1개만 per-head `[heads, q, kv]` 온디맨드.
  - 전송: typed-array(JSON shape+values 또는 binary).
- **리플레이:** 생성 중 토큰+어텐션을 디스크에 기록(npz/safetensors + json 인덱스, `recordings/`).
  리플레이 = 기록 읽기, **GPU 불필요**. → "스트리밍 또는 리플레이"가 한 메커니즘으로 해결.
- **spot 개입(기능 3, 후속 마일스톤):** 로드한 마스크로 target 모듈에 **forward pre-hook**.
  모드: suppress(×0/×α<1) · amplify(×α>1) · substitute(마스크 위치를 random/mean/zero/타 체크포인트로
  치환). **비파괴·마스크별 토글·가역**. `arch_adapter` + `apply_mask` 로직 재사용.

## 전송(transport)
- **WebSocket**: 라이브 토큰·어텐션 프레임(양방향: 정지·개입 명령).
- **REST**: 카탈로그 / 백엔드 탐색 / 세션 lifecycle / 리플레이 목록·페치.

## 신규 repo 레이아웃
```
parametic_studio/
  api.py              세션 서버(FastAPI) — WS 프록시·세션·리플레이
  catalog.py          모델/디바이스/spec 레지스트리(기존 패턴 이식)
  recordings.py       어텐션·토큰 기록 read/write
  kernel/             model_session·attention·intervention·streaming
  backends/           base.py · local.py(mps/cuda/cpu) · vessl.py(workspace)
studio_web/           React+Vite+TS 멀티패널 IDE (패널: Chat/Attention/Spot/Settings)
```
- 재사용: `scripts/save_masked_model.py`, `scripts/arch_adapter.py`,
  `scripts/create_approx_spot_masks.py`, `resolve_device`(mps 추가).
- 구 `parametic_platform/`은 **studio MVP 도달 전까지 유지**(동작하는 코드 선삭제 금지), 도달 후 제거.

## 마일스톤 (TDD — tests first, 메모 feedback-tdd 준수)

- **M0 커널 스켈레톤 + 로컬 mps:** 작은 모델 로드, 수동 스트리밍 디코드, WS 토큰 스트림,
  디바이스 선택(mps/cuda/cpu). 테스트는 **tiny eager 모델 on CPU**(GPU 없이 CI).
- **M1 어텐션 캡처 + 라이브 시각화 (기능 1+2):** eager attn, 스텝별 집계 어텐션 프레임 WS,
  프론트 2D 히트맵(현재 스텝) + 토큰-어텐션 오버레이, 정지 버튼.  ← **MVP**
- **M2 리플레이:** 디스크 기록 + 리플레이 엔드포인트 + UI 스크러버, 3D 어텐션 큐브(three.js).
- **M3 백엔드 추상화 + VESSL workspace 커널:** discover/attach, WS 프록시("job으로 찾기").
- **M4 spot 제어 GUI (기능 3):** 마스크 로드, 개입 hook(극소/극대/치환), 재생성 diff, PPL/지표 패널.
- **M5 설정/IA 마감, 구 플랫폼 제거.**

## 검증
- **유닛(CPU, GPU 불필요):** tiny eager 모델로 — 스트리밍 루프 토큰 산출 / 어텐션 프레임 shape /
  개입 hook의 ×0·×α·치환 수치 / 기록 roundtrip. WS는 mock.
- **로컬 mps 수동:** 사용자 Mac에서 3B 모델, chat → 라이브 어텐션 확인.
- **cuda:** 여기 올린 뒤 cuda 경로 확인.

## 열린 질문 / 사용자 입력 대기
1. **UI 예시**(R Studio/Matlab풍 멀티패널 레이아웃) — 받은 뒤 `studio_web` 패널 IA 확정.
2. 리플레이 기록 보존 정책(전부 vs 집계만; 디스크 용량).
3. spot 정의 입력 경로: 기존 batch 플랫폼 artifact 재사용 vs studio 내 재계산(현재는 재사용 전제).
