# Parametic Studio — Model View Architecture (design spec)

> 2026-06-24. `STUDIO_UI_SPEC.md`(셸·테마·기록)를 따르고, 여기서 **"모델을 어떻게 보는가"**
> — 뷰 시각화 방법론과 그 플러그 구조 — 를 고정한다. 합의 완료, 구현 계획(writing-plans) 직전.

## 1. Context (왜)

raw 어텐션 행렬은 연구자에게도 직관적이지 않다. Parametic Studio의 정체성은 어텐션이 아니라
**파라미터**다(연구 = Coding Spot: grad×param 상위 k%에 능력이 산다). 그래서 모델 뷰를 **파라미터
중심 + 관찰↔개입 스펙트럼**으로 재설계한다. 핵심 비유: **파라미터 믹싱보드** — 노브(구조)를
돌리고 미터(행동)로 효과를 읽는다. 단, 노브는 **선택**이다(그냥 관찰만 하려는 사용자도 있다).

## 2. 핵심 개념

- **관찰↔개입 스펙트럼.** 관찰 전용(어텐션·활성화·spot 보기)부터 개입(노브로 ×0/×α/치환 → diff)까지.
  모든 게 **모듈**이라 빼거나 더할 수 있다. knob은 그냥 또 하나의 (선택) 모듈.
- **중심 동작(개입 시) = 노브를 돌린다 → baseline｜intervened diff를 본다.** "어떻게 변하는가"가 주인공.

## 3. 네 축 (직교)

```
[모델 탭]   여러 상주 모델 = IDE 파일 탭. 전환·비교.   ← top
   ├ [소스]    prompt(라이브) / dataset(배치·grad)        ← 입력
   ├ [probe]   커널이 뽑는 데이터(구독식)                  ← 데이터
   └ [뷰 모듈] 레지스트리, C 자유분할 셸에 렌더            ← 표현
              + [knob] 개입 모듈(선택)
```

1. **모델 탭 (top, IDE 파일탭 비유).** 상단 탭 = 열려있는 모델 세션. `+`로 카탈로그에서 모델 추가,
   탭 클릭으로 활성 전환. **비교:** pane이 *어느 모델 세션*에서 왔는지 바인딩 → cross-model split
   (모델A 어텐션 ｜ 모델B 어텐션, 같은 prompt). 백엔드: 단일 `SESSION` → **세션 레지스트리**(model_id→ModelSession).
   메모리: 작은 모델(0.5–1.5B) 몇 개는 mps에 동시 상주 가능, 큰 건 VESSL(M3).
2. **소스 (입력).** `PromptSource`(라이브 디코드, 토큰별 probe — 현재 가진 것) / `DatasetSource`
   (HF 데이터셋·붙여넣기·업로드). dataset 두 모드: **forward-only**(집계 activation·attention·PPL),
   **forward+backward 누적**(importance=spot, grad×param). 샘플 보정(K개 근사, 플랫폼 방식) 또는 artifact 로드.
3. **probe (데이터).** 디코드/forward 루프가 **구독된 probe만** 계산·emit. 종류:
   `attention`(head-mean [L,kv] +per-head), `activation`(모듈 출력 norm [L,M]), `logitlens`(레이어별
   residual→lm_head top-k), `importance`(grad×param [L,M], dataset), `tensor`(온디맨드 stats).
4. **뷰 모듈 (표현).** 프론트 `viewRegistry`: 각 뷰 = `{id, label, needs:[probe], source, component}`.
   셸(C 자유분할: 탭 드래그→반갈)은 뷰 인스턴스를 담는 그릇. **빼기/더하기 = 레지스트리 등록/해제.**

## 4. 시각 언어 (opencode 다크, `STUDIO_UI_SPEC §3`)

- **색 코딩으로 의미 구분:** **블루(`#007AFF` ramp) = 활동/라이브**(attention·activation·logit lens),
  **앰버(`#E0A85E` ramp) = importance/구조**(spot). 보는 순간 "활동 vs 중요도" 구분.
- 모노 단일 서체, ASCII 브래킷 라벨, 직각 패널.

## 5. 뷰 카탈로그 (확정)

| 뷰 | probe | 소스 | 렌더 | 상태 |
|---|---|---|---|---|
| **Attention** | attention | prompt | causal 삼각 `[Q,KV]` / `[L,kv]` 지문 / per-head / 토큰 오버레이 | ✅ 일부(오버레이 미) |
| **Output/Chat** | — | prompt | 스트리밍 텍스트 | ✅ |
| **Activations** | activation | prompt·dataset | layer×module 히트맵(블루), 토큰마다 "켜짐" | 신규 |
| **Logit lens** | logitlens | prompt | 행=레이어, top-k 예측 토큰+확신 바("몇 층에서 떠오르나") | 신규 |
| **Spot map** | importance | dataset | layer×module 중요도(앰버) + top-k% 아웃라인 하이라이트 | 신규 |
| **Tensor inspector** | tensor | 무관 | 가중치 히스토그램 + shape·dtype·mean·sparsity | 신규 |
| **Knob + diff** | (개입) | prompt | baseline｜intervened 2-pane diff(출력·어텐션·PPL) | 선택·후속 |

토큰 오버레이(BertViz식: 생성 토큰 클릭→입력 단어 하이라이트)는 Attention 뷰의 직관 모드로 추가.

## 6. 재사용 자산

- spot/importance: `scripts/create_approx_spot_masks.py`(top-k·grad×param 누적), `accumulate_grad_mul_param`
  패턴, 샘플 보정. artifact 있으면 로드, 없으면 즉석 계산.
- param명→(layer,module): `scripts/visualize_heatmaps.py:parse_param` — activation/spot 격자 좌표.
- 개입 마스크: `scripts/save_masked_model.py:apply_mask` 로직 → **forward hook(비파괴·가역)** 으로(knob).
- 카탈로그/디바이스: `parametic_studio/device.py`(완), 모델 카탈로그는 `parametic_platform/catalog.py` 패턴.

## 7. 백엔드 함의

- `api.py`: `SESSION`(단일) → `SESSIONS: dict[model_id, ModelSession]` + 활성/탭 관리. WS 메시지에
  `model`(또는 session) 키 추가. probe 구독 목록을 generate 요청에 포함(`probes:[...]`).
- `ModelSession`: `generate`가 구독 probe만 계산해 이벤트에 동봉. `run_dataset(examples, mode)` 신설
  (forward-only / grad-accumulate). activation/logitlens는 forward hook + residual 디코드.
- `recordings`(M2, 보류)와 직교 — 기록은 probe 출력 그대로 저장하면 됨.

## 8. 상태 / 빌드 순서 / 보류

- **완료:** M0 커널·WS·실모델, M1 attention(삼각형·per-head·지문), Zed 셸, 다크 테마.
- **다음(이 spec):** 뷰 레지스트리 리팩터 → 관찰 뷰 순차 구축: Activations → Logit lens → Tensor →
  Spot map(dataset 소스 동반). 각 = probe(커널, TDD) + WS + 뷰 모듈(프론트).
- **그 다음:** 모델 탭(다중 세션) + 소스 선택기(dataset) → 비교. C 자유분할 셸. knob 개입 모듈.
- **보류:** M2 기록/리플레이, M3 VESSL, 토큰 오버레이는 attention 뷰에 끼워넣기.

## 9. 테스트

- probe 계산은 tiny eager 모델 CPU로 TDD: activation norm shape/값, logitlens top-k shape, importance
  grad×param 수치(작은 합성), tensor stats. WS는 mock/TestClient. 렌더는 preview MCP.

## 10. 열린 질문

1. dataset 입력 UX(HF id / 붙여넣기 / 업로드)와 샘플 K 기본값.
2. cross-model 비교를 pane-바인딩으로 갈지, 전용 "compare" 모드로 갈지.
3. spot 즉석 계산 vs artifact 재사용 우선순위(현재: 둘 다, 데모는 즉석 샘플).
