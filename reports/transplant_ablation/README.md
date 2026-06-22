# Transplant Ablation — 단순(approx) java spot

**날짜:** 2026-06-22 · **브랜치:** experiment/qwen3-8b-calibration

## 무엇을 한 ablation인가

coder 모델의 java "spot" 파라미터를 base 모델에 이식해 코딩 능력이 전이되는지 본 실험.
단, **spot을 논문 방식(차집합 정제)이 아니라 단순 근사(approx)로 잡았다:**

- spot = `|grad·param|` java calibration 상위 1% (`scripts/create_approx_spot_masks.py`)
- **논문 차집합(`extract_spot.py`: code grad AND NOT 공통/언어 영역) 미적용**
- 데이터 규모도 근사: sample 1024 (논문 full 10,000 아님)

이식 연산 자체는 논문과 동일(교집합 위치에 donor 값 덮어쓰기, 단위테스트 23개 검증).
즉 **이 ablation은 "단순 top-k spot으로는 이식이 되는가"** 를 측정한 것.

## 결과 1 — 동료 평가 방식 재현 (생모델 + MultiPL-E completion, 158문제)

가장 중요한 비교. 동료 셋업(Qwen2.5-1.5B base + completion eval) 그대로, spot만 우리 approx.
`reports/transplant_ablation/cowork/*.json`

| 조건 | pass@1 | passed | compile_err | vs base |
|---|---:|---:|---:|---:|
| base 원본 | 0.2595 | 41 | 32 | +0.0%p |
| coder 원본 | 0.3734 | 59 | 26 | **+11.4%p** |
| v1 같은위치 | 0.0 | 0 | 95 | −25.9%p |
| **v2 교집합** | 0.0127 | 2 | 141 | −24.7%p |
| v3a~d, nonspot | 0.0 | 0 | 158 | −25.9%p |

→ base/coder는 정상(coder가 상한). **이식은 전부 붕괴.** 동료의 +2%는 재현 안 됨.

## 결과 2 — instruct 모델 + BigCode synthesize (approx-1024 spot, java held-out PPL + pass@1)

`reports/transplant_ablation/instruct/*.json`

| 조건 | PPL | pass@1 |
|---|---:|---:|
| base-it | 3.86 | 0.116 |
| coder-it | 4.11 | 0.457 |
| v1 | 15.55 | 0.0 |
| v2 | 11.34 | 0.0 |
| v3b/c/d/ctrl | 5.2M~13.8M | 0.0 |

→ 평가 방식을 바꿔도(instruct/synthesize) 이식은 전부 붕괴. 평가가 원인이 아님.

## 결과 3 — spot 위치 overlap (랜덤 기대 = 1%)

| 비교 | overlap |
|---|---:|
| base ↔ base-it | 56.5% |
| coder ↔ coder-it | 36.6% |
| base ↔ coder | 18.9% |
| base-it ↔ coder-it | 18.3% |

→ 두 모델은 java spot의 ~18%만 공유(교집합 = 전체의 ~0.18%). 같은 패밀리 생↔instruct는 절반 보존.

## 결과 4 — 붕괴 메커니즘 (weight 변화 분석, it 점수)

| 전략 | mean\|Δ\| | max\|Δ\| | newscale/old | max blowup |
|---|---:|---:|---:|---:|
| v1 같은위치 | 0.021 | 8.5 | 1.15 | 2.0x |
| v2 교집합 | 0.019 | 7.8 | 0.79 | 1.6x |
| v3a 재배치 | 0.055 | 37.4 | 0.76 | 2.0x |

→ 스케일 폭발(blowup) 아님. 붕괴는 critical 위치 값의 **구조적 어긋남**(v3 재배치가 최대).

## 결과 5 — 실패 유형 (instruct, BigCode 재채점, 164문제)

| 조건 | passed | wrong | runtime | compile_err |
|---|---:|---:|---:|---:|
| base-it | 19 | 10 | 1 | 134 |
| coder-it | 75 | 24 | 7 | 57 |
| v2 | 0 | 32 | 6 | 126 |
| v3* | 0 | 0 | 0 | 164 |

→ v3는 컴파일조차 안 되는 완전 붕괴. v2는 컴파일 일부 생존하나 정답 0.

## 결론

**단순 java top-k spot으로는 이식이 능력을 전이하지 못하고, 모든 방식·평가에서 붕괴한다.**
동료가 본 +2%(교집합 이식)는 재현되지 않았다.

원인(코드/run 버그 아님 — Codex 검증 + applied=336 확인):
- 우리 spot = 단순 java grad 상위 1% → **일반/공통으로도 critical한 위치를 포함** → 이식 시 모델 전반 붕괴.
- 논문 spot = `extract_spot.py`의 **차집합**으로 그런 위치를 빼고 "코드 특화"만 남김 → 안전 + 향상.

## 다음 (논문 spot 재현)

`region_selection/extract_spot.py`(우리 레포에 원본 그대로 존재, 미수정)로 차집합 정제 spot을 만들어야 함:
1. 여러 언어 / 공통 영역 calibration
2. `extract_accumulated_core_linguistic_region.py` → 공통 영역
3. `extract_spot.py` → 차집합 spot (base/coder)
4. 교집합 이식(`transplant_mapping.py` 또는 `intersect_and_transplant.py`)
5. cowork eval (생모델 + MultiPL-E completion)

동료 `parametic-report-doyun/.../region_selection/run_qwen_region_selection.sh` 에 Qwen용 정확한 인자.

## 근사는 의도된 트레이드오프였다 (잘못이 아님)

플랫폼(`parametic_platform`)이 spot을 approx로 잡은 데는 정당한 이유가 있다:
- 논문 정식 = full calibration(10,000 예제) × 여러 언어 × 차집합 → GPU 수십 시간
- 플랫폼은 **원클릭 데모**(EMNLP surface)라 빠른 응답이 목적 → 의도적으로 sample 1024 + 단순
  top-k로 근사해 ~몇 분에 끝나게 함 (CLAUDE.md: "fast, approximate version … one-click analysis")
- 데모 목적(spot 대략 위치 시각화)엔 합리적

**어긋난 지점:** 이 transplant 실험(논문의 인과적 이식 재현·검증)을 **플랫폼의 approx 파이프라인 위에
그대로 얹은 것**. 데모용 근사 spot으로 논문 결과를 검증하려 한 게 문제. 근사가 틀린 게 아니라,
**이 실험엔 정식(논문) 차집합 spot을 썼어야 한다.** 플랫폼 데모는 approx 그대로 두어도 무방.

## 산출물 위치 (VESSL /shared)
- cowork: `/shared/seonghyeon/parametic/transplant/results-cowork/<cond>/`
- instruct: `/shared/seonghyeon/parametic/transplant/results/<cond>/`
- 생모델/instruct 점수: `/shared/.../transplant/scores/{base,coder,base-it,coder-it}/`
