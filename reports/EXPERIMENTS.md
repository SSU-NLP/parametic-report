# 파라미터 이식(Transplant) 실험 — 전체 기록

**브랜치:** experiment/qwen3-8b-calibration · **갱신:** 2026-06-22
**base:** Qwen2.5-1.5B (0.2595) · **donor/coder:** Qwen2.5-Coder-1.5B (0.3734, +11.4%p = 상한)
**평가:** MultiPL-E HumanEval-Java (completion, 158문제, greedy pass@1) · GPU: VESSL A100

핵심 질문: **coder의 특정 파라미터를 base에 이식하면 base의 java 능력이 활성화되는가? 그게 "특정 위치(bridge)" 때문인가?**

---

## 0. 용어 (논문 코드 = `scripts/paper_spot/`, per-tensor top-k)
- **A** = `|weight|` 상위 top-k (가중치 크기)
- **B(코어)** = `|grad·param|` 상위 top-k = 논문의 damage 대상(0으로 만들면 코딩 붕괴)
- **bridge** = `A ∖ B` (weight 크지만 코어는 아닌 자리)
- **이식 방식**: v1=coder bridge 위치 / v2=base_bridge∩coder_bridge 교집합 / v3=base bridge 위치 → 해당 위치에 coder 값 덮어쓰기

---

## 1. 단순 spot 이식 (실패) — `reports/transplant_ablation/`
처음엔 **단순 spot**(`|grad·param|` top-1% = 코어 B 자체)을 same-location / intersection / 정렬(v3a~d) 등으로 이식.
→ **전부 붕괴 (pass@1 ≈ 0)**. **코어는 옮길 수 없다.**

## 2. 동료 spot 정의 발견 → Bridge Theory
동료의 실제 spot은 단순 top-k가 아니라 **차집합 `A∖B`**였음. 이를 재해석:
> 모델은 java 능력을 이미 **잠재**(코어 B). base는 그걸 **활성화/연결하는 bridge(A∖B)**가 약하다.
> coder의 bridge를 이식하면 base의 잠재 능력이 *연결되어* 활성화된다.

## 3. Bridge 재현 (성공) — `reports/bridge_repro/`
정식 워크플로우(`paper_spot`)로 v1/v2/v3 비교:

| 조건 | 1024(approx) grad | 10000(full) grad |
|---|---:|---:|
| base 원본 | 0.2595 | 0.2595 |
| coder 원본 (상한) | 0.3734 (+11.4%p) | 0.3734 |
| **v1 coder bridge 그대로(같은 위치)** | −6.3%p | **−5.7%p** |
| **v2 교집합 (Bridge)** | +0.0%p (무효) | **+3.2%p** |
| v3 base bridge 단독 | −9.5%p | −6.3%p |

**결론**: ① **교집합(v2)만 향상**, coder를 *그대로 같은 위치로* 옮기면(v1) **오히려 떨어진다**. ② grad 품질(1024→10000)이 위치 정확도를 결정. 동료 +2% 재현(초과).

## 4. k 스윕 — 포화 곡선 "어디까지" (`reports/bridge_sweep/`)
A=B=top-k를 키우며 v2 교집합 이식:

| k | pass@1 | Δ | selected |
|---|---:|---:|---:|
| 0.01 | 0.2722 | +1.3%p | 698k |
| 0.03 | 0.2785 | +1.9%p | 1.87M |
| 0.05 | 0.2848 | +2.5%p | 3.71M |
| 0.10 | 0.2975 | +3.8%p | 6.96M |

→ **k=0.1까지 단조 증가, 포화 미관측.** 단, 이것만으론 "위치 효과"와 "양/interpolation"을 못 가림(아래 통제 필요).

## 5. 제외 비율 스윕 — "코어를 얼마나 빼나" (`reports/bridge_sweep/`)
A=top1% 고정, 코어 B만 키워 더 많이 제외(bridge=A∖B):

| core_k (제외) | pass@1 | Δ | selected |
|---|---:|---:|---:|
| 0.005 | 0.2658 | +0.6%p | 1.16M |
| 0.01 | 0.2722 | +1.3%p | 698k |
| 0.02 | 0.2722 | +1.3%p | 641k |
| 0.03 | 0.2785 | +1.9%p | 493k |

→ **코어를 많이 뺄수록 bridge가 순수해지고 향상↑.** 이식 개수는 *줄어드는데* 향상은 *커짐* → 개수가 아니라 **코어 회피**가 핵심. (단순 spot=코어 이식이 붕괴한 §1과 일관.)

## 6. 통제 실험 — 위치/값/인과 (순차 완료 → 배치 재측정 중)
Codex 자문으로 보강. **가장 위험한 confound**: "coder는 base의 fine-tune endpoint라 어디든 섞으면 coder로 이동(drift)". bridge가 `|coder−base|` drift 큰 자리라서 효과일 수 있음. k=0.05 기준 동시 제출:

| strategy | 의미 | bridge 지지 = |
|---|---|---|
| `rand` | 같은 N개 무작위 위치 | v2 > rand (양 아님) |
| `perm` | bridge 위치, coder 값 셔플 | v2 > perm (값-위치 정합) |
| `ndhi`/`ndlo` | non-bridge 중 drift 상/하위 N | v2 > ndhi (drift confound 차단) |
| `vhi`/`vlo` | bridge 내부 drift 상/하위 절반 | 비슷=drift 무관 |
| `reverse` | coder에 base bridge 이식 | coder 하락 (인과) |

**결과 (순차, k=0.05, base=0.2595):**

| strategy | pass@1 | 해석 |
|---|---:|---|
| v2 (bridge) | 0.2848 | 기준 |
| rand / ndlo / vhi / vlo | 0.2848 | **= v2 (위치 무관)** |
| perm | 0.0 | 값-위치 정합 셔플 → 컴파일 붕괴(158) |
| ndhi | 0.038 | non-bridge drift 큰 곳 → 붕괴 |
| reverse | 0.3481 | coder −2.5pp (약한 인과) |

- **generations 비교**: rand vs v2 = **132/158 다른 출력** → "같은 출력이라 동일"이 아니라 **다른 코드인데 통과 *개수*만 우연 일치(45/158)**.
- **status 세분화**: 정상 통제 5개가 거의 동일 (pass~45, compile_err~30, wrong_answer~80). 병목은 wrong answer = base 실력 한계. perm/ndhi만 compile_err 폭증.

**결론(잠정): bridge 위치 특이성 불지지** — v2가 random과 pass@1·status 모두 구별 안 됨. 단 perm/ndhi(해로운 영역)는 명확히 붕괴 = 파라미터 민감도는 위치 의존(그러나 "능력 활성화"는 아님). effect(2~4문제)가 디코딩 노이즈(±2문제)급이라 **158문제 pass@1으론 주장 불가**.

## 7. ⭐ 배치 분기점 (측정 패러다임 전환, 2026-06-22)

순차(batch=1) 생성은 158문제에 **26분** → 대량 평가 불가. **left-pad 배치 생성으로 ~20배 단축(1.3분)**. 단 left-pad가 greedy를 미세하게 바꿔 **순차와 ±2문제 차이**(v2 k0.05: seq 0.2848 ↔ batch 0.2975).
effect가 이 노이즈급이라 **여기서 측정 방식을 바꾼다**:
- 이후 모든 측정은 **배치(deterministic, 고정 batch_size=32)로 통일** — 조건 간 비교 valid, 절대값만 batch=1과 다름.
- **분기점 이전 순차 데이터는 `reports/results_sequential.csv`에 박제** (덮어쓰지 않고 보존).
- 배치로 base/coder + 전체 스윕/통제 재측정 → 깨끗한 비교 + McNemar 토대 + 큰 평가셋.

## 8. 운영/하네스
- **race 버그 수정**: VESSL 초기 상태 `created`를 watch/가드가 active로 안 봐 eval이 bridge 완료 전 빈 디렉토리를 읽던 문제(`applied=0`). `created` 추가 + watch는 slug 폴링.
- 코드: `scripts/transplant/{submit_jobs.sh,eval_bridge_cowork.py}`, `scripts/paper_spot/extract_spot.py(core_k)`

---

## 현재 결론 (갱신 — 통제 후)
1. **해로운 영역은 실재한다**: coder bridge를 그대로(v1) 옮기거나, 값-위치 정합을 깨거나(perm), drift 큰 자리(ndhi)를 건드리면 **붕괴**. 파라미터 민감도는 위치에 의존.
2. **그러나 "유익한 bridge 특이성"은 아직 불지지**: 정상 통제(rand/ndlo/vhi/vlo)가 v2와 pass@1·status·붕괴 안 함까지 동일 → **bridge가 random보다 능력을 더 살린다는 증거 없음**.
3. **effect가 노이즈급**: +2~4문제(158 중)가 디코딩 방식(±2문제)과 동급. → 158문제 pass@1으로는 결론 불가. **배치 통일 + 큰 평가셋 + McNemar 필수**.

## 다음 (배치 분기점 이후)
- 배치로 base/coder + 전체 스윕/통제 재측정(진행 중) → 깨끗한 비교
- **McNemar(per-problem)**: base 대비 어떤 문제가 flip되는지, 위치별로 다른지
- **더 큰 java 셋**(MBPP-Java 등) — effect를 노이즈 위로
- (살아남으면) 언어/donor 특이성, full-replacement interpolation curve
