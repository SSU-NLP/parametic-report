# Phase C 실험 정리 — "bridge가 진짜인가" 판정 (파라미터 이식)

목적: 코딩 능력 전이를 일으키는 게 **spot(코어 B)** 인가 **bridge(A∖B)** 인가 둘 다 아닌가를, 통제·통계까지 갖춰
**방어 가능하게 판정**. (이전 java-bridge는 노이즈 플로어에서 불결정으로 끝남.) codex 2라운드 리뷰 반영(CONDITIONAL-GO).

## 고정 조건
- 쌍: recipient **R = Qwen/Qwen2.5-1.5B-Instruct** ← donor **D = Qwen/Qwen2.5-Coder-1.5B-Instruct** (동일 아키텍처 유일 쌍).
- 타깃: **Spot S = B** = code spot(다언어 합산 `I_j=Σ_l|grad·param|` 상위 kB%). **Bridge = A∖B**, A=|θ| 상위 kA%, **kA≫kB**.
- 평가: **MultiPL-E 다언어**(in-distribution, spot이 정의된 비-Python 언어) pass@1 + 언어별 McNemar→CMH(Holm). batch deterministic.
- spot/score: `/shared/seonghyeon/parametic/paper-repro/{masks,scores}/<model>/` 보유(qwen 쌍). 7B/8B와 무관, 지금 가능.

## 0단계 — D0 사전 진단 (CPU only, STOP-if-negative 게이트)
| ID | 내용 | 게이트 |
|---|---|---|
| D0-a | A∖B의 drift δ_j=\|θ_D−θ_R\|를 **matched-null**(같은 텐서/모듈, 같은 \|θ\| bin 무작위 동수) 대비 enrichment + **부호/방향** | A∖B에 집중 drift 없으면 **STOP** |
| D0-b | 표현정렬 프록시 `corr(I_R, I_D)` (+ bridge 위치 정렬) | 낮으면 coordinate 이식 가정 취약(해석 보수적) |
| D0-c | A∖B/B/inter의 층위·모듈(embed/attn/mlp/norm) 분포 | 특정 층 집중이면 generic shift 의심 |
- 양성이어도 GO 증명 아님(음성이면 STOP 전용). 통과 시 본 실험.

## 본 실험 — 두 타깃 패밀리 × 전략
각 타깃 T∈{Spot, Bridge}, T_R/T_D = recipient/donor 자신의 타깃 집합.

| # | 패밀리 | type | 정의 (per tensor) | 목적 |
|---|---|---|---|---|
| 0a/0b | — | base / coder | 원본 R / D | floor / ceiling |
| S1 | Spot | direct(v1) | θ_R[T_D] ← θ_D | spot 직접 이식 |
| S2 | Spot | intersect(v2) | θ_R[T_R∩T_D] ← θ_D | spot 교집합 |
| S3 | Spot | position(v3) | θ_R[T_R] ← θ_D | 위치 이식 |
| R1 | Bridge | direct(v1) | θ_R[T_D] ← θ_D | bridge 직접 |
| R2 | Bridge | intersect(v2) | θ_R[T_R∩T_D] ← θ_D | **bridge 본체(가설)** |
| R3 | Bridge | position(v3) | θ_R[T_R] ← θ_D | bridge 위치 |
| U | — | B+bridge | θ_R[B ∪ inter(A∖B)] ← θ_D | bridge가 코어 위 추가 기여? |
| C* | 각 패밀리 | rand-mm / perm / ndhi / ndlo / vhi / vlo / reverse / drop | 로컬-매칭 통제 | 위치특이성·drift·인과 |

- **rand-mm**: 텐서/모듈/레이어 bin별 count + bin 내 donor \|θ\|(선택 δ) 매칭 무작위(전역 매칭 금지).
- **k**: Spot=kB sweep{0.000025,0.0001,0.0009,0.0025}; Bridge=(kA∈{0.01,0.05}, kB). **확증 = 사전등록 단일 설정**(Spot kB=0.0025, Bridge kA=0.05·kB=0.0025) + R2 vs rand-mm, Holm; 나머지 sweep = exploratory.

## 통계 / 검정력
- pooled McNemar 금지(다언어=같은 문제 번역=의존). 언어별 McNemar → **CMH 층화**(방향 일관 시).
- 효과 ±2~4문제(1.3~2.5pp) → 80% power에 ~1200~3000+ paired items 필요. **MultiPL-E 전 언어 동원**(~18×160≈2900)으로 최대화. 2문제·보정 후엔 underpowered 가능 → **기대효과 사전등록, 미달 시 "검출 불가" 명시**(p-해킹 금지).

## 판정 (사전 등록)
G0(D0 통과) → G1(B-direct로 코어 전이성 방향) → **G2 확증**: R2(bridge intersect) > rand-mm ∧ drift 통제 후 유지 ∧ perm 붕괴 ∧ reverse→coder↓ ∧ 층화 McNemar Holm-p<0.05.
충족=bridge 진짜 / rand-mm와 같으면 깨끗한 음성 / spot 패밀리 붕괴면 코어 비전이 재확인.

## 구현 (재사용)
- Spot 패밀리: `scripts/eval_humanevalpack_java.py:build_transplanted_model`(v1/v2/v3a-d, scores 기반).
- Bridge 패밀리: `scripts/transplant/eval_bridge_cowork.py`(전략 + bridge 마스크). bridge 마스크 = `scripts/paper_spot/extract_spot.py --core_path <code spot> --k kA --core_k kB`.
- D0: 신규 `scripts/paper_repro/d0_diagnostic.py`(CPU). 평가: 신규 다언어 MultiPL-E(bigcode multiple-{lang}). McNemar: `scripts/transplant/mcnemar.py` 언어별+CMH.
- submit: `submit_jobs.sh` d0/transplant/eval 케이스. **동시성 ≤3, /work 모니터.** 결과 → `experiments.xlsx` 신규 시트.
