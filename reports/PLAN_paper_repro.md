# 계획: 논문(Kim et al. 2024) 본 실험 재현 + Bridge 이식 확장 + McNemar

## Context (왜)

지금까지 우리는 **java만**으로 spot을 잡아 **transplant(이식)** 실험을 했고, spot도 동료(doyun)의
**차집합 `A∖B`(Bridge)**를 썼다. 그런데 논문 *Exploring Coding Spot* (Kim et al. 2024)을 정독한 결과
실제 논문은 다음과 같다:

- **spot = 다언어 importance 합산 후 단순 top-k%** — `I_j^total = Σ_l |∂L_l/∂θ_j · θ_j|` (식2,3) 정렬 top-k%.
  **차집합(Bridge)·weight 기준은 논문에 없음** (동료/우리 확장).
- **조작 = damage(zero)** → HumanEval pass@1 붕괴, 일반과제는 덜 영향. **transplant는 논문에 없음**.
- **언어 10개**(Bash, C#, C++, Go, Java, JavaScript, Julia, Ruby, Rust, TypeScript) — **Python 제외**(일반화 검증용).
- **k = 0.0025% / 0.01% / 0.09% / 0.25%** (우리 1%보다 4~400배 작음).
- **모델** = CodeLlama-7B-it, Llama-3.1-8B-it, Llama-3.2-3B-it. **데이터** = nampdn-ai/tiny-codes.
- **지표** = HumanEval pass@1 + 일반과제(GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande).

따라서 **가장 먼저 할 일은 논문 본 실험(damage)을 제대로 재현**해 "진짜 code spot"을 5개 모델에서 확보하는 것.
그 위에서 (a) 논문 Table 1 재현, (b) Qwen에서 그 code spot을 토대로 우리 Bridge 이식.

## 실험 목록 (한눈에)

| ID | 실험 | 출처 | 모델 | 무엇을 / 조작 | 평가 | 규모 |
|---|---|---|---|---|---|---|
| **0** | 엑셀 정리 | 우리 | — | 순차/배치 결과 박제 | — | 즉시 |
| **A** | McNemar | 우리(확장) | Qwen(기존 java) | 기존 transplant per-problem flip 검정 | results.jsonl(보유) | 즉시·GPU無 |
| **B1** | 다언어 grad 누적 | 논문 | 5모델 | 10언어 fine-tune → `\|grad·param\|` | — | 5×10 calib |
| **B2** | code spot (합산) | 논문 | 5모델 | 다언어 합산 → top-k% | — | 5×(k 4종) |
| **B3** | code spot (언어별 단독) | 논문+우리 | 5모델 | 언어별 top-k% + 겹침(Jaccard) 분석 | — | 5×10×(k 4종) |
| **B4** | **damage 재현** (논문 핵심) | 논문 | 5모델 | spot/random/bottom **zero** | **원본 HumanEval(Python)** + 일반과제 | 5×(k 4종)×3 |
| **C1** | bridge 이식 | 우리(확장) | Qwen 2 | code spot 기준 bridge + 통제(rand/perm/ndhi/…) | HumanEval + McNemar | 배치 |
| **C2** | 언어별 이식 평가 | 우리(확장) | Qwen 2 | MultiPL-E 다언어 transplant | 언어별 pass@1 | 배치 |

- **5모델** = CodeLlama-7B-it, Llama-3.1-8B-it, Llama-3.2-3B-it, Qwen2.5-1.5B-it, Qwen2.5-Coder-1.5B-it
- **이식(C)** = Qwen 2개만 (1.5B-it ← Coder-1.5B-it), 기준 spot = B에서 구한 다언어 code spot
- 일반과제 = GSM8K / HellaSwag / MMLU / TruthfulQA / WinoGrande (논문 동일)

## 대상 모델 (spot을 full로 봐야 하는 모델 = 5개)
- 논문 재현: `CodeLlama-7B-Instruct`, `Llama-3.1-8B-Instruct`, `Llama-3.2-3B-Instruct`
- 우리 모델: `Qwen2.5-1.5B-Instruct`, `Qwen2.5-Coder-1.5B-Instruct`
- **이식 실험은 Qwen 2개만** (1.5B-it ← Coder-1.5B-it)

## 논문 세팅 (전부 동일하게)
- 데이터 `nampdn-ai/tiny-codes`, 언어 10개(Python 제외)
- spot = 언어별 fine-tune으로 gradient 추출 → `|grad·param|` importance → **언어 합산** → **top-k% (0.0025/0.01/0.09/0.25%)**
- damage = 그 top-k% 위치를 **0으로** → HumanEval + 일반과제
- control = random / bottom-k% (붕괴가 spot 특이적인지)

---

## Phase 0 — 엑셀 정리 (실험 시작 전 먼저, 이미 데이터 있음)
파이프라인 들어가기 전에 지금까지 결과부터 깔끔히 박제.
- `reports/experiments.xlsx`에 **배치 시트**(7_batch_ksweep/exclusion, 8_batch_control) + **순차vs배치 비교 시트** 추가 (`build_excel.py`에 `results_batch.csv` 읽는 시트 함수 추가).
- `reports/results_batch.csv` + 갱신 xlsx + 분석 스크립트 커밋.

## Phase A — java transplant 배치 마무리 + McNemar (즉시, GPU 거의 無)
우리가 한 java 실험(k 스윕 **1/3/5/10%**, 제외 비율, 통제 rand/perm/ndhi/…)은 **이미 배치 재측정 보유**
(`results_batch.csv`: `v2_k0.01/0.03/0.05/0.1`, 제외, 통제). 새 대규모 재측정 아님 — **마무리만** 한다.
- **status 보강**: 배치 status를 k0.05 root만 받았으므로, 나머지 k(0.01/0.03/0.1)·제외 root에 `analyze-status` 배치 1회씩(가벼움).
- **McNemar**: `scripts/transplant/mcnemar.py` — base vs strategy paired. b=base✓→strat✗, c=base✗→strat✓, χ²=(|b−c|−1)²/(b+c), p-value + flip 문제 목록. 데이터=배치 `results.jsonl`(object volume, 보유). 컨테이너 job(`analyze-mcnemar` case) 또는 download.
- **종결**: java/현 bridge는 노이즈급(이미 시사)임을 McNemar로 확정 → 이후는 **논문 code spot(B) 기반으로 전환**.
- 산출: McNemar p + flip 표, 배치 status 보강 → 엑셀/CSV.
- ⚠️ 이 java 실험의 k(1~10%)는 **논문 k(0.0025~0.25%)와 무관** — 별개 실험(B와 혼동 금지).

## Phase B — 논문 본 실험 재현 (최우선): 다언어 code spot + damage
정식 파이프라인은 repo에 완비(`data_preprocess/`, `training/further_training/`, `region_selection/` 또는
`scripts/paper_spot/`, `damage/`, `scripts/evaluate_masked_ppl.py`). **damage용 spot은 차집합(extract_spot)이
아니라 `extract_accumulated_core_linguistic_region`의 출력(`code-region/{model}/top{k}` = 합산 top-k% bool
mask)** 을 그대로 쓴다.

- **B1 데이터**: `data_preprocess/create_code_dataset.py`(tiny-codes) 10언어 + 모델별 tokenizer로 `run_preprocess.sh`.
- **B2 grad·param 누적**: 5모델 × 10언어 (`training/further_training`의 accumulate). 각 (모델,언어) job.
  - Qwen은 java grad 일부 보유하나 **모델이 -it로 바뀌었고 경로/언어가 다르므로 통일 재실행** 검토.
- **B3 spot — 언어별 단독 + 합산 둘 다 (5모델 전부)**:
  - **합산(논문)**: `--language_list '[10개]'` → 다언어 code spot `code-region/{model}/agg/top{k}`.
  - **언어별 단독**: 각 언어 `--language_list '["java"]'` … 10개 → `code-region/{model}/{lang}/top{k}`.
    - **겹침(Jaccard) = 논문 핵심 주장 직접 검증**: 한 모델 안에서 언어별 spot이 같은 위치를 가리키는지. 랜덤 기대(≈k) 대비 **높으면 "언어 공통 coding region 실재"**(논문 지지), ≈k면 "합산 top-k는 허상". 모델별 공통성 차이(CodeLlama vs Qwen 등)도 비교. (cf. ablation의 *모델 간* overlap과 달리 이건 *언어 간* 겹침.)
  - k = {0.000025, 0.0001, 0.0009, 0.0025} (= 0.0025/0.01/0.09/0.25%). 동료 `run_qwen_region_selection.sh`는 인자 형식만 참조(논문 아님).
- **B4 damage + control**: `damage/damage_model.py`로 `code-region` top-k% zero → damaged model. random/bottom mask는 `scripts/create_approx_spot_masks.py`.
- **B5 eval (논문과 동일하게)**: **원본 HumanEval (Chen et al. 2021, Python, 164문제 pass@1)** — 우리가 쓰던 **MultiPL-E java가 아님**. BigCode harness `humaneval` task 또는 python 채점기(`run_one_python`: `python3` exec+assert) 신규. + **일반과제**(GSM8K/HellaSwag/MMLU/TruthfulQA/WinoGrande, lm-eval-harness 셋업). **주의**: calibration은 Python 제외이지만 eval은 Python HumanEval = 논문의 generalization 검증 의도.
- **B6 재현 기준**: spot damage → HumanEval 급락(→0 근처), 일반과제는 덜(논문 Table 1 패턴). control은 덜 붕괴.
- **우선순위**: 작은 모델(Llama-3.2-3B-it 또는 Qwen-1.5B-it) + **java 단일**로 파이프라인 끝까지 1회 검증 → 10언어·5모델·(k 4종) 확장.

## Phase C — Bridge 이식 (Qwen 2개, code spot 기준)
java가 아니라 **B에서 구한 다언어 code spot**을 기준으로 우리 Bridge 이식을 재실험.
- 대상: `Qwen2.5-1.5B-Instruct` ← `Qwen2.5-Coder-1.5B-Instruct`.
- spot 기준 = 논문식 code spot(10언어 합산 top-k%). bridge = `A(|weight| top-k) ∖ B(code spot)`(우리 정의) 등.
- 기존 `scripts/transplant/eval_bridge_cowork.py`의 strategy(v2/rand/perm/ndhi/...) 재사용, bridge 마스크만 code spot 기반으로 교체.
- 평가: 논문과 통일하려면 **원본 HumanEval(Python)** 권장(또는 MultiPL-E 다언어로 언어별). 배치(deterministic) + McNemar.

## 측정 정책 (batch vs 정확도)
- 전부 **배치(deterministic, 고정 batch_size)** 통일. greedy는 batch=1도 deterministic이나 left-pad 배치와 ±2문제(부동소수점) 차이 — 어느 쪽도 절대 정답 아님.
- **damage(B)**: 붕괴(→0)라 배치로 명확 재현, batch=1 불필요. 5모델×10언어×(k 4종)가 현실적 시간에 끝남.
- **bridge(C, 작은 효과)**: batch=1로도 158문제는 노이즈에 묻힘 → **큰 평가셋 + McNemar**로 정확도 확보(batch 방식이 아니라 표본·통계).
- 논문 Table1 중간-k 비-0 값 정밀 대조가 필요한 행만 선택적 batch=1.

## 규모/리스크
- B는 5모델 × 10언어 grad 누적(최대 50 calibration) + damage + 2종 eval(HumanEval/general). VESSL job 수백 → **단계적**(검증→확장).
- 7B/8B 모델은 GPU 메모리·시간 큼(A100). k가 매우 작아(0.0025%) damage 위치 수 적음 — 마스크/damage 코드가 극소 k 처리하는지 확인.
- 일반과제 harness(lm-eval) 컨테이너 셋업이 신규.

## 검증 (어떻게 확인)
- A: McNemar p-value + flip 표 (base/coder가 sanity: coder가 더 많이 flip↑).
- B: damaged-spot HumanEval ≪ control HumanEval, 일반과제 보존 (논문 Table 1 재현). 5모델에서 일관.
- C: bridge vs 통제(배치, McNemar) — code spot 기반에서도 위치 특이성 보이는지.

## 산출물
- `scripts/transplant/mcnemar.py`, 다언어 calibration/damage/eval 드라이버(submit_jobs 확장 case), 일반과제 harness 셋업.
- `reports/`: McNemar 표, 논문재현 damage 표(5모델×(k 4종)×{spot,control}, 언어별 단독 spot 겹침 분석), bridge(code spot) 결과. `experiments.xlsx` 시트 확장.
