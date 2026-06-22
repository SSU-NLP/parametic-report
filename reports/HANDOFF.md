# Handoff — 논문 재현 + Bridge 실험 (2026-06-22)

> 다음 세션은 이 파일 + `reports/EXPERIMENTS.md` + 승인 plan(`~/.claude/plans/delightful-juggling-cookie.md`)을 먼저 읽어라.
> (CLAUDE.md의 `handoff.md`는 *플랫폼* 작업용. 이 파일은 *transplant/논문재현 실험*용.)

## 지금 위치
- **Phase 0(엑셀 정리)·A(McNemar) 완료. Phase B(논문 재현) 착수 직전.**
- 마지막 push: `bd0f355` (origin/experiment/qwen3-8b-calibration). 미push 변경 없음.

## 핵심 결론 (지금까지)
- **java transplant(우리 확장) = 노이즈로 종결.** McNemar(`reports/mcnemar_k0.05.csv`): coder만 base 대비 유의(net+18, p=0.004); **v2(bridge)·rand·ndlo·vhi·vlo 전부 유의無(p>0.2) → bridge가 random과 구별 안 됨(위치 특이성 불지지)**; perm/ndhi만 유의 손상(p<0.001, 해로운 영역은 실재).
- **배치 분기점**: left-pad 배치 greedy ≠ batch=1 (±2문제 부동소수점 차이). 순차의 k-스윕 단조성도 배치선 깨짐 = effect가 노이즈급. 이후 **배치(deterministic, HE_BATCH=32) 통일**.
- 박제: `reports/{results_sequential,results_batch,mcnemar_k0.05}.csv`, `experiments.xlsx`(13시트, 용어사전 포함), `EXPERIMENTS.md`.

## 논문(Kim et al. 2024, repo 루트 PDF) — 우리가 했던 것과 다름 ⚠️
- spot = **다언어 importance 합산 후 단순 top-k%** (`I_total=Σ_l |grad·param|`). **차집합(Bridge)·transplant는 논문에 없는 우리/doyun 확장** (doyun=동료 재현코드, 논문 아님).
- 조작 = **damage(zero)** → HumanEval 붕괴, 일반과제는 덜.
- 언어 **10개(Python 제외)**: Bash,C#,C++,Go,Java,JS,Julia,Ruby,Rust,TS.
- **k = 0.0025/0.01/0.09/0.25%** (우리 1%보다 4~400배 작음).
- 모델 = CodeLlama-7B-it, Llama-3.1-8B-it, Llama-3.2-3B-it. 데이터 = nampdn-ai/tiny-codes.
- 평가 = **원본 HumanEval(Chen 2021, Python, 164문제)** + 일반과제(GSM8K/HellaSwag/MMLU/TruthfulQA/WinoGrande). (우리가 쓰던 MultiPL-E java 아님!)

## 실험 목록 (승인 plan)
| ID | 실험 | 출처 | 모델 | 조작 | 평가 | 규모 |
|---|---|---|---|---|---|---|
| 0 | 엑셀 정리 ✅ | 우리 | — | 박제 | — | done |
| A | McNemar ✅ | 우리 | Qwen(java) | flip 검정 | results.jsonl | done |
| B1 | 다언어 grad 누적 | 논문 | 5모델 | 10언어 fine-tune→grad·param | — | 5×10 |
| B2 | code spot(합산) | 논문 | 5모델 | 다언어 합산 top-k% | — | 5×k4종 |
| B3 | code spot(언어별 단독)+겹침 | 논문+우리 | 5모델 | 언어별 top-k%, Jaccard(언어 공통성 검증) | — | 5×10×k4종 |
| B4 | **damage 재현** | 논문 | 5모델 | spot/random/bottom zero | **원본 HumanEval(Py)**+일반과제 | 5×k4종×3 |
| C1 | bridge 이식 | 우리 | Qwen 2 | code spot 기준 bridge+통제 | HumanEval+McNemar | 배치 |
| C2 | 언어별 이식 평가 | 우리 | Qwen 2 | MultiPL-E 다언어 transplant | 언어별 pass@1 | 배치 |

- **5모델** = CodeLlama-7B-it, Llama-3.1-8B-it, Llama-3.2-3B-it, Qwen2.5-1.5B-it, Qwen2.5-Coder-1.5B-it
- **이식(C)** = Qwen 2개만, 기준 spot = B의 다언어 code spot

## 다음 (Phase B 1차 — 검증부터)
`Qwen2.5-1.5B-it/damage-spot/0.0025%/full/java` 파이프라인을 **끝까지 1회** 검증 → 5모델×10언어×k4종 확장.
- 인프라(repo 완비): `data_preprocess/`(create_code_dataset, run_preprocess) → `training/further_training/`(accumulate_grad_mul_param) → `region_selection/`(또는 `scripts/paper_spot/`) `extract_accumulated_core_linguistic_region`(합산 top-k% = `code-region/{model}/top{k}`) → `damage/damage_model.py`(zero) → eval.
- **damage용 spot = `code-region`(합산 top-k bool mask) 그대로** (extract_spot의 차집합 아님!).
- **신규 필요**: 원본 HumanEval(Python) 채점기(`run_one_python`: python3 exec+assert) + 일반과제 harness(lm-eval). VESSL submit_jobs에 damage/eval case.

## 측정 정책
- 전부 **배치(deterministic, 고정 batch_size)**. damage는 붕괴(→0)라 배치로 충분; bridge(작은 효과)는 큰 평가셋+McNemar로 정확도(batch=1로 바꿔도 158문제론 노이즈 못 넘음). 논문 Table1 중간-k 정밀대조 행만 선택적 batch=1.

## 표기 규약 (memory: notation-experiment-items)
`[model] / type / k / sample / lang` — sample=full(10000)/approx(1024), lang=agg(다언어 합산)/단일언어.
예: `Qwen2.5-Coder-1.5B-it/v2/0.05/full/agg`.

## 운영 (VESSL) — 깨먹지 말 것
- `scripts/transplant/submit_jobs.sh` cases: cal-base/coder, bridge/bridge-all/bridge-eval `<strat>`, analyze-gen/status/mcnemar. env: `K`, `CORE_K`, `MODE`(full-10000/approx-1024), `HE_BATCH`(32), `ANALYZE_STRATS`, `MCNEMAR_REF`.
- 경로: `TX_OBJ=/shared/<ns>/transplant`. 결과 `results-bridge-<sample>-<blabel>/<strat>/{generations,results,summary}.json`; bridge masks `bridge/<tag>-<sample>-<blabel>`. blabel=`k{K}` 또는 `k{K}-c{CORE_K}`.
- **race 버그(수정됨)**: VESSL 초기 상태 `created`. `active_job_exists`/watch 패턴에 `created` 포함 안 하면 갓 만든 job "없음" 오판 → eval이 빈 디렉토리 읽음. **watch는 slug 폴링이 안전**(`vesslctl job show <slug>` State; terminal=succeeded/failed/terminated).
- 코드 변경 후 `bash scripts/vessl/push.sh`. **호스트엔 /shared 없음**(object volume = 컨테이너만; 분석은 컨테이너 job으로 results 읽어 출력).
- 호스트 python = `/usr/bin/python3`(openpyxl 설치됨 `--break-system-packages`). 엑셀 빌드 `python3 scripts/transplant/build_excel.py`.
- 백그라운드 job watch는 Bash run_in_background + slug 폴링 패턴 재사용.

## 미완/주의
- status 보강(k0.01/0.03/0.1·제외 root의 compile/runtime) 미실시 — pass@1+McNemar로 종결이라 보조. 필요시 `K=.. analyze-status`(단 제외 root는 case가 blabel 미반영 → 수정 필요).
- damage 코드가 극소 k(0.0025%=0.000025)에서 마스크 생성 정상인지 확인.
- 7B/8B 모델 GPU 메모리·시간 큼(A100). 우선순위 = 작은 모델부터.
- `parametic-report-doyun/`은 gitignore(동료 재현코드, 논문 아님 — 인자 형식만 참조).
