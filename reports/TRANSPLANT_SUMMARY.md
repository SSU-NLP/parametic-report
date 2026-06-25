# Parameter Transplant 연구 — 종합 기록 (2026-06-24)

> Qwen2.5-1.5B(recipient) ← Qwen2.5-Coder-1.5B(donor) 사이에서 **coding 능력을 파라미터 이식으로
> 옮길 수 있는가**를 검증한 트랙의 전체 기록. 평가는 달리 명시 없으면 **MultiPL-E HumanEval-Java
> completion 158문제 pass@1**(base=0.272, coder=0.386). 모든 수치·코드·엑셀 시트 매핑 포함.
> 상세 운영 메모: `~/.claude/.../memory/paper-repro-status.md`, 계획: `~/.claude/plans/handoff-md-fuzzy-snowglobe.md`.

## 연구질문의 진화
1. (초기) Coding Spot 파라미터를 이식하면 coding이 전이되는가? → **No**
2. (수정) ablation-critical region이 transplantable skill module인가? → **No** (importance ≠ transplantability)
3. (현재 답) **무학습(static) 파라미터 이식은 이 모델쌍에서 닫혔다.** 남은 길 = 학습형 함수공간(LoRA distillation).

## 핵심 결론 (한 문장)
> Ablation-critical coding parameters are **not learning-free transplantable**. Across raw weight copy,
> output-space gating, gradient-aligned/Fisher-safe gated surgery, and every saliency definition, **no static
> donor-coordinate operation transfers coding ability above base or random controls.** 원인 = donor·recipient의
> representation/weight-space **co-adaptation(공적응)**: 개별 좌표·층은 호환돼도 깊이방향으로 누적되어 붕괴하거나,
> 1차 loss-descent가 pass@1 스킬로 이어지지 않음.

## 실험별 결과 요약 (엑셀 시트 = `reports/experiments.xlsx`, build_excel.py로 재생성)

| § | 실험 | 핵심 결과 | 시트 |
|---|---|---|---|
| C-2 | base 쌍 bridge/spot 이식 (java) | **spot 직접이식=인과 붕괴**(v1/v2/v3→0~9/158, p<.001, reverse도 coder붕괴); **bridge=null**(v2 vs rand p=.39) | 16 |
| C-2 | 다언어 MultiPL-E (py/java/cpp/js/go, 798쌍) | bridge v2 vs base CMH **p=.901**, v2 vs rand **p=.556**; coder vs base +75 p<.001(검정력 충분) → **bridge "up"=노이즈 확정** | 17,18 |
| delta | global interp(전 파라미터) α-sweep | α≥0.3 붕괴(0.076/0.019); spot delta α↑붕괴·α↓floor; bridge delta 전 α floor | 19 |
| 10 | FFN-only global interp gate | max α pass@1=0.298 < base+0.04, α≥0.3 붕괴 → boundary 탓 의심 | 20 |
| 11 | 호환성 진단(activation map + 단일모듈 NLL) | 중간층(2-25) 단일 MLP swap 무해(Δnll +0.02~0.06); 경계 0/1/26/27 비호환(L1 cos0.17, L27 rel_delta3.9). **NLL은 coder≈base→전이 신호 못 잼** | 21 |
| 12 | boundary-excluded / middle-only FFN | 중간 24층 **동시** swap도 0/158 붕괴 = **깊이방향 누적(compositional)**; 경계 개별도 붕괴 | 22 |
| 13 | residual-gated output injection (β-sweep) | 출력만 혼합: 작은 β neutral, β=0.3 유의손상(p=.025); weight-interp보다 안정→추가손상=MLP 내부 nonlinear path | 23 |
| 14 | compatibility-gated static surgery | T=max(0,−GΔ)/(FΔ²+ε)·ρ + gain/descent-rand/matched-rand. 전부 pass@1≈base(p≥0.48); gain **pred_ΔL=−30.8인데도 base** → 1차 loss-descent ≠ pass@1 | 24 |
| 15 | abs/signed saliency closure | signed≈abs≈signed∩abs≈abs∖signed≈rand≈base, saliency 정의 무관 → static saliency surgery 종결 | 25 |

**모든 이식 경로(raw·output·gated·saliency)가 base/random 노이즈 밴드(±2~3문제, McNemar p>0.4) 안.**

## 메서드 교훈 (재사용 시 주의)
- **NLL ≠ pass@1 전이**: 이 java 토큰셋에서 coder NLL(1.12)≈base(1.10). teacher-forced NLL/1차 loss-descent는 generation pass@1 전이를 못 잡음 → **전이 판정은 반드시 pass@1**.
- **greedy decoding brittleness**: ~1M 좌표 미세 이식에도 경계 1~3문제가 pass↔fail 양방향 flip → net ±2 노이즈. "random>base"는 착시.
- **abs 누적 importance = |θ|·A** (θ 좌표별 상수라 |·| 밖으로): signed=|θ|·|G|, abs=|θ|·Σ|g| → 박제된 G/A에서 계산, **재backprop 불필요**.
- **토크나이저**: Qwen2.5 base↔coder 바이트 동일(검증). 코드는 모델 디렉토리에서 토크나이저 동시 로드.

## 코드 아티팩트 (scripts/transplant/, scripts/)
- `eval_bridge_cowork.py` — 이식 빌드(v1/v2/v3/rand/perm/.../interp) + `keep_tensor(modules,layers)`(torch-free) + `--alpha/--modules/--layers`
- `eval_humaneval_java_cowork.py` — java completion 하네스; `generate_completions(...,model=,tokenizer=)` 사전로드 지원
- `eval_multipl_bridge.py` + `multipl_exec.py`(5언어 실행기) + `prep_multipl_data.py` + `mcnemar_cmh.py` — 다언어 798
- `activation_compat.py` — forward-only 진단(activation map + 단일모듈 NLL)
- `residual_gate.py` — 출력공간 게이팅(`blend`)
- `grad_calib.py` — recipient G/A/F (per-sample backward, 무학습)
- `gated_delta.py` — `gated_mask`(T/gain/descent-rand/matched-rand/signed/abs/signed-and-abs/abs-not-signed)
- 테스트: `tests/test_{keep_tensor,multipl_exec,residual_gate,gated_delta}.py`
- submit 케이스(`submit_jobs.sh`): bridge-eval-full, spot-eval-full, multipl-eval-set, ffn-interp-sweep, boundary-ffn-screen, residual-gate-sweep, activation-compat, grad-calib, gated-delta-primary/controls, saliency-sweep. **잡 종료 시 /work scratch 자동정리 패치 적용.**
- 결과 CSV: `reports/paper_repro_{basepair,basepair_multipl,basepair_multipl_mcnemar,basepair_delta,basepair_module,basepair_boundary,actcompat,resgate,gated,saliency}.csv`
- /shared 잔존: `transplant/{scores,bridge,spot,grad/{G,A,F},data,multipl,multipl-java}` (재calib 불필요)

## 다음 (parked)
- **LoRA distillation** (최우선 실제-전이): donor logits → recipient+LoRA KL; boundary 제외 middle-only LoRA vs all-layer. peft 인프라 확인부터. 함수공간 전이라 weight-space 비호환 무관.
- adapter-mediated transplant(donor FFN 고정 + low-rank P/Q만 학습).
- (무관) 7B/8B Table 1 완성, B3 언어별 Jaccard viz.

## 논문 프레이밍
*Neural Transplant Rejection: Ablation-Critical Coding Parameters Do Not Transfer* / *From Parameter Importance
to Transplant Compatibility*. 주장: (1)중요도는 ablation엔 causal이나 이식엔 실패 (2)원인=representation/weight-space
co-adaptation (3)FFN-only·boundary 제외해도 실패→boundary confound 아님 (4)1차 정렬·Fisher·sign·saliency 어느 것도
무전이 (5)학습형 함수공간만 남음.
