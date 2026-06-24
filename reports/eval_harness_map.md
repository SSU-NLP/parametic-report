# Eval 하네스 매핑 (벤치마크 → 하네스 → task/플래그)

**목적**: 벤치마크마다 올바른 하네스·task·플래그를 고정해, **형식/task 선택 착오로 인한 0점·과소평가 재발 방지**.
(실제 사고: instruct 모델에 completion용 task를 써서 HumanEval 전 조건 0점 → `humaneval_instruct`로 해결. 아래 "핵심 규칙" 참고.)

---

## A. Phase B — 논문 재현 (Kim et al. "Coding Spot")  →  EleutherAI **lm-evaluation-harness**

| 벤치마크 | lm-eval task | 필수 플래그 | 드라이버 | 결과 키 |
|---|---|---|---|---|
| **Python HumanEval** (원본 `openai_humaneval`, 164, pass@1) | **`humaneval_instruct`** | `--apply_chat_template --confirm_run_unsafe_code` | `scripts/eval_humaneval_python.py` | `results.humaneval_instruct["pass@1,*"]` |
| **일반 5종** | `gsm8k`, `hellaswag`, `mmlu`, `truthfulqa_mc2`, `winogrande` | `--num_fewshot` 5 / 10 / 5 / 0 / 5 | `scripts/eval_general_lmeval.py` | task별 primary: `exact_match`/`acc_norm`/`acc`/`acc`/`acc` |

- **HumanEval은 반드시 `humaneval_instruct`** — 같은 openai 데이터 + pass@1이지만 instruction 프롬프트 + gen_prefix + ` ```python ` **코드블록 추출 필터**(`build_predictions_instruct`) 포함. instruct/chat 모델용.
- **`humaneval`(plain)은 금지** — completion(함수 본문 이어쓰기)용이라, chat 모델의 마크다운 코드블록 응답을 추출 못해 **정상 모델도 0점**(original·control 전부 0). `--apply_chat_template`만 씌워도 안 됨(task 자체를 바꿔야 함).
- 일반 5종은 chat template 불필요 — loglikelihood 4종 + GSM8K(5-shot)은 few-shot raw로 정상. collapsed(damage) 모델만 `--gen-max-toks 64`로 생성 길이 제한(GSM8K가 EOS 안 내고 max까지 가는 것 방지).

## B. Transplant / Bridge — 우리 확장 (논문 외)  →  **bigcode-evaluation-harness** + 커스텀

| 벤치마크 | task / 방식 | 드라이버 |
|---|---|---|
| **HumanEvalPack Java** (synthesize) | bigcode `humanevalsynthesize-java` | `scripts/eval_humanevalpack_java.py` |
| **MultiPL-E HumanEval-Java** (completion, 158문제) | 커스텀(MultiPL-E java 데이터 로드 → java 컴파일/실행) | `scripts/eval_humaneval_java_cowork.py` |

---

## 핵심 규칙 (실패 방지 체크리스트)

1. **하네스는 벤치마크로 고정**: Python HumanEval + 일반 5종 = **lm-eval** / 다국어(Java 등) 코드 = **bigcode**. (둘 다 Python humaneval 가능하지만, Phase B는 instruct 대응 + 일반과제 통합 때문에 lm-eval로 통일.)
2. **instruct 모델 + 코드 생성 평가 → 반드시 instruct/chat 대응 task** (lm-eval `humaneval_instruct` = 코드블록 추출 필터 포함). raw completion task에 `--apply_chat_template`만 씌우면 0점.
3. **우리가 만든 건 채점기가 아니라 wrapper** — `eval_*.py`는 표준 하네스를 호출만 함(pass@1 직접 계산 X). 실패는 거의 항상 "task/형식 선택" 문제지 자작 채점 버그가 아님.
4. **gated 모델은 제출 전 HF 접근 승인 확인** (모델별 개별 grant; Llama-3.2-3B 된다고 3.1-8B 되는 것 아님).
