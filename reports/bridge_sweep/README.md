# Bridge 스윕 — k 스윕 + 제외 비율 스윕

**날짜:** 2026-06-22 · **브랜치:** experiment/qwen3-8b-calibration
**모델:** base=Qwen2.5-1.5B, donor=Qwen2.5-Coder-1.5B · **평가:** MultiPL-E HumanEval-Java (completion, 158문제, greedy)
**공통:** full(10000) grad, v2(교집합) 이식, 동일 파이프라인(weight float32). base 원본=0.2595, coder 원본=0.3734(+11.4%p).

[[bridge_repro]]의 +3.2%p 재현에 이어, **"얼마나 옮길까(k)"** 와 **"코어를 얼마나 뺄까(제외 비율)"** 두 축을 스윕했다.

## 1. k 스윕 — bridge 크기 (A=B=top-k 동시 확장)

`A=|weight| top-k`, `B=grad top-k`를 같은 k로 키우며 bridge=`A∖B` 교집합 이식.

| k | selected | pass@1 | Δ vs base |
|---|---:|---:|---:|
| 0.01 | 697,992 | 0.2722 | +1.3%p |
| 0.03 | 1,869,385 | 0.2785 | +1.9%p |
| 0.05 | 3,709,031 | 0.2848 | **+2.5%p** |

→ **bridge 영역을 키울수록 단조 향상.** 더 넓은 weight-중요 영역을 이으면 더 많은 잠재 능력이 활성화된다.

## 2. 제외 비율 스윕 — 코어를 얼마나 빼나 (A=top1% 고정, B만 변화)

A를 `|weight| top-1%`로 **고정**하고, 빼낼 코어 `B=grad top-core_k`의 크기만 키운다. bridge=`A∖B`.

| core_k (제외) | selected | pass@1 | Δ vs base |
|---|---:|---:|---:|
| 0.005 | 1,155,837 | 0.2658 | +0.6%p |
| 0.01 | 697,992 | 0.2722 | +1.3%p |
| 0.02 | 640,974 | 0.2722 | +1.3%p |
| 0.03 | 492,811 | 0.2785 | **+1.9%p** |

→ **코어(B)를 더 많이 뺄수록 bridge가 순수해지고 향상↑.** 적게 빼면(c0.005, 코어 일부가 bridge에 섞임) 오히려 향상이 떨어진다.
이식량(selected)은 제외가 커질수록 *줄어드는데도* 향상은 *커진다* — **개수가 아니라 코어 회피가 핵심**.

## 핵심 결론

1. **두 축 모두 Bridge Theory를 지지**:
   - bridge 영역(A∖B)은 **클수록** 좋다(k 스윕, +2.5%p @ k0.05).
   - 단, 그 안에 **코어(B)가 섞이면 안 된다** — 코어를 많이 뺄수록 좋다(제외 스윕, +1.9%p @ c0.03).
2. **코어는 이식 대상이 아니다** — 제외 비율을 줄여 코어를 bridge에 남기면(c0.005) 향상이 +0.6%p로 저하. [[transplant_ablation]]에서 코어 자체(단순 spot)를 이식하면 전부 붕괴했던 것과 일관.
3. **절대 차이는 작다**(158문제에서 1~4문제, +0.6~2.5%p)지만 **두 스윕 모두 단조**라 노이즈가 아닌 경향. 더 큰 평가셋/다seed로 확증 권장.

## 운영 메모 (하네스 버그 수정)

- **race 버그**: VESSL job 생성 직후 상태가 `created`인데 `active_job_exists`/watch 패턴이
  `running|pending|idle|initializing`만 봐서 갓 만든 job을 "없음"으로 오판 → eval이 bridge 완료 전
  빈 디렉토리를 읽어 `applied=0`. 패턴에 **`created` 추가**로 수정. 이름 중복 가드가 안 걸리던 것도 동일 원인.
- watch는 **이름 패턴 대신 slug 단위 폴링**이 안전(`vesslctl job show <slug>` State).

## 산출물
- k 스윕 bridge: `/shared/.../transplant/bridge/{base,coder}-10000-k{0.01,0.03,0.05}/`
- 제외 비율 bridge: `/shared/.../transplant/bridge/{base,coder}-10000-k0.01-c{0.005,0.02,0.03}/`
- v2 summary는 각 `tx-bre-v2-*` job logs의 `pass_at_1`
- `scripts/paper_spot/extract_spot.py` — `core_k` 인자(A의 k와 코어 B의 k 분리; default=k는 논문 동작)
- `scripts/transplant/submit_jobs.sh` — `CORE_K` env, 경로/이름 라벨 `k{K}-c{CK}`
