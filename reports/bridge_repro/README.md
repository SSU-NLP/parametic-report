# Bridge Theory 재현 — 성공 ✅

**날짜:** 2026-06-22 · **브랜치:** experiment/qwen3-8b-calibration
**모델:** base=Qwen2.5-1.5B, donor=Qwen2.5-Coder-1.5B · **평가:** MultiPL-E HumanEval-Java (completion, 158문제)

## Bridge Theory

모델은 java 코딩 **능력을 이미 잠재**하고 있다. 핵심 spot(`B = |grad·param| top-1%`)은 그 능력의
*코어*(0으로 만들면 붕괴). 하지만 base는 코어를 **활성화/연결하는 bridge**(`A∖B`)가 약해 능력이
발현되지 않는다. coder의 bridge를 base에 이식하면 base의 잠재 능력이 *연결되어* 활성화된다.

## 정식 spot 워크플로우 (논문 코드 = `scripts/paper_spot/`)

```
A = |weight| top-1%                         (save_model_layer_weights)
B = |grad·param| top-1% = 코어              (extract_accumulated_core_linguistic_region, java)
bridge = A ∖ B                              (extract_spot, A & ~B)
교집합 = base_bridge ∩ coder_bridge         (intersect_spot_masks, logical_and)
이식: 교집합 위치에 coder 값 덮어쓰기        (eval_bridge_cowork v2)
```

## 결과 — grad 품질(1024 vs full)이 결정적

| 조건 | sample 1024 (approx) | sample 10000 (full) |
|---|---:|---:|
| base 원본 | 0.2595 | 0.2595 |
| coder 원본 | 0.3734 (+11.4%p) | 0.3734 (+11.4%p) |
| v1 coder-bridge 단독 | −6.3%p | −5.7%p |
| **v2 교집합 (Bridge)** | **+0.0%p (무효)** | **+3.2%p (향상)** |
| v3 base-bridge 단독 | −9.5%p | −6.3%p |

이식량은 거의 같다(1024: 731,537 / full: 705,282). **개수가 아니라 *어느 위치*냐가 전부였고,
그건 grad 품질이 결정**했다. approx(1024) bridge는 "무해한 자리"(base≈coder, greedy 생성 무변화),
full(10000) bridge는 **실제로 능력을 잇는 자리**를 잡았다.

## 핵심 결론

1. **Bridge Theory 성립** (full): 코어(B) 비켜가고 bridge(A∖B) 교집합 이식 → base 잠재 능력 활성화 → **+3.2%p**. 동료가 본 +2% 재현(초과).
2. **교집합(v2)만 향상** — coder bridge 단독(v1)·base bridge 단독(v3)은 손상(−5~6%). **두 모델이 공통으로 bridge로 본 자리**가 핵심.
3. **grad 품질이 필수** — 1024 approx bridge는 위치가 부정확해 무효. full(10,000)이어야 bridge가 능력 자리를 잡는다.
4. 앞선 ablation(`reports/transplant_ablation/`)의 단순 spot(=코어 B 자체)을 이식하면 전부 붕괴했던 것과 대비 — **코어는 못 옮기고, bridge는 옮길 수 있다.**

## 산출물
- summary: `reports/bridge_repro/s{1024,10000}/{base,coder,v1,v2,v3}.json`
- bridge 마스크 (VESSL): `/shared/.../transplant/bridge/{base,coder}-{1024,10000}/`
- full(10000) 점수: `/shared/.../transplant/scores/{base,coder}/seed_*/java/grad-mul-param_checkpoint_10000/`

## 다음 후보
- **k 스윕**: bridge를 얼마나(top-k) 옮길 때 향상이 최대인가
- **확률성 검증**: 여러 seed로 +3.2%가 안정적 향상인지 (동료 "확률적" 언급)
- **제외 비율 스윕**: 코어 B를 얼마나 빼야(A∖B의 B 크기) 향상이 시작/최대인지 — Bridge Theory의 핵심 곡선
