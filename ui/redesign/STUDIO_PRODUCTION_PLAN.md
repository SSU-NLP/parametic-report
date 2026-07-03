# parametic-studio 프로덕션화 종합 설계 (전수조사 + 플랜)

> 2026-07-03. 대상 4영역: ① 프로세스 라이프사이클 ② 외부 GPU/클라우드 ③ 메모리 최적화 ④ UX 디테일.
> 조사 기준: `parametic_studio/`(kernel·api) + `studio_web/src/App.tsx` 전량 + 운영 중 실측(메모리 top, WS e2e).
> 현황 기능 목록은 `HANDOFF.md` §2, WS 계약은 §3 참조. 이 문서는 **격차와 해결 설계**만 다룬다.

---

## 1. 프로세스 라이프사이클 — 프론트·커널 동시 동작

### 1.1 전수조사 (현황과 문제)

| # | 사실 | 근거 |
|---|---|---|
| L1 | 커널(uvicorn :8000)과 웹(vite :5173)은 **완전히 독립된 두 프로세스** — 각각 수동 실행, 상호 감시 없음 | `api.py serve()` / `.claude/launch.json`엔 `studio_web`만 등록 |
| L2 | 커널이 죽어도 웹은 계속 뜸 — 새 세션은 "kernel offline" 빈 상태를 보여주지만, **이미 열린 페이지는 감지 못 함** (소켓 onclose 핸들러 없음, 상태바 `kernel :8000` 하드코딩) | App.tsx `socket()` — onmessage만 등록 |
| L3 | WS **자동 재연결 없음** — 커널 재시작 후 기존 페이지는 다음 send 때 소켓을 새로 만들지만(readyState 체크), 모델 세션·구독 상태는 어긋남. 사실상 새로고침 필요 | App.tsx `socket()`, 이 세션에서 실사용자도 반복 경험 |
| L4 | 웹만 꺼도 커널은 모델을 쥔 채 잔존 (RAM 수 GB) — "브라우저 닫으면 정리"가 없음 | WS disconnect 시 세션 유지가 현재 의도(재접속 빠름)이지만 옵션도 없음 |
| L5 | 커널 코드 수정 시 **수동 재기동 필수** (uvicorn reload 미사용) — 개발 루프 마찰 | HANDOFF §1 함정 항목 |

### 1.2 설계 — **결정(2026-07-03): Tauri 데스크탑 앱 + 시스템 python** 이 P1의 답

앱 셸이 수명 결합을 구조적으로 해결하므로 런처 스크립트 단계를 건너뛰고 바로 앱화한다.
배포 대상 = 본인·연구실 (python+torch 번들 안 함 — 앱 ~10MB, 커널은 시스템 python으로 spawn).

**P1' 구성**
- **Tauri 셸**: `studio_web` dist/를 웹뷰로 로드. Rust 사이드가 커널을 child process로 spawn
  (`python -m parametic_studio.api`), **창 닫으면 커널 종료** — 수명 결합 완성.
- **python 발견**: 설정(초기: `~/.parametic_studio/config.json`의 `python_path`, 기본 `python`) →
  실패 시 앱 첫 화면에서 경로 입력 + `pip install -r requirements-studio.txt` 안내.
  `requirements-studio.txt` 신규(커널 의존성 분리: torch/transformers/fastapi/uvicorn/websockets).
- **프론트 연결 상태 감시** (앱에서도 커널 크래시는 발생): 소켓 `onclose`/`onerror` → 상태바
  `kernel :8000` ↔ `kernel offline · reconnecting…` + **지수 백오프 재연결**(1s→…→30s).
  재연결 성공 시 `catalog`/`regions`/`datasets` 재요청 + 열린 모델 `open` 재전송으로 재동기화.
- **dev 모드 유지**: 개발은 기존대로 vite+커널 수동 (Tauri `devUrl`이 :5173를 가리키므로 동일 코드).
- 선행 조건: Rust 툴체인 설치 필요(현재 미설치 — `rustup` 1회), P3-c의 `window.prompt/confirm` 제거는 앱화 전 필수로 승격.

**검증**: 앱 실행 → 커널 자동 기동·모델 로드. 창 닫기 → 커널 프로세스 사망. 커널만 kill → 상태바 offline → 자동 재기동은 하지 않되(중복 방지) "restart kernel" 버튼 노출 → 복구·재동기화. dev 모드(vite) 회귀 없음.

---

## 2. 외부 GPU / 클라우드 연결

### 2.1 전수조사

| # | 사실 | 근거 |
|---|---|---|
| C1 | 커널은 **이미 순수 WS 서버** — 프론트와의 결합은 URL 하나뿐. 즉 "원격 실행"의 기술적 장벽은 낮음 | `App.tsx:3 WS_URL='ws://localhost:8000/ws'` **하드코딩**이 유일한 결합 |
| C2 | `device.py resolve_device('auto')`가 mps→cuda→cpu 순 — **cuda 머신에서 커널을 그대로 실행 가능** (검증은 안 됨: bf16 cuda 경로, eager attention 대형 모델 성능) | device.py |
| C3 | repo에 **VESSL 하네스 기존재**: `scripts/vessl/{config,push,submit,watch}.sh` — A100 spec, 오브젝트/클러스터 볼륨, `.vesslrc` 네임스페이스. 단 **batch job용** (제출→종료), 상주 커널용 아님 | scripts/vessl/config.sh |
| C4 | HANDOFF §4 C에 방향 합의 존재: "VesslBackend: workspace(exposed port)에서 커널 실행 → api가 WS 프록시" | HANDOFF.md |
| C5 | **인증 없음** — 커널 WS는 무인증. localhost 전제라 괜찮았지만 원격 노출 시 모델·파일시스템(datasets link_path!) 접근이 열림 | api.py 전체 |

### 2.2 설계 — "커널은 어디서든, 프론트는 URL만"

원격화의 핵심 통찰: **커널을 옮기는 게 아니라 프론트가 가리키는 곳을 바꾸는 것**. 세 단계로 점진.

**P4-a. 프론트 kernel URL 설정 (필수 기반, 소형)**
- `WS_URL` 하드코딩 제거 → 상단바 `[⚙ kernel]`: URL 입력(기본 `ws://localhost:8000/ws`) + localStorage. 커널 전환 시 전체 상태 리셋(모델·데이터 재요청).
- 다중 커널 등록(로컬 mps + 원격 cuda 동시)은 후속 — 현 타일 시스템이 model-id 키라 kernel-id 축 추가가 큰 작업. v1은 **한 번에 하나**.

**P4-b. SSH 터널 경로 (코드 변경 최소, 가장 실용적)**
- 원격 GPU 서버: `pip install -r requirements-studio.txt`(신규 — 커널 의존성 분리) 후 `python -m parametic_studio.api`.
- 로컬: `ssh -N -L 8000:localhost:8000 user@gpu-host` → 프론트는 기존 URL 그대로. **보안도 SSH가 해결** (커널 무인증 유지 가능, C5 완화).
- 산출물: `docs/REMOTE_KERNEL.md` 가이드 + `scripts/studio-remote.sh <host>` (ssh 터널 + 원격 커널 기동 + 로컬 vite까지 원커맨드).
- `PARAMETIC_STUDIO_HOME`이 원격 파일시스템이 됨 — regions/datasets가 원격에 저장됨을 문서화. `link_path`도 원격 경로 기준.

**P4-c. VESSL workspace 경로 (상주 클라우드)**
- batch 하네스와 달리 **workspace**(상주 컨테이너 + exposed port) 사용: `vesslctl workspace create --image pytorch/... --port 8000` → 커널 실행 → exposed URL을 P4-a 설정에 입력 (`wss://`).
- exposed port는 공개 URL이므로 **최소 인증 필수**: 커널에 `PARAMETIC_STUDIO_TOKEN` env → WS 접속 첫 메시지 `{auth: token}` 검증 (1회, 이후 통과). 프론트 설정에 token 필드.
- 기존 `scripts/vessl/config.sh`의 org/spec/볼륨 상수 재사용, `scripts/vessl/workspace.sh` 신규(생성·기동·URL 출력).
- 모델 캐시: HF_HOME을 오브젝트 볼륨에 → workspace 재생성에도 다운로드 1회.

**검증**: (a) URL 바꿔 두 커널 전환, 상태 재동기화. (b) ssh 터널로 cuda 서버 커널에 붙어 spot→knob→train 전 루프. (c) VESSL workspace에서 7B 모델 로드 + 토큰 인증 거부/허용.

---

## 3. 메모리 최적화 — "끝난 결과는 디스크로, 메모리에서 내리기"

### 3.1 전수조사 — 잔존 데이터 인벤토리

이 세션에서 이미 수정된 것(커널 allocator hoard +3.3GB/spot→+6MB, SESSION 참조 고정, 렌더 재파싱 GB churn)은 제외. **남아 있는 잔존물**:

**프론트 (JS heap)**
| # | 잔존물 | 크기 감각 | 언제까지 |
|---|---|---|---|
| M1 | `data[mid].frames` — 생성의 **전 스텝 attention 행렬** (JS number = 8B/원소). 256tok·kv 300이면 행렬 합계 ~1M 원소 ≈ 수십 MB/생성 | 생성당 수~수십 MB | **다음 프롬프트까지** (핵심 지적 대상) |
| M2 | `ab.base/inter` 출력 텍스트, `output`, `act`, `logit` | 소형(KB) | 다음 프롬프트까지 |
| M3 | `expLog` — spot/train 액션의 `py` 문자열에 **examples 전문 직렬화** (2000예제 spot 1회 = ~1MB 문자열) | 액션당 최대 MB, **세션 내 무한 누적** | 세션 끝까지 |
| M4 | `datasets[].content` + `dsMeta.examples` — 로드한 데이터셋 원문+파싱본 이중 보관 | 파일당 ~2× | 세션 끝까지 |
| M5 | `regionInfo`/`compareData` 그리드 | 소형(24×12) | 무시 가능 |

**커널 (프로세스 RSS)**
| # | 잔존물 | 크기 감각 | 언제까지 |
|---|---|---|---|
| M6 | `_trained` 백업 — full/spot-freeze는 **CPU 전량 스냅샷** (0.5B≈1GB, 1.5B≈3.1GB) | GB급 | **reset_training까지** (의도된 가역성 비용이지만 옵션 없음) |
| M7 | `regions` 마스크 + `region_grids` — region당 ~0.36GB(0.5B), **시작 시 전부 로드** | region 5개 ≈ 1.8GB | 세션 끝까지 |
| M8 | `last_raw`(마지막 스텝 per-head attention), `_acts` | 소형 | 무시 가능 |
| M9 | region `.pt` bool 덤프 (디스크) — region당 ~340MB | 디스크 | packed indices로 ~1/8 가능 |

### 3.2 설계

**P2-a. 생성 결과 flush (M1·M2, 사용자 지적 직결) — M2 기록/리플레이와 통합**
- 생성 `done` 시 프론트가 런 전체(`{prompt, output, frames, act, logit, settings, ts}`)를 커널 `save_run`으로 전송 → `$PARAMETIC_STUDIO_HOME/runs/<model>/<ts>.json.gz` 저장 → **프론트는 frames를 즉시 비움**(마지막 프레임 1장만 유지 — attention 뷰 기본 표시용).
- attention 뷰에서 과거 스텝(인과 삼각형)을 열면 `load_run`으로 lazy 재로드. 뷰 닫으면 다시 해제.
- 이것이 보류 백로그의 **M2 기록/리플레이(recordings) 기반 인프라**가 된다 — runs 디렉토리 = 리플레이 소스. experiment log의 결과 연동도 여기 붙음.
- 절감: 생성당 수십 MB → 마지막 프레임 1장(수십 KB). 트레이드오프: 삼각형 뷰 첫 오픈 시 디스크 왕복(~수십 ms).

**P2-b. frames 표현 최적화 (M1 보완)**
- WS attention payload를 현재 `number[][]` JSON 대신 **base64 Float32**(또는 binary frame)로: 파싱 GC·보관 메모리 ~8× 절감. 커널 `row.numpy().tobytes()` → 프론트 `Float32Array`. Grid 렌더는 TypedArray 인덱싱으로 동일.
- P2-a로 보관이 사라지면 우선순위 낮음 — **전송·파싱 churn**이 남는 대형 생성에서만 가치. 측정 후 결정.

**P2-c. expLog 대형 문자열 (M3)**
- `actionPy`에서 examples가 N자(기본 2KB) 초과 시 `examples  # <name>, 2000 examples — see datasets/<name>` 로 참조 치환. replay.py 재현성은 데이터셋 이름 참조로 유지(어차피 데이터셋이 디스크에 있음).
- log 자체도 P2-a의 runs 디렉토리에 주기 flush(세션 로그 영속화 겸).

**P2-d. 커널 백업·region 상주 (M6·M7)**
- `_trained` 백업을 **디스크 스냅샷 옵션**으로: `train_steps(..., snapshot='disk')` → `runs/train-backup-<ts>.pt` 저장 후 메모리 해제, reset 시 로드. 기본은 현행(메모리 — 빠른 reset). 1.5B full FT 시 UI가 disk 스냅샷 권고 (HANDOFF의 "full+1.5B 경고"와 통합).
- `regions`를 **lazy 로드**: 시작 시 이름·count 메타만(사이드카 `.meta.json` — count는 저장 시 기록), 마스크는 intervene/train/compare 첫 사용 때 로드 + LRU 1~2개 유지. region 5개 상주 1.8GB → ~0.
- 디스크 포맷 v3(M9): bool 덤프 → `nonzero int32 인덱스` 저장 (~1/8). v1/v2 로드 호환 유지.

**P2-e. 관측성**
- 상태바에 **커널 RSS + 로드 모델 수 + JS heap** 표시 (커널 `stats` WS, `psutil` 없이 `resource.getrusage`). "메모리가 왜 이래"를 사용자가 즉시 보게 — 이 세션의 2.9GB류 이슈 조기 발견용.

**검증**: 256tok 생성 10회 후 JS heap 플랫 확인(P2-a). 커널: region 5개 + 학습 1회 시나리오에서 RSS 피크 비교(전/후). runs 디렉토리에서 리플레이 재로드 렌더.

---

## 4. UX 디테일 전수조사

### 4.1 액션별 피드백 커버리지 (전수)

| 액션 | pending 표시 | 취소 | 에러 표시 | 갭 |
|---|---|---|---|---|
| generate | ▌커서·`●` 칩 | `■ stop` ✅ | ❌ (WS error 미소비) | 에러 무표시 |
| spot | progress bar+`i/N` ✅ | `[stop]` ✅ | ❌ | — |
| train | loss 커브+`n/N` ✅ | `[stop]` ✅ | ✅ (train 뷰 한정) | — |
| 모델 load | ⟳+경과+슬라이드 바 ✅ | ❌ (취소 불가) | ❌ (**실패 시 ⟳ 영구**: 잘못된 HF id 입력하면 loading 고착) | **높음** |
| ppl / eval | ❌ (누르고 수 초 무반응) | — | ❌ | 중간 |
| intervene(spot-kind) | ❌ (**locate 백워드로 수 초~분**, 무반응) | ❌ | ❌ | **높음** |
| save_region(spot-kind) | ❌ (동일 — 재계산 수 초~분) | ❌ | ❌ | **높음** |
| region_info/compare | `loading…`/`comparing…` ✅ | — | ❌ | — |
| dataset read/save/link | 목록 갱신으로 암시 | — | ❌ (**link_path 실패 시 무반응** — confirm/에러 안 보임) | 중간 |
| drilldown | ❌ | — | ❌ | 낮음 |
| A/B compare | `generating…` 패널 ✅ | `■ stop`(수정됨) | ❌ | — |
| WS 끊김 | ❌ (§1 L2와 동일) | — | ❌ | **높음** (P1에서 해결) |

### 4.2 설계 — 공통 피드백 인프라 + 개별 패치

**P3-a. 에러 토스트 (공통 기반)**
- 커널 `error` 메시지를 전역 소비: 우하단 토스트 스택(5초 자동 소멸, hover 유지). 현재 train 뷰만 error를 쓰고 나머지는 **버려짐** — 커널의 모든 예외 응답이 사용자에게 보이게 됨.
- 커널 쪽: 핸들러 예외를 try/except로 `error{reason, op}` 응답 통일 (현재 일부 op는 예외 시 WS 연결이 죽음 — generate 중 커널 예외 등).

**P3-b. 장기 작업 pending 통일**
- `sendTo`에 요청 op 기반 pending 레지스트리: `pending: Set<op>` → 해당 버튼 스피너·disable. 대상: ppl/eval, intervene, save_region, drilldown. (spot/train은 자체 progress 유지)
- intervene·save_region의 spot-kind는 **커널이 locate 진행을 `spot_progress`로 재사용** 스트림 → knob 보드에 미니 progress. (커널: locate_spot에 progress 콜백 — spot 핸들러와 공용화)
- 모델 load 실패: `_ensure` 예외를 `error` + `load_failed{model}` 응답 → 칩 제거+토스트. 타임아웃 안내(대형 모델 다운로드 중 "n GB 모델은 수 분 걸립니다").

**P3-c. 마이크로 디테일 (인벤토리 즉시 수정 목록)**
- 상태바: 정적 `kernel :8000` → 연결 상태·RSS·활동 인디케이터(P1·P2-e와 통합).
- 버튼 hover/active 스타일 부재(현재 커서만 변함) → 공통 버튼 컴포넌트로 수렴 (App.tsx에 인라인 스타일 버튼 ~30곳 — 중복 정리 겸).
- `window.prompt`/`confirm` 3곳(HF id, dataset name, region 삭제) → 인라인 입력/팝오버로 (앱화 시 prompt 미지원 環境 대비).
- 숫자 입력 유효성(음수 steps, topk>100% 등) 가드.
- 단축키: `⌘Enter` 전송, `Esc` stop.
- log 뷰 자동 스크롤 + 항목 수 뱃지.

**검증**: 각 갭 항목 재현 시나리오(잘못된 HF id, 커널 kill 중 eval, link_path 오타)가 전부 시각 피드백을 내는지 수동 체크리스트.

---

## 5. 실행 계획 (라운드)

| 라운드 | 내용 | 규모 | 의존 |
|---|---|---|---|
| **P1' 앱화+라이프사이클 ✅ (2026-07-03 완료)** | Tauri 셸(+시스템 python spawn, 창 닫으면 커널 종료 + 커널 부모감시로 고아 방지), WS onclose+재연결+재동기화, 상태바 연결 상태, requirements-studio.txt, prompt/confirm 제거. e2e: spawn 2s·SIGTERM 후 커널 자살 6s·고아 0 | 중 | ~~Rust 툴체인~~ 설치됨 |
| **P2 메모리 ✅ (2026-07-03 완료, opus/sonnet 위임)** | 생성 flush(runs/ gzip+index+retention 50)+`[load full history]` lazy 복원, expLog truncate, region lazy(meta 사이드카+LRU 2+legacy 마이그레이션 off-loop), stats(RSS 상태바). 95 tests. (frames binary·train 디스크 스냅샷·포맷 v3는 측정 후 후속) | 중 | — |
| **P3 UX 패스 ✅ (2026-07-03 완료, opus/sonnet 위임)** | 에러 토스트(error{model,op,reason} 통일+연결 생존)+locate_progress 스트림, pending 레지스트리(⟳/disabled), load 실패 토스트, Btn 공통화(25곳)·Esc stop·클램프·로그 자동 스크롤. 109 tests, 토스트 라이브 e2e | 중 | P1(상태바) |
| **P4 원격 커널 ✅ (2026-07-03 완료, opus/sonnet 위임)** | kernel URL/token 설정(전환=reload), `PARAMETIC_STUDIO_TOKEN` auth 게이트(첫 프레임/4401), studio-remote.sh(ssh)+workspace.sh(EXPERIMENTAL)+REMOTE_KERNEL.md. 113 tests, 토큰 커널 라이브 e2e. 미검증: VESSL 라이브·bf16 cuda·다중 커널(후속) | 중~대 | P1(재연결이 원격에서 필수), P3(에러 표면) |
| **P5 UI 개선 ✅ (2026-07-03 완료, opus×2/sonnet 순차 위임)** | IDE 스타일 전환: 산세리프 UI+모노 데이터 전용, 브래킷 34곳 제거, 트리 IDE화(SVG 아이콘·인덴트 가이드·선택), 우클릭 ContextMenu+Del/Cmd+S, CodeMirror 6 에디터(gzip +174KB). 113 tests, build 통과 | 중 | P3(Btn/토스트 기반) |
| (후속) | 완전 번들 배포판(python+torch 포함 DMG, 코드사이닝), frames binary, 다중 커널, CodeMirror 코드 스플리팅 | 대 | P1~P5 |

각 라운드 완료 기준: 기존 원칙 유지 — pytest green + tsc 0 + 라이브 검증 + HANDOFF 갱신.

## 6. 리스크 / 미결정
- **원격 인증 수준**: 토큰 1개로 충분한가(연구실 내부망 전제) vs TLS+사용자 — v1은 토큰+ssh 터널 권장으로 결정 필요.
- **runs 저장 용량**: attention 전 스텝 저장은 생성당 수 MB(gz) — 자동 정리 정책(최근 N개) 필요.
- **재연결 시 커널 상태 정합**: 학습 중 재연결하면 busy 가드와 프론트 상태가 어긋날 수 있음 — 커널에 `status` 스냅샷 op(모델별 busy/training/knob 수) 추가로 해소 예정.
- **frames flush와 M2 리플레이 스코프 경계**: P2-a는 저장+해제까지, Timeline 스크러버 UI는 별도 백로그 유지.
