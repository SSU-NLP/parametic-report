# parametic-studio — 핸드오프 (세션 재개용)

> 콜드 재개용. 이 문서 먼저 읽고, 설계는 `STUDIO_VIEW_ARCH.md`·`STUDIO_UI_SPEC.md` 참조.
> 마지막 커밋: `05f09b6` (브랜치 `ui/redesign`) + **미커밋 대규모 작업분** (아래 §2 후반부 전부).

## 0. 무엇 / 어디에

**연구자용 모델 파라미터 워크벤치** (MATLAB/RStudio 지향). 살아있는 커널 + VS Code풍 멀티뷰 웹.
- `parametic_studio/` — `device.py`(mps bf16), `kernel/model_session.py`(로딩·디코드·probe·spot·**knob·train**),
  `kernel/lora.py`(**vendored LoRA** — training/utils의 deepspeed 의존 제거판), `api.py`(FastAPI WS·다중세션), `catalog.py`.
- `studio_web/` — React+Vite+TS, opencode 다크. 단일 파일 `src/App.tsx`.
- `tests/studio/` — tiny eager 모델 CPU + WS TestClient. **`python -m pytest tests/ -q` → 113 passed.**

## 1. 실행법

```bash
# 데스크탑 앱 (P1' — 권장): 커널을 자동 spawn, 창 닫으면 커널 종료. vite(:5173) 먼저 떠 있어야 dev 모드.
npm --prefix studio_web run dev &          # dev 웹뷰 소스
cd studio_web && npx tauri dev             # 앱 실행 (요: rustup — 설치됨. :8000 선점 시 attach 모드)
# 설정: ~/.parametic_studio/config.json {"python_path","kernel_dir","model"} — 없으면 python + repo 루트 추정

# 웹 전용 (기존 방식):
python -m parametic_studio.api            # :8000. env PARAMETIC_STUDIO_MODEL로 모델 변경(기본 1.5B).
npm --prefix studio_web run dev           # :5173
python -m pytest tests/ -q                # 106 green
cd studio_web && npx tsc --noEmit -p .    # 0
```
**함정: `api.py`/`kernel` 수정 후 커널 반드시 재기동**(reload 없음). pkill 금지 — `ps ax | grep parametic_studio.api | grep -v grep | awk '{print $1}' | while read p; do kill $p; done`.

## 2. 완료된 것

**커밋 dd53850/05f09b6**: 커널(eager, 수동 그리디, probe: attention 정사각 N×N·activation·logitlens), 다중세션 WS(모델당 소켓 1), 비블로킹 지연로드, 5뷰 드래그 탭 + 2단계 그리드 + Chat dock + sync, 모델 load/unload.

**미커밋 (이번 세션들)**:
- **A. 스트리밍 spot + 프리셋**: `spot_step(example, acc, n)` 분해 → WS `spot_progress{i,total,grid}` 스트림(to_thread 비블로킹) + `spotmap`. 프론트 `[python spot]`·`[java spot]` 프리셋 + progress bar + 라이브 grid.
- **빈 상태 UX**: 모델 1개도 eject 가능, 0개면 `[no model loaded]` + (catalog 유무별) 안내. 재오픈 시 타일 재배정.
- **knob 믹싱보드 (B1+B2)**: `locate_spot(examples, topk)`(**per-param top-k%** — create_approx_spot_masks와 동일 의미), `locate_cell`, `intervene(region, op, alpha, key)`(**다중 key 동시**, 백업/복원 가역, op=scale|zero|mean|random), `clear(key|all)`. 프론트: spot 셀 클릭→knob 행(독립 op/α), `[+ top-k% spot]` whole-spot knob, α 숫자 정밀 입력.
- **R1 A/B compare**: `suspend()`/`resume()`(weight 원상↔재적용, 엔트리 유지) → **baseline｜intervened output 2-pane**. 검증: spot ×0 → baseline 정상 quicksort vs intervened 쓰레기 토큰.
- **R2 region 워크스페이스 + eval**: `save_region(name)` + `{kind:"named"}` region 참조, Explorer regions 목록, `[eval code｜general]` 2열 PPL(선택적 손상 지표).
- **R3 region-aware fine-tune/LoRA**: `train_steps(examples, mode, steps, lr, region, lora_dim)` 제너레이터(AdamW **weight_decay=0 필수** — decay가 mask 무시하고 weight 움직임) + `reset_training()`(모드별 백업: full/spot-freeze=CPU 전량, spot-only=선택값만, lora=모듈 제거). 모드: full / **spot-freeze**(grad hook `masked_fill`) / **spot-only**(requires_grad+`g*m`) / **lora**(vendored, base+embed/norm/lm_head 전부 동결). WS `train`→`train_step` 스트림→`trained`, `stop_train`, `reset_train`, **TRAINING busy 가드**. 프론트 train 뷰: 모드+region+steps/lr + 라이브 loss 커브(SVG) + before/after code|general PPL. 검증: spot-freeze 8step → reset → PPL bit-exact 원상.
- **R4 experiment log**: 모든 액션/결과 자동 기록(`sendTo` 단일 지점), log 뷰(테이블/py 토글), **JSON/CSV/replay.py 내보내기**. 세션 휘발성(M2 기록과 통합 예정).
- **Explorer 에디터화**: `[ models ]` — **모델 노드 하위** VS Code식 재귀 폴더 트리(모든 세그먼트 접이식, leaf만 shape·dtype, WS `tensors` lazy fetch, 모델별 독립). `[ data ]` — 파일 클릭 → **타일에 `data:<name>` 동적 에디터 탭**(편집 즉시 datasets 반영 + `[spot]`·`[→ train data]` 액션), `[+ file]`/`[+ folder]`(webkitdirectory)/`[+ editor]` 추가. Tile.tabs는 string[](View + `data:*`). 폭 150→230. 데이터는 프론트 상태(세션 휘발) — spot/train 메시지가 examples 운반하므로 커널 무관.
- **데이터셋 스토어 + 라이트모드**: 데이터셋은 커널 관리 `$PARAMETIC_STUDIO_HOME/datasets/`로 — WS `datasets`/`read_dataset`(lazy)/`save_dataset`/`delete_dataset`/`link_path`(외부 파일·폴더 **symlink** 연결, path escape 가드). 업로드(`[+ file]`/`[+ folder]`)는 스토어에 저장(영속), `[+ path]`로 경로 연결, 에디터 탭 `[save]`. 프리셋 3종만 클라 세션(▢ 표시). 라이트모드: `[data-theme="light"]` 팔레트 + 상단 `[☾/☀]` 토글(localStorage), 히트맵은 양 테마 다크 패널 유지.
- **region 디스크 영속화**: `save_region` → `$PARAMETIC_STUDIO_HOME`(기본 `~/.parametic_studio`)`/regions/<model-id>/<name>.pt` 저장, `ModelSession(model_id=…)` 생성 시 자동 복원(모델 id 없으면 메모리만 — 테스트 무영향). 프론트는 catalog/opened 시 `regions` 요청. ⚠️ bool 덤프 ~1B/param(0.5B 풀 region ≈ 340MB) — 디스크가 아까우면 packed indices로. datasets/experiment log는 아직 휘발(M2).
- **데이터 파싱 + 샘플링 + spot stop**: `toExamples(content, fields?)` — JSON 배열/JSONL 레코드 인식, **per-dataset 필드 칩 토글**(복수=조합, auto=text-ish→최장 문자열) + parsed[0] 미리보기. spot 뷰 **sample n + first-k/random**(실체화 — textarea에 샘플본). spot 진행 중 `[stop]`(`stop_spot` → partial spotmap, reason).
- **시각화 가독성**: 테마별 컬러맵(라이트=밝은 베이스→진한 램프), 전 히트맵 **셀 툴팁**(값 포함)+**ScaleBar**(0→max), attention layer **hover=미리보기/click=고정**, activations 설명문(마지막 토큰 모듈 출력 L2 norm), **region 뷰어 탭**(Explorer ◈ 클릭 → 선택 비율 그리드, WS `region_info`).
- **P7 인앱 SSH 원격 커널 (russh, opus+sonnet 위임)**: "로컬만 됨 → 사용자 원격 GPU를 SSH로". Rust `ssh.rs`(순수 러스트 `russh` 0.54 — 시스템 SSH 무의존, Windows 배포 가능): `ssh_connect(host,port,user,password,repo_dir,python_path?,model?)` — 비번 인증 → 원격에서 `PARAMETIC_STUDIO_HOST=127.0.0.1` 커널 기동(이미 :8000 있으면 재사용, 60s 폴링) → **로컬 8422→원격 127.0.0.1:8000 direct-tcpip 포워드**(raw 양방향, WS 업그레이드 통과, Rust 소유라 webview reload 생존) → `ssh-status` 이벤트. `ssh_disconnect`는 터널만 닫고 원격 커널 유지. 보안=SSH(토큰 불필요), 비번은 메모리만(로그·디스크 금지). 프론트: 설정 **Local | Remote(SSH)** 토글 + 필드 + Connect/Disconnect, 성공 시 기존 P4 kernel-URL 재사용(`ws://localhost:8422/ws`+reload), 브라우저에선 비활성. **검증**: cargo check·tsc 0, 앱 빌드, **라이브 e2e** — localhost sshd에 틀린 비번 Connect → `error: auth failed: password rejected` 인라인 표시(russh↔sshd 연결·인증·에러 UX 전 경로 확인). 성공 분기(정상 비번→커널→터널→WS)는 실 자격증명 필요라 사용자 확인 몫. 다중 커널·키 인증은 후속.
- **P6 후속 버그픽스 (실사용 신고 2건, 리드 직접)**: ① **스플래시 무한 로딩** — 8s fallback 타이머가 숨겨진 메인 창 webview에 있었는데 macOS WKWebView가 숨김 창 JS 타이머를 정지 → 커널 부재 시 영원히 스플래시. **Rust 스레드로 이전**(setup에서 8s 후 `do_close_splash`, 웹뷰 사정 무관). ② spawn_kernel에 `dir/parametic_studio` 존재 검증(패키징 앱 cwd=`/` fallback 무의미) — 없으면 spawn 포기+로그. ③ "강제 종료에도 python 생존"은 **attach 모드**(기존 :8000 커널)의 설계 동작으로 판명 — spawn 커널은 SIGKILL 후 ~2s 내 부모감시 자살 **실검증**. 시나리오 3종(무설정→8s 오프라인 UI / spawn→SIGKILL→자살 / config 완비 해피패스) 패키징 앱 실기 통과. PACKAGING.md Troubleshooting 추가.
- **P6 크로스플랫폼 패키징 (opus 위임 + 리드 빌드 검증)**: v1 = **앱 셸만 번들, 커널은 사용자 python**(config.json 연결 또는 원격 커널). lib.rs Windows 호환 — `USERPROFILE` fallback, `CREATE_NO_WINDOW`(콘솔 팝업 방지), python 후보 순회(config→python3→python, win은 python 우선). `_watch_parent`는 **posix 전용 분기**(win은 ppid 불변이라 고아 감시 불가 — RunEvent::Exit이 주 방어선, 비정상 종료 시 작업관리자 caveat). `.github/workflows/studio-build.yml`(tauri-action, macos aarch64+windows msi/nsis, `workflow_dispatch`/`studio-v*` 태그) + `docs/PACKAGING.md`. package.json에 `tauri` 스크립트 추가(없어서 CI가 깨질 뻔). **로컬 mac 릴리즈 빌드 검증**: `Parametic Studio_0.1.0_aarch64.dmg` **3.2MB**(ad-hoc 서명), 패키징 앱 실행→기존 :8000 커널 attach→WS 정상→종료 시 attach 커널 생존(계약 준수). Windows는 CI 아티팩트로 실기기 검증 필요(미검증). 113 passed·cargo check·tsc 유지.
- **P5 UI 개선 — IDE 스타일 전환 (opus×2+sonnet 순차 위임, 사용자 방향 확정: 데스크탑 앱/IDE + CodeMirror)**: ① **디자인 시스템** — body `--font-ui`(SF Pro 계열 산세리프), 모노는 `.mono`로 코드·텐서·수치·로그에만 명시, `[ ... ]` 브래킷 레이블 34곳 제거(`.section-h` 헤더 / `.btn`·`.btn-primary` 실제 버튼 / 점 인디케이터 상태), `--radius`·`.panel` 토큰. ② **트리 IDE화** — `.tree-row`(22px, hover/선택 `color-mix` accent, ellipsis), 인라인 SVG `Icon`(cube/folder/file/database/link/diamond/tensor), 회전 chevron+레벨별 인덴트 가이드, ×는 hover 시만, `selected` 상태. **자작 ContextMenu**(우클릭 — 데이터셋: Open in editor/Use for spot/Use as train data/Delete · region: View/Compare/Delete · 모델: Unload — 전부 기존 함수 재사용, 삭제는 armed confirm을 메뉴 안에서 통과). 단축키: Del/Backspace=선택 행 삭제(armed 경유), Cmd+S=에디터 저장, Esc=stop 유지. ③ **CodeMirror 6 에디터** — `@uiw/react-codemirror`+lang-python/json(신규 dep 3개, gzip +174KB — 코드 스플리팅 후속), CSS 변수 연동 자작 테마+4색 하이라이트, Cmd+S는 CM keymap으로(전역 핸들러가 contentEditable bail-out이라 필수). 검수 중 수정: `tsc -b` TS2352(sendTo 캐스트 → `as unknown as`) — **`npm run build` 통과**, 에디터 컨테이너 배경 통일. tsc 0, 113 passed 유지, 프리뷰 e2e(트리/컨텍스트 메뉴/에디터 13줄 라인넘버) 확인. ⚠️ P5-C 에이전트가 작업 중 `git stash`×2 실행 사고 — pop으로 완전 복구, 무결성 전수 확인됨.
- **P4 원격 커널 (opus+sonnet 위임)**: **"커널은 어디서든, 프론트는 URL만"**. ① 프론트 — `WS_URL` 하드코딩 제거 → localStorage(`ps_kernel_url`/`ps_kernel_token`), 설정 패널 `[ kernel connection ]` 섹션(URL+token+`[connect]`=저장 후 `location.reload()` — 전환 시 전체 상태 리셋을 리로드로), 토큰 비어있지 않으면 모든 소켓 첫 프레임 `{type:"auth",token}`, close 4401 → 토스트 1회, 상태바 `kernel <host>`. ② 커널 — `PARAMETIC_STUDIO_TOKEN` env 설정 시 accept 직후 첫 프레임 auth 검증(불일치 `close(4401)`), env 없으면 게이트 없음+auth no-op(양쪽 커널에 같은 클라 동작). ③ `scripts/studio-remote.sh`(ssh 터널 원커맨드 — 보안=ssh, 토큰 불필요) + `scripts/vessl/workspace.sh`(**EXPERIMENTAL**, 라이브 미검증 — `VESSL_CLUSTER` env 필요) + `docs/REMOTE_KERNEL.md`(경로 a/b/c + 원격 `PARAMETIC_STUDIO_HOME`·`link_path`·bf16 cuda 미검증 주의). **113 passed**(+4), 라이브 e2e: 토큰 커널(:8001) 무인증/오토큰 4401·정토큰 catalog, 프리뷰에서 URL 전환→`kernel localhost:8001` 재동기화·stats 왕복. 다중 커널 동시 등록은 후속(v1은 한 번에 하나).
- **P3 UX 본선 (opus+sonnet 위임)**: ① **WS 에러 통일** — dispatch 전체 try/except(`WebSocketDisconnect`만 re-raise), 모든 예외 → `error{model, op, reason}` 송신 후 **루프 계속**(연결 생존), 흩어진 개별 error에도 op 필드. ② **locate 진행률** — `locate_spot(..., progress=None)` 콜백 → spot-kind intervene/save_region에서 `locate_progress{model, op, i, total}` 스트림(`asyncio.run_coroutine_threadsafe`, cell/named는 없음 — 종결 메시지 전까지 progress 드레인 필요). ③ 프론트: **에러 토스트 스택**(`[op] reason`, 우하단 max 4, 5s 자동+클릭 제거, load_failed도 토스트로), **pending 레지스트리**(`${op}:${mid}`, ppl/intervene/save_region/drilldown/region_compare/region_info → ⟳+disabled, error에서도 해제), `locating… i/N` 표시, **Btn 공통 컴포넌트**(25곳 치환+hover), Esc=stop(입력 포커스 제외), 숫자 클램프(steps≥1/lr>0/topk 0.1~100/n≥1), 로그 자동 스크롤(하단 pinned, 위로 스크롤 시 비간섭). 검수에서 TDZ 크래시(`anyBusy` 선언 전 참조 — tsc 미검출, 앱 전체 백지) 발견→직접 수정. **109 passed**(+3), tsc 0, 토스트 라이브 e2e 확인.
- **P3' 앱 완성 (oMLX 패턴, opus×2 위임)**: **"Parametic Studio" 리네이밍** + 로고(`public/logo.png`·앱 아이콘 `tauri icon` 전 사이즈, 원본 `assets/`) + **스플래시 창**(2-window, 첫 kernelUp 시 `close_splash`) + **네이티브 메뉴바**(App: Settings ⌘, / Model: Load ⌘L·Unload All / View: 테마 ⌘⇧T·탐색기 ⌘B — `emit("menu")`→프론트 매핑) + **설정 패널**(config.json 편집·캐시 모델 삭제·테마). **HF 다운로드 진행률**: `snapshot_download` + allow_patterns(safetensors/config/tokenizer만 — 단일 필터로 total과 동기화, blobs-only 폴링, pct 클램프) → `download_progress{pct,done_mb,total_mb}` 스트림 → 칩 실제 % 바. catalog에 installed/size_mb(+캐시 전용 모델 노출), `installed_models`/`delete_cached`/`get_config`/`set_config`. 검수 중 pct 144% 결함 발견→재지시→해소(272MB/100% 정확, 다운로드량 7배 절감). 106 passed.
- **P2 메모리 최적화 (STUDIO_PRODUCTION_PLAN, opus/sonnet 위임 구현)**: ① 생성 결과 flush — done 시(frames>32) `save_run` → `$HOME/runs/<model>/<ns>.json.gz`(+index.json, retention 50) 저장 후 **프론트 frames 마지막 1장만 유지**, attention 삼각형은 `[load full history]`로 lazy 복원 (WS `save_run/runs/load_run`). ② **region lazy 로드** — init은 파일 스캔만, `get_region()` LRU cap 2(디스크 백업분만 evict), `.meta.json` 사이드카(count+importance grid — region_grid가 마스크 없이 응답), legacy 1회 마이그레이션(off-loop). ③ expLog examples truncate(>2KB→elided 표기). ④ `stats` WS(ps RSS)+상태바 `rss N.NG`(15s 폴링). 검증: runs 왕복·retention·escape 가드·LRU cap e2e, 106 passed.
- **P1' 데스크탑 앱 + 라이프사이클 (STUDIO_PRODUCTION_PLAN)**: **Tauri 셸**(`studio_web/src-tauri/`, ~100줄 Rust) — 실행 시 커널 spawn(:8000 선점 시 attach), 창 닫으면 kill + **커널 부모 감시**(`PARAMETIC_STUDIO_PARENT_WATCH=1` → ppid 변화 시 self-exit — 강제종료/크래시에도 고아 0, e2e 검증). 설정 `~/.parametic_studio/config.json`. **WS 재연결**(onclose→지수 백오프→catalog/regions/datasets+open 재동기화, unload는 `_intentional` 마킹), 상태바 `kernel offline · reconnecting…`. **prompt/confirm 전부 제거**(웹뷰 대비 — 인라인 입력 3곳 + 2단계 `sure?` 3곳). `requirements-studio.txt` 신규.
- **region importance 저장 + 삭제**: spot region 저장 시 **|g×w| importance 그리드 동봉**(디스크 v2 `{"masks","grid"}`, v1 호환 — per-param top-k%라 fraction 그리드는 구조상 균일해 "찾을 때 본 spot"과 달랐던 문제 해결). region 뷰어·compare가 importance를 표시(legacy는 fraction 라벨). `delete_region`(메모리+디스크, WS 응답=갱신된 regions 목록), Explorer ◈ 행 × (confirm).
- **region compare (데이터셋별 spot 비교)**: 커널 `region_compare(names)` — per-region 셀 grid + **전체 교집합 grid** + **pairwise Jaccard**(마스크 단위). WS `region_compare`→`region_comparison`. 프론트: `[ regions ]` 헤더 `[compare]` → compare 탭 — region 칩 토글(2~4개, **region별 고유 색상** REGION_HUES), 각자 색 램프 grid 나란히 + intersection grid + Jaccard %. 검증: py-spot vs gen-spot = **18.1%** (분포 따라 spot이 실제로 다름). 흐름: 데이터셋별 [spot]→[save region]→[compare].
- **custom HF 모델 로드**: `+ model` 드롭다운에 `custom (HF id)…` — 임의 HuggingFace id 입력 → 기존 open 경로(from_pretrained가 hub 다운로드). 제약: Llama/Qwen식 디코더(model.layers.*·self_attn/mlp) 구조만 probe 동작. 검증: SmolLM2-135M 12s 로드→생성→unload.
- **생성 off-loop 수정 (stop·커널 프리즈 해결)**: `_run_generation`이 동기 제너레이터를 이벤트 루프 위에서 직접 돌려 긴 생성(256tok O(n²)) 동안 stop 리스너·WS keepalive가 굶던 문제(`keepalive ping failed` → 연결 사망 = "커널 죽음"처럼 보임). spot/train과 동일하게 `await asyncio.to_thread(next, gen)` 루프로. 검증: 256tok 생성 중 stop → **0.14s 내 중단**. 프론트 `■ stop` 버튼(기존 [x] 비가시), A/B 중 stop 체인 버그 수정, UI 한글 전부 영어화, **데이터셋 파싱 useMemo**(토큰 리렌더마다 전 JSONL 재파싱 → GB급 힙 churn 해결).
- **생성 옵션 패널**: chat dock `[⚙]` — max_tokens(기본 256, 기존 64 하드코딩 제거), **temperature**(0=greedy, >0 multinomial 샘플링 — 커널 generate에 추가), probe 토글(attention/activation/logitlens — 끄면 빨라짐). A/B에도 적용. 검증: greedy 결정적/샘플링 가변.
- **⚠️ 데이터 삭제 사고 + 수정**: 초기 `delete_dataset`이 폴더 symlink를 **통과해 원본 파일을 삭제**하는 버그(실사고 — repo `data/` 사본에서 복구됨). 수정: 링크 내부 항목 삭제 거부(에러), 링크 자체는 unlink만(원본 유지), 스토어 실파일만 삭제. `datasets` 항목에 `link` 필드, 프론트 ×는 종류별 confirm(⇗링크 해제/▤스토어 삭제/▢목록 제거). 회귀 테스트로 잠금. 사이드바(explorer·chat) **드래그 리사이즈**(localStorage).
- **메모리 누수 수정**: spot 1회에 +3.3GB 상주하던 것 → +6MB/run 정체. `free_memory()`(zero_grad+mps empty_cache)를 spot/intervene/save_region/generate 뒤 호출, region 마스크 **CPU 보관**(사용 시 디바이스 이동 — locate_*가 cpu 반환, intervene/_restore/train hook에서 `.to(device)`).
- **버그 수정**: ① spot per-param top-k(global→param별, Codex 리뷰) ② intervene 원자화(clear→locate→apply) ③ **모델 라우팅**(`_loaded` 폴백 제거→`_target`=`_ensure` — knob이 다른 모델에 걸리던 실버그, catalog에 `default` 실어 프론트 칩 동기화) ④ addTab 중복 키 ⑤ `_LAYER_RE` lora_* 제외.

## 3. WS 계약 (모델당 소켓, 모든 server→client에 `model`)

```
client→server: catalog · open · close · generate{prompt,max_tokens,temperature,probes} · stop · drilldown{layer}
  · spot{examples} · stop_spot · region_compare{names} · intervene{region,op,alpha,key} · clear{key?} · suspend · resume
  · ppl{examples,tag?} · save_region{name,region} · regions · tensors
  · datasets · read_dataset{name} · save_dataset{name,content} · delete_dataset{name} · link_path{path}
  · train{mode,examples,steps,lr,region?,lora_dim} · stop_train · reset_train
server→client: catalog{models,default} · loading/opened/closed · token · attention · activation · logitlens · perhead
  · spot_progress{i,total,grid} · spotmap{reason:done|stopped} · intervened{key} · cleared{key} · suspended · resumed
  · ppl{value,tag} · region_saved{name,count} · regions · tensors{tensors} · datasets{items} · dataset_content{name,content}
  · dataset_saved{name} · train_step{step,total,loss} · trained{mode,steps,reason}
  · train_reset · error{reason} · done{reason}
region spec: {kind:"spot",examples,topk} | {kind:"cell",layer,module} | {kind:"named",name}
학습 중(TRAINING set) 해당 모델의 다른 op → error"busy: training".
```

## 4. 다음 후보
- **`STUDIO_PRODUCTION_PLAN.md` (2026-07-03)** — 프로덕션화 종합 설계(전수조사): P1 라이프사이클(런처+WS 재연결) → P2 메모리(생성 결과 runs/ flush, region lazy) → P3 UX(에러 토스트+pending 통일) → P4 원격 커널(URL 설정→ssh→VESSL). **여기서 재개.**
- **커밋** (미커밋 작업분 큼 — 요청 시).
- knob: 겹치는 region 처리(현재 disjoint 전제 last-write-wins), transplant op, ROME rank-one(EasyEdit 어댑터), **pyvene 활성화 블록**(mps smoke test 선행 — deep-research 결론: 활성화 레고는 pyvene 재사용, weight는 자작이 우리 차별점. 메모리 `knob-intervention-design` 참조).
- train: 재학습 스태킹(v1은 reset 강제), full+1.5B 메모리 경고 UI, LoRA fuse("채택" 기능).
- log: 디스크 저장(M2 기록/리플레이 통합), drilldown/generate 결과의 로그 연동 확대.
- C. 클라우드(VESSL) — 큰 모델일 때만.

## 5. 보류 백로그 (기존 유지)
- M2 기록/리플레이(recordings.py + Timeline 스크러버), 토큰↔어텐션 오버레이(BertViz식), logit lens emergence-curve, 어텐션 시각적 1:1 셀, 드릴다운 last-step-only, 점진적 공개 UX.

## 6. 규약
- python = pyenv `python`(torch+mps). TDD(tiny eager 모델 CPU). ponytail(최소 코드).
- 커밋은 **요청 시에만**, 메시지 끝 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- 생성 dir·node_modules·체크포인트 커밋 금지.
