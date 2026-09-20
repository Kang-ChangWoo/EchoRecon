# HANDOFF — EchoRecon, 2026-09-20 ~12:40 KST

이 파일은 EchoRecon 작업을 다른 세션으로 넘기기 위한 인계서다. 원래 세션은
이후 av_localization(echoloc)만 다룬다. 아래는 코드·결과 파일 기준 사실이고,
md 문서의 주장이 아니다.

## 0. 소유자 지시 (그대로)

- "posterior(분포) 융합이 point 융합보다 나은가"의 판정판이 공정한지 점검하고,
  불공정한 부분을 고친 뒤 다시 돌려라. 지금 커밋된 결과로 결론을 내리지 마라.
- 1차 판정 지표는 F1 랭킹이 아니라 **ray 단위**(E21: coverage, Rescue@K,
  NLL/Brier/ECE, AUSE/Spearman). F1은 2차.
- 결과를 보기 전에 판정 기준을 파일에 적고, 본 뒤 바꾸지 마라.
  → `results/E30_stage_d/CRITERIA.md` (커밋 b5611cd, v3 실행 전에 작성).
- 보고 형식: 방법 × N × (수정 전, 수정 후) 표, 각 줄에 근거 파일·커밋, 재실행
  안 한 숫자는 [미확인], 결론은 "어떤 조건에서 어떤 지표가 어떻게 달랐다"로.
- val 점수로 랭킹 매기지 말 것. frac 같은 하이퍼만 val에서 고르고 값은 test.
- HEAR360 이름·코드를 레포에 노출하지 말 것(“the base model”, `../hear360` 참조).
- GPU 0 은 다른 사람 것. GPU 1만 쓴다(2–7은 다른 테넌트가 20 GB+ 점유).
- push: `git push https://<token>@github.com/Kang-ChangWoo/EchoRecon.git main`
  토큰은 파일에 저장 금지(소유자에게 받을 것). 커밋 author Kang-ChangWoo
  <branden.c.w.kang@gmail.com>, 트레일러 `Co-Authored-By: Claude Fable 5.1
  <noreply@anthropic.com>` + `Claude-Session: <세션 URL>`.

## 1. 지금 돌아가고 있는 것 (GPU 1)

| 작업 | PID | 로그 | 출력 | 예상 종료 |
|---|---|---|---|---|
| posterior_r2 재학습 (`--warmup-epochs 4 --lr 5e-5 --epochs 12`) | 554365 | `logs/train_posterior_r2_w4.log` | `outputs/posterior/posterior_r2_w4/{best,last}.pth, history.json` | 약 15:00 (12:45 재시작) |
| posterior_r8 재학습 (동일 설정) | 554474 | `logs/train_posterior_r8_w4.log` | `outputs/posterior/posterior_r8_w4/` | 약 16:00 (12:45 재시작) |

`ps -eo pid,args | grep "[t]rain_posterior"`로 확인. 죽이려면 PID로 kill
(pkill -f 는 자기 셸까지 죽인 전력이 있으니 쓰지 말 것). 같은 GPU에서
av_localization의 DisCo 재학습 두 개(각 2–4 GB)가 며칠간 돈다. 건드리지 말 것.

**12:40 사건**: 첫 시도는 같은 카드에 Stage D 4개를 겹쳐 띄우는 바람에 epoch 4에서 CUDA OOM으로 죽었다(로그 상단에 기록). 12:45에 처음부터 재시작했고, 그 뒤로는 이 카드에 다른 EchoRecon 작업을 띄우지 않는다. 재학습 중에는 Stage D를 동시에 2개 이상 띄우지 말 것(각 3–4 GB).

재학습 이유: 기존 head는 backbone이 풀린 첫 epoch(ep2)이 val KL 최소이고
이후 val KL이 단조 상승(1.35→2.13), argmax MAE는 ep9까지 개선(0.292→0.274).
즉 ep2 체크포인트는 "보정은 최고, argmax는 덜 익음". 소유자 요구로 warm-up을
늘리고 backbone lr을 낮춘 재학습을 1회 돌린다. 선택 규칙은 동일(val KL 최소).

## 2. 끝난 것 (커밋 b5611cd 코드로 실행, 결과는 아직 미커밋 — 아래 3번)

`results/E30_stage_d/` 아래:

| 디렉터리 | 내용 | 비고 |
|---|---|---|
| `r2`, `r8` | 최초 실행(0e85bc7 / 48f5398). 밴드 whole-bin, T=1, 격자 박스에 GT 사용 | 수정 전 |
| `r2_v2`, `r8_v2`, `*_v2_val` | git_commit.txt = a57f2cd 이지만 실행 시 밴드 보간이 **미커밋 상태**였음 → 출처 불일치. 참고용으로만 | 폐기 대상 |
| `r2_v3`, `r8_v3` | test. 밴드 interp, T val-fit(1.378 / 1.302), 박스 = A∪B 예측 범위 + 0.3 m (GT 무관), `boxes.csv`에 A/B 포함 여부 기록(234/234) | **정본** |
| `r2_v3_val`, `r8_v3_val` | 같은 설정, val 씬 3개(37 시퀀스). kept fraction 선택 전용 | |
| `r2_v3_whole` | v3와 동일하되 `--band whole`: 밴드 반올림 효과만 분리한 ablation | |
| `CRITERIA.md` | 사전 판정 기준 | |

수정 전/후 표(`python src/stage_d_compare.py --runs r2 r2_v3_whole r2_v3 --val - r2_v3_val r2_v3_val`)
핵심 행, test F1@0.2, v3는 val에서 고른 frac:

| 방법 | N | r2 (전) | r2_v3_whole | r2_v3 (후) | r8 (전) | r8_v3 (후) |
|---|---|---|---|---|---|---|
| A_point | 1 / 4 / all | 0.572 / 0.611 / 0.562 | 0.572 / 0.610 / 0.562 | 0.572 / 0.610 / 0.562 | 0.614 / 0.651 / 0.590 | 0.614 / 0.651 / 0.590 |
| B_argmax | 1 / 4 / all | 0.576 / 0.618 / 0.556 | 0.576 / 0.609 / 0.556 | 0.576 / 0.609 / 0.556 | 0.604 / 0.636 / 0.574 | 0.604 / 0.636 / 0.575 |
| C_on_B | 1 / 4 / all | 0.563 / 0.608 / 0.526 | 0.563 / 0.607 / 0.523 | 0.574 / 0.605 / 0.520 | 0.579 / 0.627 / 0.554 | 0.589 / 0.621 / 0.549 |
| C_grid | 1 / 4 / all | 0.595 / 0.600 / 0.524 | 0.586 / 0.602 / 0.521 | 0.587 / 0.594 / 0.512 | 0.601 / 0.622 / 0.546 | 0.612 / 0.613 / 0.534 |
| D_conf_sum | 1 / 4 / all | [미확인] | 0.589 / 0.617 / 0.549 | 0.589 / 0.617 / 0.549 | [미확인] | 0.613 / 0.642 / 0.573 |
| E_conf_filter50 | 1 / 4 / all | [미확인] | 0.424 / 0.473 / 0.400 | 〃 | [미확인] | 0.435 / 0.485 / 0.412 |
| B_argmax_matchE50 (같은 kept 수) | 1 / 4 / all | [미확인] | 0.502 / 0.553 / 0.476 | 〃 | [미확인] | 0.523 / 0.565 / 0.485 |
| oracle (B 복셀 위) | 1 / 4 / all | 0.719 / 0.834 / 0.861 | 〃 | 〃 | 0.745 / 0.837 / 0.848 | 〃 |

읽을 때 주의: 밴드 보간(whole→interp)만의 효과는 C_on_B N=1 +0.011, N=all
−0.003; C_grid N=1 +0.001, N=all −0.009. 즉 소유자가 걱정한 "부분 bin 버림
페널티"는 존재했지만 크기가 0.01 안팎이고 방향이 일정하지 않다. 격자 박스에서
GT를 뺀 효과(r2→r2_v3_whole, 둘 다 whole)는 C_grid N=1 −0.009. **결론 문장은
아직 쓰지 말 것** — 재학습 head의 결과(아래 4번)와 ray 단위 1차 지표까지
CRITERIA.md 규칙으로 판정한 뒤에 쓴다.

E vs B_argmax_matchE: 같은 kept 개수로 맞춰도 E가 B보다 낮다(r2 N=4:
0.473 vs 0.553). "적게 뽑아서 precision이 올랐다"는 반박은 성립하지 않고,
저신뢰 ray를 버리는 것 자체가 손해다.

ray 단위(1차 지표), 현재 head, test, `results/E21_posterior_rays/posterior_{r2,r8}_test.json`
(밴드 보간·온도 재적합 후 재실행본): r2 argmax MAE 0.246, 틀린 ray 26.5 %,
mode rescue@5 8.1 %, 틀렸을 때 ±0.2 m 질량 18.9 %, NLL 2.391, ECE 0.154;
r8 0.209 / 23.7 % / 6.8 % / 20.3 % / 2.301 / 0.163. fake Gaussian은 mode
rescue 0 %, NLL 최소 3.14(σ 0.4). CRITERIA.md 기준으로는 rescue "weak",
mass "weak", NLL "yes"(0.75 낮음), Spearman "yes"(−0.66).
GT-resize 변형(corner11, blockmean)으로 Spearman/AUSE 재계산: 차이 ≤0.002 /
0.0003 m → half-pixel 오프셋은 AUSE 주장에 영향 없음.

## 3. 다음 세션이 할 일 (순서대로)

1. 지금 결과 커밋: `git add results/E30_stage_d/{r2_v3,r8_v3,r2_v3_val,r8_v3_val,r2_v3_whole} src/stage_d_compare.py HANDOFF.md`
   → 커밋 "exp: Stage D v3 (pre-registered) ..." → push. v2 디렉터리는
   출처 불일치를 REPORT에 적은 뒤 삭제하거나 `_superseded`로 이름 변경.
2. 재학습 완료 확인(`history.json` 12 epoch, best.pth의 epoch·val KL 기록).
3. 재학습 head ray 평가:
   `python src/eval_posterior_rays.py --run posterior_r2_w4 --split test --gpu 1 --fit-temperature --batch-size 2`
   (r8_w4 동일). 결과 `results/E21_posterior_rays/posterior_r2_w4_test.json`.
4. 재학습 head Stage D v3: `python src/stage_d.py --run posterior_r2_w4 --mode r2 --gpu 1 --split test --tag _w4_v3`
   와 `--split val --tag _w4_v3` (r8 동일). 각 1시간. 온도는 3번 json에서 자동 로드.
5. 표: `python src/stage_d_compare.py --runs r2 r2_v3 r2_w4_v3 --val - r2_v3_val r2_w4_v3_val` (r8 동일).
6. CRITERIA.md 규칙으로만 판정해 REPORT.md에 "Stage D v3" 절 추가. 형식은
   0번 지시대로. v2 절(현재 REPORT.md 끝부분)은 출처 불일치를 명시하고
   v3로 대체됐다고 적는다. EXPERIMENT_STATUS.md E30–E43 갱신. 커밋·push.
7. 그 다음은 소유자 결정(Stage E occupancy / 단일 뷰 예측기 / 중단).

## 4. 아직 안 한 것 / 알려진 한계

- Stage B–D 모두 뷰 subset seed 0 하나(등간격). Stage A의 3 seed 편차 0.03이
  CRITERIA.md의 tie band 근거.
- 참조(GT 융합)가 N마다 달라짐(선택된 N개 뷰의 GT). REPORT v2 절에 공개됨.
- C_grid 격자 조대화: `grid_voxel` 열 > 0.1인 행(r2 32 %, r8 14 %)은 0.1 m
  지표·IoU 비교 불가. F1@0.2만 비교.
- Stage D oracle은 B 복셀 위에서만 계산(C_grid의 천장 아님).
- posterior head 초기화는 균등 시작(warm start 아님). `src/posterior_head.py` 주석 참조.
- 온도 fit 표본: 이제 val 전 프레임 × 1,000 ray (이전 48프레임 1씬).
- fb, r6 관측 세트는 Stage D 안 돌림.
- ZInD: 이 머신에 데이터 없음(렌더 서버 `/mnt/sdb/zind_raw`). EchoRecon과 무관.

## 5. 주요 파일

- `src/stage_d.py` (v3, `--split --band --hop --temperature --tag`), `src/select_frac.py`,
  `src/stage_d_compare.py`, `src/posterior.py`(ViewPosterior space/rounding),
  `src/eval_posterior_rays.py`, `src/train_posterior.py`, `src/posterior_head.py`,
  `src/test_fusion.py`(21 tests: `python src/test_fusion.py`).
- 환경: `/opt/conda/envs/shared_audio/bin/python`, `REPLICA_ROOT=/root/storage/replica_0422`,
  `R0422_SPLIT=off3`, 시퀀스 `/root/storage/replica_0422_forward_seq`.
- 예측: `outputs/pred/{r2,r8}/<scene>/seq_XXXX.npz` (test 3씬 + val 3씬, hop 160).
