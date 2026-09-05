# masteer 검토된 결론

현재 계획은 [PLAN_masteer](../plan/PLAN_masteer.md), 방법론은 [METHOD_V2](../METHOD_V2.md)를 따른다.
자동 수집 결과는 [AUTO_RESULTS_masteer](AUTO_RESULTS_masteer.md)이며,
**이 파일은 검토된 비교·판정의 기준이다.**

**프레이밍**: ma(A=2 이식 + teammate 토큰)와 steer(창의 길이가 곧 속도 명령)를 합친 트랙이다.
묻는 것은 협동 자체가 아니라 **명령 충실도** — "외부 스케줄러가 이 에이전트의 도착 시각을
명령한 만큼 옮길 수 있는가". 경로가 겹쳐도 시간이 어긋나면 안 만난다는 것이 가설이고,
그 반대 결과(정책이 속도 명령을 못 따른다)도 플래너 설계를 정하는 답이다.

<!-- AUTO_RESULTS -->

## 자동 결과 (최신순)

> 자동 생성 영역이다. 최신 결과가 위, 가장 오래된 결과가 아래다.
> 비교·채택·기각 해석은 이 블록 아래의 수동 기록에 남긴다.

| 완료 시각 | 상태 | tag | 자동 요약 | 큐 질문 |
|---|---|---|---|---|
| 2026-08-22 20:38 | 완료 | `ms14_mloL_s0` | MS_EVAL_SUMMARY tag=ms14_mloL_s0 rc=0 sr=0.4258 eps=2340 fin=0.4346 t50=344 colEp=0.2897 colStep=20.11 enter=0.44 still=173.63 path=9.62 pathA=9.72,9.47 lat=0.086 spd_err=0.461 v_real=0.488 off50=0.017 gait=0.865 both=0.1855 one=0.4473 indep=0.1889 vslope=0.380 vc=0.80,1.03,1.25 | **M=0(완전 정지)을 배우나.** `mult.clamp(min=MS_M_LO)` 가 0.25 라 정책이 정지 명령을 한 번도 본 적이 없다(최저 0.375 m/s). 보상에는 `stopped = v_tar<0.05` 분기가 이미 있는데 이 하한 때문에 안 돈다. `VEL_W=1` 로 맞춰 **`ms14_vw1L_s0` 이 그대로 대조군**이다. 되면 월드모델이 정지를 명령할 수 있고 시각화 9·10 이 해금된다 |
| 2026-08-22 20:35 | 완료 | `ms14_vw2L_s0` | MS_EVAL_SUMMARY tag=ms14_vw2L_s0 rc=0 sr=0.6069 eps=2326 fin=0.5890 t50=346 colEp=0.2889 colStep=17.47 enter=0.44 still=128.41 path=11.13 pathA=11.30,11.01 lat=0.097 spd_err=0.407 v_real=0.568 off50=0.011 gait=0.848 both=0.3711 one=0.4570 indep=0.3469 vslope=0.670 vc=0.58,0.84,1.13,1.32 | 1 과 5 의 기하 중간(√5≈2.2). 1 이 집기를 지키고 5 가 속도를 이기면 **중간에 최적점이 있는지**. `VEL_W` = 1·2·5 세 점이 된다 (5 는 `ms12_base4L_s0` 이 이미 돈다) |
| 2026-08-22 17:27 | 완료 | `ms13_sep20L_s0` | MS_EVAL_SUMMARY tag=ms13_sep20L_s0 rc=0 sr=0.6416 eps=2226 fin=0.6397 t50=334 colEp=0.0000 colStep=0.00 enter=0.00 still=140.28 path=11.06 pathA=11.24,10.96 lat=0.091 spd_err=0.371 v_real=0.566 off50=0.010 gait=0.848 both=0.4199 one=0.4238 indep=0.4092 vslope=0.783 vc=0.53,0.83,1.15,1.40 | sep 20 + 9000 iter. `ms12_base4L_s0`(sep 0) 와 3단 비교가 된다 |
| 2026-08-22 17:13 | 완료 | `ms13_sep9L_s0` | MS_EVAL_SUMMARY tag=ms13_sep9L_s0 rc=0 sr=0.6660 eps=2210 fin=0.6407 t50=346 colEp=0.0271 colStep=2.63 enter=0.06 still=143.51 path=11.88 pathA=11.89,11.88 lat=0.113 spd_err=0.401 v_real=0.590 off50=0.010 gait=0.854 both=0.4180 one=0.4316 indep=0.4105 vslope=0.761 vc=0.59,0.85,1.17,1.43 | sep 9 + 9000 iter. 속도 채널이 3000 에서 아직 점화 중이었다 |
| 2026-08-22 13:40 | 완료 | `ms15_we8L_s0` | MS_EVAL_SUMMARY tag=ms15_we8L_s0 rc=0 sr=0.6162 eps=2296 fin=0.6119 t50=351 colEp=0.2674 colStep=14.82 enter=0.44 still=136.96 path=11.84 pathA=11.82,11.90 lat=0.114 spd_err=0.421 v_real=0.606 off50=0.026 gait=0.867 both=0.3887 one=0.4004 indep=0.3745 vslope=0.723 vc=0.60,0.87,1.19,1.40 | **CLIP+ENDCLAMP 조합 -- 한 번도 안 해봤다.** `ms15_clip8L_s0` 과 `MS_ENDCLAMP` 하나만 달라 효과가 분리된다. CLIP 은 관측(창)을 끝에서 포화시키고 ENDCLAMP 은 보상 v_tar 를 0 으로 만드는 서로 다른 기제다. train.sh in-place 수정 사고로 유실된 `ms10_we` 도 일부 메운다. `ms14_mloL_s0`(M_LO=0)이 먹히면 끝처리 자체가 불필요해지므로 그 결과와 같이 본다 |
| 2026-08-22 13:23 | 완료 | `ms15_clip8L_s0` | MS_EVAL_SUMMARY tag=ms15_clip8L_s0 rc=0 sr=0.6665 eps=2272 fin=0.6329 t50=358 colEp=0.2632 colStep=13.21 enter=0.44 still=134.25 path=11.97 pathA=12.02,11.88 lat=0.139 spd_err=0.426 v_real=0.619 off50=0.018 gait=0.858 both=0.4551 one=0.4219 indep=0.4006 vslope=0.682 vc=0.61,0.87,1.16,1.37 | **현재 최고 설정의 장기판.** `ms11_clip_s0/s1`(3000)이 fin 0.746/0.733, both 0.604/0.600 으로 MRAND=4 계열(`ms12_base4` fin 0.684, both 0.504)을 앞서는데 **9000 판이 없다**. `ms6_m8_long_s0` 이 3000->6100 에서 vslope 0.076->0.348 로 4.5배 오른 전례가 있어 장기가 결정적일 수 있다. 최종 베이스 1순위 후보 |
| 2026-08-22 11:03 | 완료 | `ms14_vw1L_s1` | MS_EVAL_SUMMARY tag=ms14_vw1L_s1 rc=0 sr=0.6421 eps=2306 fin=0.6318 t50=344 colEp=0.2862 colStep=16.52 enter=0.47 still=144.93 path=11.87 pathA=11.73,11.96 lat=0.091 spd_err=0.452 v_real=0.606 off50=0.013 gait=0.887 both=0.4219 one=0.3789 indep=0.3992 vslope=0.519 vc=0.72,0.87,1.12,1.28 | **시드 잡음 재기.** `ms12_base4_s0`/`s1` 이 같은 설정인데 fin 0.684 vs 0.648, vslope 0.460 vs 0.413 으로 갈렸다 -- 시드 하나로는 VEL_W 판정이 안 된다. `ms14_vw1L_s0` 의 시드 짝 |
| 2026-08-22 10:42 | 완료 | `ms14_vw3L_s0` | MS_EVAL_SUMMARY tag=ms14_vw3L_s0 rc=0 sr=0.6025 eps=2256 fin=0.6033 t50=356 colEp=0.2775 colStep=13.88 enter=0.44 still=137.29 path=11.24 pathA=11.34,11.22 lat=0.088 spd_err=0.387 v_real=0.569 off50=0.011 gait=0.863 both=0.3574 one=0.4512 indep=0.3639 vslope=0.719 vc=0.55,0.83,1.13,1.35 | `VEL_W` 스윕을 1·2·**3**·5 로 채운다. 2 와 5 사이가 넓어 최적점을 놓칠 수 있다 (5 는 `ms12_base4L_s0` 이 이미 돈다) |
| 2026-08-22 10:41 | 완료 | `ms14_vw1L_s0` | MS_EVAL_SUMMARY tag=ms14_vw1L_s0 rc=0 sr=0.6504 eps=2330 fin=0.6348 t50=347 colEp=0.2918 colStep=17.04 enter=0.50 still=136.84 path=12.03 pathA=11.96,12.14 lat=0.086 spd_err=0.474 v_real=0.607 off50=0.020 gait=0.902 both=0.4062 one=0.4062 indep=0.4029 vslope=0.525 vc=0.71,0.90,1.14,1.28 | **빠져 있던 대조군.** 돌고 있는 `ms12_base4L_s0` 과 `MS_VEL_W` 한 글자만 다르다(5→1). `VEL_W=5` 가 속도항 비중을 50%→83% 로 올려 `handheld`(집기)를 12.5%→4.2% 로 눌렀다는 진단을 여기서 판정한다. `ms4_m4_s0`(VEL_W=1)은 CLIP 이 없어 이 비교가 불가능했다 |
| 2026-08-22 07:20 | 완료 | `ms12_base4L_s0` | MS_EVAL_SUMMARY tag=ms12_base4L_s0 rc=0 sr=0.6152 eps=2318 fin=0.6066 t50=349 colEp=0.2796 colStep=12.84 enter=0.44 still=131.29 path=11.22 pathA=11.26,11.20 lat=0.096 spd_err=0.393 v_real=0.585 off50=0.016 gait=0.848 both=0.3867 one=0.4375 indep=0.3679 vslope=0.705 vc=0.57,0.84,1.15,1.35 | 같은 설정 + 9000 iter. 속도 채널이 3000 에서 아직 점화 중이었다 |
| 2026-08-21 23:46 | 실패 | `ms6_m8_long_s0` | training script rc=2 | **속도 곡선 장기.** 3000 에서 기울기 0.076. A=1 은 같은 창 코드로 0.45~0.49 를 냈다 |
| 2026-08-21 21:04 | 완료 | `ms11_clip_s0` | MS_EVAL_SUMMARY tag=ms11_clip_s0 rc=0 sr=0.7695 eps=2324 fin=0.7457 t50=338 colEp=0.2883 colStep=16.46 enter=0.48 still=168.13 path=11.99 pathA=11.94,12.09 lat=0.169 spd_err=0.589 v_real=0.630 off50=0.043 gait=0.991 both=0.6035 one=0.2793 indep=0.5561 vslope=0.360 vc=0.95,1.02,1.21,1.34 | **끝 클램프.** 호길이·조준점·창을 실제 경로 끝에서 포화. TokenHSI traj 의 `clip(t/dur,0,1)` 을 거리 버전으로. 지금까지 최선(속도항 ×5)과 함께 |
| 2026-08-21 20:54 | 완료 | `ms11_clip_s1` | MS_EVAL_SUMMARY tag=ms11_clip_s1 rc=0 sr=0.7559 eps=2294 fin=0.7328 t50=350 colEp=0.2877 colStep=15.91 enter=0.45 still=161.42 path=12.76 pathA=12.69,12.83 lat=0.168 spd_err=0.589 v_real=0.647 off50=0.049 gait=0.987 both=0.5996 one=0.2812 indep=0.5370 vslope=0.351 vc=0.94,1.03,1.18,1.32 | 끝 클램프 시드 1 |
| 2026-08-21 20:48 | 완료 | `ms12_base4_s0` | MS_EVAL_SUMMARY tag=ms12_base4_s0 rc=0 sr=0.7021 eps=2326 fin=0.6840 t50=336 colEp=0.2855 colStep=15.93 enter=0.45 still=158.24 path=11.99 pathA=12.25,11.69 lat=0.145 spd_err=0.553 v_real=0.608 off50=0.045 gait=0.940 both=0.5039 one=0.3301 indep=0.4679 vslope=0.460 vc=0.84,0.95,1.19,1.33 | **일반 베이스 후보.** 휜 v3 경로(free) + 4구간 속도 명령 + 속도항 ×5 + 끝 클램프. m4 는 lat 0.114 / vslope 0.252 로 m8 보다 좋은데 knob 을 안 얹어봤다 |
| 2026-08-21 20:03 | 완료 | `ms12_base4_s1` | MS_EVAL_SUMMARY tag=ms12_base4_s1 rc=0 sr=0.6787 eps=2378 fin=0.6476 t50=342 colEp=0.2851 colStep=16.40 enter=0.47 still=150.99 path=11.87 pathA=11.73,12.03 lat=0.181 spd_err=0.557 v_real=0.628 off50=0.062 gait=0.951 both=0.4688 one=0.3535 indep=0.4194 vslope=0.413 vc=0.88,1.00,1.19,1.34 | 일반 베이스 시드 1 |
| 2026-08-21 17:26 | 실패 | `ms10_we_s1` | training script rc=2 | 합침 시드 1 |
| 2026-08-21 17:18 | 실패 | `ms10_we_s0` | training script rc=2 | **두 knob 합침.** 속도항 ×5 + 패딩꼬리 차단. 각각 vslope 0.105→0.154 / 0.158 이고 독립이라 더 갈 것 |
| 2026-08-21 14:16 | 실패 | `ms9_velw10_s0` | training script rc=2 | 속도항 ×10. 5 에서 0.154 였다 — 더 키우면 계속 오르나 포화하나 |
| 2026-08-21 14:03 | 실패 | `ms9_velw5_s1` | training script rc=2 | **속도항 확인.** velw5 의 vslope +0.049 가 시드 산포(0.019)의 2.5배라 재현 확인이 필요하다 |
| 2026-08-21 09:08 | 완료 | `ms5_pin35_s0` | MS_EVAL_SUMMARY tag=ms5_pin35_s0 rc=0 sr=0.5518 eps=2630 fin=0.5240 t50=337 colEp=0.2753 colStep=17.16 enter=0.44 still=176.08 path=9.52 pathA=9.43,9.59 lat=0.098 spd_err=0.622 v_real=0.549 off50=0.038 gait=0.992 vslope=0.125 vc=1.03,1.06,1.12,1.17 | **pin_walk.** 0.5→0.35. 운반 중 죽어 있던 루트 속도항을 되살린다 |
| 2026-08-21 09:07 | 완료 | `ms5_endc_s0` | MS_EVAL_SUMMARY tag=ms5_endc_s0 rc=0 sr=0.6772 eps=2320 fin=0.6685 t50=344 colEp=0.2905 colStep=18.12 enter=0.46 still=188.98 path=10.95 pathA=11.12,10.87 lat=0.131 spd_err=0.625 v_real=0.547 off50=0.060 gait=0.990 vslope=0.158 vc=1.02,1.06,1.13,1.19 | **패딩 꼬리.** 경로 끝 이후 M=0. 목표에 서 있어야 할 때 전진 명령하는 오염 제거 |
| 2026-08-21 09:05 | 완료 | `ms7_par20_s0` | MS_EVAL_SUMMARY tag=ms7_par20_s0 rc=0 sr=0.6797 eps=2140 fin=0.8065 t50=293 colEp=0.0009 colStep=0.01 enter=0.00 still=305.86 path=12.78 pathA=12.81,12.75 lat=0.077 spd_err=0.937 v_real=0.588 off50=0.000 gait=1.209 dTw=+0.000 wpass=0.480 encd=1.424 | 간격 2.0 m. 접촉이 사라지는 지점을 찾는다 |
| 2026-08-21 08:54 | 완료 | `ms7_par15_s0` | MS_EVAL_SUMMARY tag=ms7_par15_s0 rc=0 sr=0.7905 eps=2132 fin=0.8565 t50=294 colEp=0.0019 colStep=0.01 enter=0.00 still=294.33 path=12.74 pathA=12.75,12.73 lat=0.085 spd_err=0.939 v_real=0.587 off50=0.000 gait=1.210 dTw=+0.000 wpass=0.621 encd=0.900 | **반응 한계 위쪽.** 1.0 m 에서 colEp 0.64, 0.6 m 는 0.9955 로 한계가 1.0 위다 |
| 2026-08-21 05:53 | 완료 | `ms5_velw5_s0` | MS_EVAL_SUMMARY tag=ms5_velw5_s0 rc=0 sr=0.8374 eps=2314 fin=0.8133 t50=317 colEp=0.2766 colStep=14.15 enter=0.45 still=189.89 path=12.24 pathA=12.25,12.23 lat=0.188 spd_err=0.701 v_real=0.623 off50=0.076 gait=1.089 vslope=0.154 vc=1.18,1.19,1.27,1.34 | **저울 ③ 반대방향.** 속도항 ×5. `lat` 을 안 해치고 되는지 |
| 2026-08-21 05:52 | 완료 | `ms5_velk15_s0` | MS_EVAL_SUMMARY tag=ms5_velk15_s0 rc=0 sr=0.6919 eps=2360 fin=0.6801 t50=344 colEp=0.3025 colStep=20.70 enter=0.50 still=175.29 path=11.34 pathA=11.46,11.22 lat=0.127 spd_err=0.631 v_real=0.595 off50=0.040 gait=1.014 vslope=0.062 vc=1.08,1.09,1.12,1.15 | **포화.** 커널 k 5→1.5. 명령 1.5·실제 0.58 이 2.9σ 밖이라 평지인 문제 |
| 2026-08-21 05:51 | 완료 | `ms5_c05_s1` | MS_EVAL_SUMMARY tag=ms5_c05_s1 rc=0 sr=0.8496 eps=2272 fin=0.8217 t50=335 colEp=0.2887 colStep=17.67 enter=0.48 still=178.22 path=12.70 pathA=12.71,12.68 lat=0.211 spd_err=0.695 v_real=0.592 off50=0.096 gait=1.054 vslope=0.103 vc=1.14,1.17,1.21,1.25 | 저울 ① 시드 1 |
| 2026-08-21 05:49 | 완료 | `ms5_c00_s0` | MS_EVAL_SUMMARY tag=ms5_c00_s0 rc=0 sr=0.8735 eps=2258 fin=0.8574 t50=317 colEp=0.3127 colStep=16.33 enter=0.50 still=200.15 path=12.40 pathA=12.21,12.65 lat=0.318 spd_err=0.784 v_real=0.578 off50=0.251 gait=1.104 vslope=0.040 vc=1.30,1.31,1.34,1.34 | **저울 ② 상한.** 이탈 벌점 0. 벌점이 없으면 속도가 어디까지 가나 |
| 2026-08-21 05:40 | 완료 | `ms5_c05_s0` | MS_EVAL_SUMMARY tag=ms5_c05_s0 rc=0 sr=0.8208 eps=2362 fin=0.7811 t50=322 colEp=0.2997 colStep=17.88 enter=0.45 still=175.39 path=11.93 pathA=11.98,11.86 lat=0.216 spd_err=0.696 v_real=0.597 off50=0.098 gait=1.064 vslope=0.084 vc=1.16,1.18,1.23,1.25 | **저울 ①.** 이탈 벌점 2.0→0.5 (속도 대비 10:1 → 2.5:1). 기울기가 오르고 `lat` 은 나빠져야 한다 |
| 2026-08-21 05:39 | 완료 | `ms5_both_s0` | MS_EVAL_SUMMARY tag=ms5_both_s0 rc=0 sr=0.8438 eps=2352 fin=0.8078 t50=312 colEp=0.3036 colStep=15.46 enter=0.47 still=190.32 path=12.02 pathA=12.05,11.93 lat=0.247 spd_err=0.735 v_real=0.625 off50=0.138 gait=1.102 vslope=0.181 vc=1.23,1.23,1.33,1.42 | 저울 양쪽 동시. 하나로 부족할 때 |
| 2026-08-21 00:53 | 완료 | `ms4_cross_s1` | MS_EVAL_SUMMARY tag=ms4_cross_s1 rc=0 sr=0.5713 eps=2186 fin=0.7434 t50=297 colEp=0.1500 colStep=1.99 enter=0.17 still=305.76 path=12.70 pathA=12.76,12.62 lat=0.072 spd_err=0.954 v_real=0.587 off50=0.004 gait=1.173 vslope=-0.267 vc=1.42,1.42,1.22 dTw=-0.233 wpass=0.357 beta=0.007 encd=0.479 | 핵심 시드 1 |
| 2026-08-21 00:48 | 완료 | `ms4_solo_s0` | MS_EVAL_SUMMARY tag=ms4_solo_s0 rc=0 sr=0.7495 eps=2178 fin=0.8269 t50=300 colEp=0.0000 colStep=0.00 enter=0.00 still=278.53 path=12.83 pathA=12.82,12.83 lat=0.074 spd_err=0.931 v_real=0.589 off50=0.001 gait=1.174 vslope=0.333 vc=1.00,1.15,1.25 dTw=+0.667 wpass=0.588 beta=0.460 encd=nan | **관문.** 짝 9 m 밖 + 같은 감속 명령. `β̂<0.5` 면 교차 실험을 하지 않는다 |
| 2026-08-21 00:47 | 완료 | `ms4_m4_s0` | MS_EVAL_SUMMARY tag=ms4_m4_s0 rc=0 sr=0.6777 eps=2300 fin=0.6839 t50=347 colEp=0.2878 colStep=18.42 enter=0.47 still=179.16 path=11.48 pathA=11.46,11.51 lat=0.114 spd_err=0.587 v_real=0.590 off50=0.035 gait=0.968 vslope=0.252 vc=0.95,1.00,1.14,1.22 | **속도 명령.** 경로 4구간, M ∈ {1.0,.75,.5,.25} |
| 2026-08-21 00:47 | 완료 | `ms4_par06_s0` | MS_EVAL_SUMMARY tag=ms4_par06_s0 rc=0 sr=0.5532 eps=2242 fin=0.7168 t50=296 colEp=0.9955 colStep=214.70 enter=3.30 still=295.46 path=12.77 pathA=12.77,12.78 lat=0.064 spd_err=0.942 v_real=0.586 off50=0.004 gait=1.186 dTw=+0.033 wpass=0.383 encd=0.760 | 나란히 0.6 m. 몸통 폭 0.456 m 라 여유 0.144 m |
| 2026-08-21 00:46 | 완료 | `ms4_cross_s0` | MS_EVAL_SUMMARY tag=ms4_cross_s0 rc=0 sr=0.5698 eps=2234 fin=0.7198 t50=304 colEp=0.1235 colStep=2.79 enter=0.18 still=291.52 path=12.75 pathA=12.81,12.70 lat=0.069 spd_err=0.942 v_real=0.589 off50=0.003 gait=1.159 vslope=0.285 vc=1.02,1.12,1.23 dTw=+0.700 wpass=0.338 beta=0.327 encd=0.887 | **핵심.** 직각 교차, 지연 랜덤. `ΔT_w` 가 명령에 비례해야 한다 |
| 2026-08-21 00:45 | 완료 | `ms4_reg_s0` | MS_EVAL_SUMMARY tag=ms4_reg_s0 rc=0 sr=0.7344 eps=2472 fin=0.7193 t50=305 colEp=0.2913 colStep=17.01 enter=0.46 still=207.35 path=11.18 pathA=11.32,11.04 lat=0.115 spd_err=1.018 v_real=0.637 off50=0.051 gait=1.189 | **회귀 (관문).** `MS_MRAND=0` 고정 M. ma 기준선 0.8604 근처가 나와야 병합이 검증된다 |
| 2026-08-21 00:45 | 완료 | `ms4_reg_s1` | MS_EVAL_SUMMARY tag=ms4_reg_s1 rc=0 sr=0.7446 eps=2494 fin=0.7185 t50=299 colEp=0.3055 colStep=16.54 enter=0.49 still=204.21 path=11.13 pathA=11.09,11.20 lat=0.130 spd_err=1.029 v_real=0.639 off50=0.059 gait=1.192 | 회귀 시드 1 |
| 2026-08-21 00:45 | 완료 | `ms4_par10_s0` | MS_EVAL_SUMMARY tag=ms4_par10_s0 rc=0 sr=0.6479 eps=2216 fin=0.7802 t50=287 colEp=0.6426 colStep=17.64 enter=1.26 still=295.17 path=12.78 pathA=12.78,12.77 lat=0.092 spd_err=0.941 v_real=0.589 off50=0.001 gait=1.226 dTw=+0.000 wpass=0.467 encd=0.477 | **반응 한계.** 나란히 1.0 m, 둘 다 평속 |
| 2026-08-20 21:05 | 완료 | `ms4_m8_s0` | MS_EVAL_SUMMARY tag=ms4_m8_s0 rc=0 sr=0.6816 eps=2390 fin=0.6594 t50=338 colEp=0.3038 colStep=18.96 enter=0.50 still=187.24 path=10.71 pathA=10.74,10.67 lat=0.120 spd_err=0.655 v_real=0.562 off50=0.049 vslope=0.057 vc=0.56,0.58,0.59,0.62 | 8구간. 전환이 잦다 — 명령 갱신 주기의 한계 |
| 2026-08-20 21:02 | 완료 | `ms4_zero_s0` | MS_EVAL_SUMMARY tag=ms4_zero_s0 rc=0 sr=0.3545 eps=2296 fin=0.3802 t50=262 colEp=0.2221 colStep=14.47 enter=0.34 still=266.34 path=8.48 pathA=8.48,8.47 lat=0.251 spd_err=0.830 v_real=0.431 off50=0.128 | **대조군.** 창을 0 으로. 이제 teammate 토큰은 살아 있다 |
| 2026-08-20 20:58 | 완료 | `ms4_m4_s1` | MS_EVAL_SUMMARY tag=ms4_m4_s1 rc=0 sr=0.6304 eps=2382 fin=0.6201 t50=358 colEp=0.2771 colStep=19.55 enter=0.49 still=170.65 path=11.15 pathA=11.16,11.10 lat=0.125 spd_err=0.574 v_real=0.594 off50=0.048 | 속도 명령 시드 1 |
| 2026-08-20 16:28 | 완료 | `ms1_reg_s1` | MS_EVAL_SUMMARY tag=ms1_reg_s1 rc=0 sr=0.4956 eps=2306 fin=0.5221 t50=231 colEp=0.2151 colStep=11.90 enter=0.33 still=290.07 path=9.01 pathA=9.12,8.87 | 회귀 시드 1 |
| 2026-08-20 16:22 | 완료 | `ms1_m4_s0` | MS_EVAL_SUMMARY tag=ms1_m4_s0 rc=0 sr=0.4277 eps=2288 fin=0.4515 t50=250 colEp=0.2351 colStep=14.82 enter=0.38 still=287.83 path=8.61 pathA=8.58,8.66 | **속도 명령.** 경로 4구간, M ∈ {1.0,.75,.5,.25}. `spd_err` 이 내려가야 한다 |
| 2026-08-20 15:04 | 완료 | `ms1_reg_s0` | MS_EVAL_SUMMARY tag=ms1_reg_s0 rc=0 sr=0.4526 eps=2374 fin=0.4520 t50=227 colEp=0.2519 colStep=12.49 enter=0.39 still=284.60 path=8.40 pathA=8.40,8.38 | **회귀.** `MS_MRAND=0` 이면 M 고정 = 원본 carry 와 수학적으로 같다. ma 기준선 0.8604(sep0) 근처가 나와야 한다 |
| 2026-08-20 14:47 | 완료 | `ms1_zero_s0` | MS_EVAL_SUMMARY tag=ms1_zero_s0 rc=0 sr=0.4019 eps=2384 fin=0.4299 t50=269 colEp=0.2416 colStep=16.27 enter=0.36 still=263.50 path=8.36 pathA=8.41,8.33 | **대조군.** 창을 0 으로. 명령 없이 얼마나 하나 |

<!-- /AUTO_RESULTS -->

## 해석

### 0. 순차 stacking release의 pickup 회귀 추적 (2026-09-04)

목표는 모델 구조를 바꾸지 않고, 중앙 controller가 base 배치 뒤 가상 박스로 후퇴를
지시하는 것이다. CLEAR의 base도 object-free loco AMP가 아니라 부모 carry AMP를 받는다.

실험 경과:

- `ms25_virtualretreat_adapteronly_r50_s0_e1024`: adapter-only, 가상 박스, 50% carry
  rehearsal. epoch 9100 viewer에서 pickup이 깨져 기각했다.
- 이후 CLEAR의 loco AMP override를 제거하고 carry AMP로 복원했다. 가상 박스는 유지했다.
- `ms27_virtualbox_ms20clear_b3_tokunfreeze_s0`: ms18 epoch 9000에서 steer/new-carry
  tokenizer와 adapter를 학습했다. CLEAR steering 추가항은 0이지만 transition bonus 3.0,
  CLEAR 진입 bonus 1.5가 남아 있었다. epoch 9300에서도 pickup이 깨져 중단했다.
  epoch 9200은 ms18 대비 new-carry tokenizer L2 0.358, adapter L2 1.580이 변했다.
  따라서 head 동결만으로는 pickup action을 보존하지 못한다.
- `ms28_virtualbox_ms20bonus0_stage_s0`: 확인된 `ms20_stackrel_gate_bonus0_s0`에 맞춰
  transition/CLEAR 진입 bonus를 모두 0으로 내렸다. 가상 박스, carry AMP, 50% rehearsal,
  tokenizer unfreeze, ms20 CLEAR 식은 유지한다. ms18 epoch 9000에서 GPU 7, 1024 env,
  3000 iteration으로 실행 중이다.

단계 원인을 직접 가르기 위해 episode 지표에 양손 OBB 접근, 10-frame pickup, 운반 중
break, base 배치, 손 떼기, body clear, 물리 실패, 최대 lift와 역할을 추가했다. 학습 중
448 iteration까지 누적된 **비정상(non-stationary) 예비치**는 base pickup 0.856,
top pickup 0.850이었다. base의 최초 정지 위치는 접근 0.136, lift 0.009,
carry/place 0.717, release 0.057, clear 0.039, body-clear 통과 0.044였다. 이는 고정 정책
평가가 아니므로 최종 비교값으로 사용하지 않고 실패 후보를 고르는 용도로만 쓴다.

서버 `common_agent.py`에는 `save_archive_frequency` 구현이 빠져 `Humanoid.pth`만
덮어쓰는 것도 확인했다. 실행을 건드리지 않는 `checkpoint_watch.py`를 붙여 epoch 9450부터
50 epoch 간격 번호 checkpoint를 보존한다. 고정 비교는 아래 두 명령의 동일 sidecar 평가로
한다.

```bash
bash scripts/masteer/stack_stage_eval.sh ms28_virtualbox_ms20bonus0_stage_s0 7 initial 512
bash scripts/masteer/stack_stage_eval.sh ms28_virtualbox_ms20bonus0_stage_s0 7 9450 512
```

현재 판정은 **진행 중**이다. `initial` 대비 어느 checkpoint에서 pickup 또는 carry/place가
먼저 하락하는지 확인하기 전에는 bonus 0의 보존 성공 여부를 결론 내리지 않는다.

### 1. extra 토크나이저가 한 번도 학습된 적이 없었다 (2026-08-20)

`MA_TOKENIZER_ZERO=1` 이 마지막 Linear 를 0 으로 두는데 그 뒤에 ReLU 가 하나 더 있어서
`ReLU'(0)=0` 으로 gradient 가 끊겼다. **ma·masteer 의 모든 런에서 teammate·steer 토큰이
초기값 그대로였다** (`MS_GRADCHK` 로 `|grad| = 0` 실측, 체크포인트 가중치 max 가
`1/sqrt(fan_in)` 과 소수점까지 일치).

무효가 된 결론:

    masteer  zero 0.4019 / reg 0.4526 / m4 0.4277 이 같았던 것 -- 세 조건이 같은 네트워크였다
    ma       "teammate 토큰 효과 없음 (Δ=+0.004)" -- 토큰이 두 조건 모두에서 0 이었다

수정 후 같은 조건: `reg 0.4526 -> 0.7344/0.7446` (**+0.28**, 시드 산포 0.010).

### 2. 회귀 격차는 전부 이탈 벌점이었다

`ms4_reg`(0.719) 와 ma 기준선(0.847) 의 격차 0.13 을 "경로 추종 제약" 으로 봤는데,
`MS_POS_C` 를 낮추면 그대로 회복된다. 병합 자체에는 손실이 없다.

| `POS_C` | `fin` | `lat` | `off50` | `vslope` |
|---|---|---|---|---|
| 0 | **0.857** | 0.318 | 25.1% | 0.040 |
| 0.5 | 0.781 / 0.822 | 0.216 | 9.8% | 0.084 / 0.103 |
| 2.0 (기본) | 0.652 | **0.120** | 4.4% | 0.105 |
| ma 기준선 | 0.847 | — | — | — |

단조 trade-off 다. **경로 정확도를 사고 성공률을 판다.** ma 는 경로 제약이 없는 다른
태스크이므로 "넘었다" 는 조심해서 읽는다.

### 3. 속도 명령 추종 -- 작동하는 knob 은 속도항 크기 하나뿐이다

| 조작 | `vslope` | 판정 |
|---|---|---|
| 기준선 (`VEL_W=1, VEL_K=5, POS_C=2`) | 0.105 | — |
| 커널 넓힘 `VEL_K=1.5` | **0.062** | ✗ 반박. 넓히면 속도 간 구별이 흐려진다 |
| 벌점 낮춤 `POS_C=0.5` | 0.084 / 0.103 | ✗ 효과 없음 |
| 벌점 0 `POS_C=0` | **0.040** | ✗ 악화. 벌점이 없으면 그냥 빨리 달린다 |
| **속도항 `VEL_W=5`** | **0.154** | ✓ **유일하게 작동** |
| 속도항 + 벌점 낮춤 | **0.181** | ✓ 약간 더 |

`velw5` 가 `c05` 를 **세 축 모두에서** 이긴다 (`fin` 0.813 vs 0.822 은 시드 산포 안,
`lat` 0.188 vs 0.211, `vslope` 0.154 vs 0.103). 음의 항을 없애는 것보다
**양의 항을 키우는 것**이 경로 정확도를 덜 잃는다.

### 4. 타이밍 조율이 작동한다 -- 대조군 5종 통과

같은 정책(`ms4_cross_s0`)에 명령만 바꿔 평가:

| 조건 | 도착 시간차 | `encd` | `colEp` | `latw` |
|---|---|---|---|---|
| `swap` a0 에 감속 | +0.200 | 0.347 | 32.3% | 0.057 |
| `plac` 교차점 **뒤**에 감속 | +0.200 | 0.496 | 19.9% | 0.046 |
| `sync` 명령 없음 | +0.500 | 0.465 | 21.3% | 0.052 |
| `both` 둘 다 감속 | +1.000 | 0.954 | 8.6% | 0.052 |
| `off2` a1 에 감속 | +1.267 | **1.207** | 6.9% | **0.048** |

    encd ~ 도착 시간차     기울기 +0.835   R² 0.98
    encd ~ 느려진 정도     기울기 +1.386   R² 0.39
    기하 예측: 직각 교차 1.5 m/s 에서 최근접 = 1.06 · Δ

세 대안 설명이 각각 죽는다. **위약**(교차점 뒤 감속)은 효과가 없어 "조심스럽게 걸어서"를
배제하고, **양쪽 감속**은 더 느린데 덜 안전해서 "느려서"를 배제한다. 그리고 대가가 없다 --
성공률 0.7173 -> 0.7165, `lat` 0.069 -> 0.068, `latw` 는 오히려 감소.

`latw` 가 안 오른 것이 핵심이다. **여유를 옆으로 비켜서 번 게 아니라 늦게 와서 벌었다.**

### 5. 열려 있는 문제

**역할을 외웠다.** 학습 내내 `MS_DECEL=a1` 로 고정했더니 정책이 위치 규칙으로 외웠다.
같은 명령을 a0 에 주면 창속도가 `1.500 -> 1.324` 밖에 안 내려가고(a1 은 `1.233 -> 0.957`)
기울기가 `-0.158` 로 뒤집힌다. **크기는 창에서 읽지만 역할은 못 읽는다.**
`MS_DECEL=rand` 를 넣고 `ms8_*`·`ms9_crossw_*` 로 검증 중이다.

**시드 하나가 완전히 실패.** `cross_s1` β̂ 0.007, a1 이 오히려 더 빠르다. 2 중 1.

**반응 한계가 1.0 m 위.** 간격 1.0 m 에서 `colEp` 0.64, 0.6 m 에서 0.9955.
공간만으로는 안 된다 -- 그래서 타이밍 결과가 더 의미 있다. `ms7_par15/20` 이 상한을 찾는다.

**창 통과율 0.32~0.59.** 타이밍 결과가 쌍의 3 분의 1 위에 서 있어 선택 편향 위험이 있다.

### 6. 지표를 세 번 고쳤다

이 트랙에서 **잘못된 지표로 두 번 잘못된 결론을 냈다.** 기록해둔다.

1. `v_real` 이 정지 스텝(32%)까지 분모에 넣어 보행속도를 0.58 로 표시했다. 실제 1.02 이고
   참조 모션 중앙값 0.932 와 일치한다. 이 값으로 "정책이 참조보다 느리다, AMP 가 원인" 이라는
   틀린 진단을 냈다.
2. 명령 속도별 구간도 같은 문제라 `vslope` 가 0.057 로 나왔다. 정지 제외하면 0.105 다.
   이 값으로 "속도 명령을 거의 무시한다" 고 보고했다.
3. 평가 요약 블록이 f-string 따옴표 누락으로 SyntaxError 였는데 `>> "$LOG"` 가 stdout 만
   받아서 **조용히 죽었다.** 평가 3 건이 요약 없이 끝났다. stderr 도 로그로 보내게 고쳤다.

교훈: **지표를 바꾸면 같은 자로 다시 재기 전에는 조건 간 비교를 하지 않는다.**
