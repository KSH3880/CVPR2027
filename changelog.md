# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-05

### penalty_v2 12500 / fusion_v1 7500 / fusion_v2 5000 중간 저장본 평가

- GPU3·4·5에서 각 512환경×3seed×1200step Plane Ever 정식 평가, GPU6에서 같은 저장본 각각 64환경×seed0 정지 진단을 수행했다. random_scenario/near50 유지, viewer 거리 지정 해제, GPU0·1 미사용. 기존 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 SIT53.79/98.77·CLIMB43.09/92.88·ON_TOP32.76/91.33·전체39.52/93.25, fusion_v1 56.05/97.06·54.84/94.74·33.41/94.57·전체43.09/95.11, fusion_v2 49.72/94.48·42.51/91.55·22.86/94.02·전체32.96/93.56. 단독5행동 평균은 각각94.04/96.89/95.70%다.
- 중간 저장본을 기존 이력에 삽입해 penalty10→12.5→15k 전체 후행33.47→39.52→44.70%, 선행91.34→93.25→93.47%를 확인했다. 두 구간 모두 전체 후행이3개 평가 seed에서 공통 상승했다. fusion_v1 5→7.5→10k는40.58/95.87→43.09/95.11→40.66/95.99로 등락했고, v2 3→5→7.5k는31.67/93.38→32.96/93.56→35.98/95.33이며 CLIMB는 유지·최근 ON_TOP은3seed 공통 상승했다.
- 같은5k/7.5k에서 fusion_v2 전체 후행은 v1보다 각각7.62/7.11%p 낮고 모두3seed 공통 차이다. 7.5k 선행은v1 95.11/v2 95.33%로 비슷하므로 후행 격차를 선행 성공률만으로 설명하지 않는다. 모델 외에 대기 반경·ON_TOP 진입 기준도 달라 개별 변경의 인과 효과를 단정하지 않는다.
- armed·AT Ever 전 root XYZ≤.3m/s·30프레임 정지의 초기거리 감소율 중앙값(SIT/CLIMB/ON_TOP)은 penalty43.5/67.0/2.2%, fusion_v1 12.4/4.2/1.5%, fusion_v2 15.6/1.5/6.3%; 출발점 대비XY 변위 중앙값은1.08/1.76/.44m, .40/.43/.37m, .46/.51/.40m다. 정지/관찰가능 쌍은10/10·9/9·22/27, 9/9·8/8·23/23, 9/11·7/8·17/17이다. penalty12.5k와15k 모두 정지한 CLIMB8쌍의 목표거리 중앙값은 .34→1.89m, 감소율75.4→19.4%로 변했다.64환경·조건부 진단이며 전신/지속 정지를 뜻하지 않는다.
- 정식9회·진단3회 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·native 진입/AT Ever 해제·동결 encoder/RMS·원본 checkpoint·운영파일262개·기존3학습 보존 확인. 이전 평가와 운영파일 해시도 동일하다. [전체 성공률 이력](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/before_comparison.csv), [구간/동일 epoch 비교](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/comparison.json), [정규화 정지 이력](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/stop_comparison.csv), [공통 정지 쌍](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/stop_common_pairs.csv), [통합 검증](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/combined_verification.json).

### penalty_v2 15000 / fusion_v1 10000 / fusion_v2 7500 최신 평가

- GPU3·4·5에서 시작 시 고정한 세 checkpoint를 각각 512환경×3seed×1200step Plane Ever로 평가하고, GPU6에서 같은 저장본을 각각 64환경×seed0로 정지 진단했다. random_scenario/near50 유지, viewer 거리 지정 해제, GPU0·1 미사용. 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 SIT59.64/97.39·CLIMB48.93/91.66·ON_TOP37.54/92.79·전체44.70/93.47, fusion_v1 52.13/96.57·53.95/94.09·30.93/96.54·전체40.66/95.99, fusion_v2 46.84/96.60·42.49/93.92·29.23/95.48·전체35.98/95.33. 단독5행동 평균은 각각96.99/95.81/96.50%다.
- 직전10k/5k/3k 대비 전체 후행 변화는 +11.23/+0.08/+4.31%p. penalty 세 행동과 fusion_v2 SIT·ON_TOP은3seed 공통 상승했고, fusion_v1 SIT는3seed 공통 하락했다. 서로 다른 epoch의 비교이며 중간 추세를 보간하지 않는다.
- 첫 armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 정지의 사람→목표XY 중앙값(m)은 penalty1.07/1.89/2.51, fusion_v1 1.90/2.11/2.81, fusion_v2 1.49/2.01/1.93(SIT/CLIMB/ON_TOP). 정지/관찰가능 쌍은 각각8/9·8/8·24/28, 9/9·6/7·23/24, 8/9·6/6·15/16이다.
- 쌍별 초기거리 대비 감소율 `100*(1-d_stop/d_initial)` 중앙값은 penalty32.3/19.4/4.6%, fusion_v1 14.0/−5.4/1.5%, fusion_v2 17.8/−3.3/5.0%; 실제 출발점 대비XY 변위 중앙값은 .63/.64/.47m, .40/.35/.33m, .47/.42/.44m다. 양 checkpoint 모두 정지한 같은 쌍 비교도 별도로 저장했다. root 정지는 전신 정지나 선행 완료까지 지속 대기를 뜻하지 않는다.
- 정식9회·진단3회 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 일치·native 진입/AT Ever 해제·동결 encoder/RMS·checkpoint/운영파일262개·기존3학습 보존을 확인했다. [성공률 비교](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/before_comparison.csv), [정규화 정지 비교](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/stop_comparison.csv), [공통 정지 쌍](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/stop_common_pairs.csv), [통합 검증](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/combined_verification.json).

### fusion_v2 3000 평가 및 세 모델 64환경 정지 거리 추세

- GPU3에서 고정 fusion_v2 3000을512환경×3seed×1200step Plane Ever로 평가하고, GPU4에서 같은 저장본64환경×seed0 정지 진단을 수행했다. 새 viewer 거리 옵션을 명시적으로 해제하고 기존 random_scenario/near50·평가 규약을 유지했다. 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%는 SIT39.97/96.53·CLIMB42.84/90.07·ON_TOP23.95/93.59·전체31.67/93.38. 전체는2k27.33/92.69에서+4.35%p, 단독5행동 평균95.97→96.04. CLIMB/ON_TOP/전체는3seed 공통 상승, SIT 변화는 혼재했다. 같은3k fusion_v1 전체34.40/94.03보다3seed 공통 낮으며 penalty_v2 전체23.16/84.91보다3seed 공통 높다.
- 첫 root XYZ≤.3m/s·1초 연속 정지의 사람→목표XY 중앙값 SIT1.42m(9/10)·CLIMB1.80m(7/7)·ON_TOP1.90m(18/20), ON_TOP payload1.63m. 2k1.42/1.84/1.88m에서 대체로 유지됐다. 출발지 초기XY .5m 이내 유지16/72쌍, 그중 성공 후3초 관찰가능16쌍 중14쌍이 .5m 이상 이동했다.
- penalty_v2 6.5/8.5/9/10k·fusion_v1 1.5/3/4/5k·fusion_v2 .5/1/2/3k의64환경 정지 기록을 집계했다. 12개 기록의 초기 상태 정확 일치, root 속력/시간·native 장면 종료 규약을 확인하고32환경 기록은 제외했다. 같은 초기 장면이어도 정지/관찰가능 부분집합은 checkpoint별로 달라지며 정지 위치만으로 접근 후 대기라고 해석하지 않는다.
- 정식3회/진단1회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 일치·native human 진입/AT Ever 해제·동결 encoder/RMS·checkpoint/운영파일262개·기존3학습 보존 확인. [성공률 비교](output_etc/check/fusion_v2_3000_eval_stop_20261105/before_comparison.csv), [정지 추세](output_etc/check/fusion_v2_3000_eval_stop_20261105/stop_trends.csv), [그래프](output_etc/check/fusion_v2_3000_eval_stop_20261105/stop_trends.png), [검증](output_etc/check/fusion_v2_3000_eval_stop_20261105/combined_verification.json).

### goalroute BEFORE viewer의 목표→후행 사람 거리 지정

- `BEFORE_FOLLOWER_GOAL_DISTANCE=<m>` / `--before_follower_goal_distance <m>`를 추가했다. 사용자 확인에 따라 SIT·CLIMB·ON_TOP 모두 선행 AT 목표→후행 사람 root 초기 XY 거리다. 기존 선행 목표 거리 옵션과 독립 사용/병용 가능하며 생략 시 기존 동작이다.
- ON_TOP은 샘플링된 사람/payload를 함께 평행 이동해 집기 배치를 유지한다. payload→목표 거리는 그에 따라 변할 수 있다. 수동 거리 지정 시 near50 경로 부등식과 목표→사람 최소1m 제약을 대체하고 기존 물리 충돌 검사는 유지한다. 학습·정식 평가·비loco/비goalroute·잘못된 수치는 거부하며 YAML·checkpoint 계약·모델·보상·gate는 변경하지 않는다. config/USAGE에 명령을 기록했다.
- CPU76개 통과: .5/4/12m와 선행 목표 거리 조합, 여러 BEFORE 쌍·셔플된 목표/물체 매핑·부분 reset·선행/독립 목표·Z/회전 보존·ON_TOP 집기 벡터 보존·물리 충돌 검사 유지·formal 거부. 실제 CLI 로더의 유효7m+4m 및 잘못된7경로 거부 확인.
- GPU3·4·5에서 fusion_v1 5000 strict 복원, 각64환경 random_scenario·before_sit·before_climb·before_stack·all_solo 5경우16step finite Actor/Critic·가중치/가상 상자·부분 reset 검증. 총463쌍의 목표7m/후행 사람4m 오차는 .00005m 이내이며 ON_TOP 사람→payload 1–5m를 확인했다. GPU6·PORT6187 실제 VNC wrapper1환경·12step 제한1회 정상 종료. 원본/snapshot checkpoint·비대상 운영파일258개·기존3학습 보존. [검증](output_etc/check/before_follower_goal_distance_20261105/verification.json), [VNC 로그](output_etc/check/before_follower_goal_distance_20261105/vnc.log).

### goalroute BEFORE viewer의 선행 AT 목표 거리 지정

- test/VNC에 `BEFORE_AT_GOAL_DISTANCE=<m>`와 직접 CLI `--before_at_goal_distance <m>`를 추가했다. goalroute BEFORE loco 추론에서 선행 공유상자→AT 목표 초기 XY 거리만 지정하며 기존 목표 방향·Z·사람→상자 샘플링 규칙을 유지한다. 후행은 새 목표/선행 경로 길이를 기준으로 기존 goalroute/near50 배치를 적용한다. 생략 시 기존 동작이며 독립 과제·환경 YAML·checkpoint 계약·모델·보상·gate는 변경하지 않는다.
- 학습·정식 평가·비loco 및 0/음수/NaN/Inf를 거부한다. 기존 launcher·config CLI·formal guard·goalroute 배치에 최소 연결하고 config 추론 옵션/USAGE에 사용법을 기록했다.
- CPU50개 통과(새 거리/다중쌍·goal index 매핑·부분 reset·선행 물리 상태/Goal Z/독립 목표 보존 및 formal 거부 포함), 실제 config 로더의 유효7m/잘못된7경로 확인. GPU3·4·5에서 fusion_v1 5000 strict 복원 후 각64환경 random_scenario·before_sit·before_climb·before_stack·all_solo 5경우, 16step finite Actor/Critic·가중치 보존·가상 XYZ/실물 quaternion·부분 reset·near 조건을 확인했다. 총463쌍 목표거리 오차는 float32 월드좌표 기준 .00005m 이내다.
- GPU6·PORT6186 실제 VNC wrapper의 7m·CLIMB·1환경·12step 제한1회 종료까지 확인했다. 원본/snapshot checkpoint·비대상 운영파일258개·기존3학습 보존. [검증](output_etc/check/before_at_goal_distance_20261105/verification.json), [VNC 로그](output_etc/check/before_at_goal_distance_20261105/vnc.log).

## 2026-11-04

### penalty_v2 10000 / fusion_v1 5000 성공률·정지·출발지 분석

- GPU3·4에서 고정 penalty_v2 10000/fusion_v1 5000을 각각512환경×3seed×1200step Plane Ever로 평가하고, GPU5·6에서 같은 저장본64환경×seed0 정지 진단을 병행했다. 기존 random_scenario/near50·평가 규약 유지, 운영 코드·학습 설정 변경 없음.
- penalty_v2 9k→10k 후행 동시/같은 쌍 선행 AT 평균%는 전체34.42/91.06→33.47/91.34, 단독5행동 평균94.13→93.67. 행동별 후행/선행은 SIT47.12/96.20·CLIMB39.51/89.79·ON_TOP25.97/90.18. 전체 변화는seed별−5.00/+0.89/+1.27%p로 혼재하여 대체로 정체다.
- fusion_v1 4k→5k 전체39.39/94.78→40.58/95.87, 단독평균94.68→96.32. SIT59.17/98.31·CLIMB51.33/93.91·ON_TOP29.38/95.82. SIT/전체는3seed 공통 상승, ON_TOP은3seed 공통 하락, CLIMB 변화는 혼재했다. 모델별epoch가다르므로 같은 학습량 비교로 해석하지 않는다.
- 첫 root XYZ≤.3m/s·1초 연속 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m): penalty .92/.63/2.18(정지/관찰가능9/10·11/11·24/29), fusion1.82/2.01/2.83(9/10·7/8·23/23). ON_TOP payload1.40/1.56m. root 정지는 발/팔 정지를 뜻하지 않으며 정식 성공률과 진단을 합산하지 않는다.
- 선행 성공 전 관측 구간 내내 초기XY .5m 이내 유지한 후행은 penalty11/72·fusion37/72쌍이다. 모두 성공 후3초를 관찰했고 각각11/11·33/37쌍이 그 안에 .5m 이상 이동했다. fusion 출발지 근처 유지 중14쌍은 성공 전 페널티가 활성화되지 않았으며, 정지 거리만으로 목표 접근 후 대기라고 해석하지 않는다.
- 정식6회/진단2회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·native 진입/AT Ever 해제/정지 집계 일치·동결 encoder/RMS·checkpoint/운영파일262개 보존·기존3학습 생존 확인. [성공률 비교](output_etc/check/penalty10k_fusion5k_eval_stop_20261104/before_comparison.csv), [정지 비교](output_etc/check/penalty10k_fusion5k_eval_stop_20261104/stop_comparison.csv), [검증](output_etc/check/penalty10k_fusion5k_eval_stop_20261104/combined_verification.json).

### fusion_v2 2000 성공률·정지 거리·출발지 대기 진단

- GPU3에서 고정 fusion_v2 2000을 512환경×3seed×1200step Plane Ever로 평가하고, GPU4에서 같은 저장본 64환경×seed0 정지 진단을 수행했다. 기존 random_scenario/near50 유지, 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%는 SIT40.39/94.28·CLIMB36.74/90.84·ON_TOP18.63/92.88·전체27.33/92.69. 전체는1k11.30/86.10에서 상승했고 후행 세 행동 모두3seed 공통 개선했다. 단독5행동 동일 가중치 평균93.46→95.97%. 같은2k penalty_v2는전체15.58/80.88, fusion_v1은29.41/90.65로 v2의 전체 우위를 확정하지 않는다.
- 첫 root XYZ≤.3m/s·1초 연속 정지의 사람→목표XY 중앙값 SIT1.42m(9/11)·CLIMB1.84m(8/8)·ON_TOP1.88m(18/18), ON_TOP payload1.51m. 분모는 armed·미해제1초 관찰 가능 쌍이며 발/팔 정지를 뜻하지 않는다.
- 별도 출발지 분석에서 선행 성공 전 관측 구간 내내 초기XY .5m 이내인 후행13/72쌍, 그중 성공 후3초 관찰 가능한13쌍 중10쌍이3초 내 .5m 이상 이동했다. 출발지 근처 유지와 목표로 접근한 뒤 정지를 구분하며, 정지 거리만으로 접근 후 대기라고 해석하지 않는다.
- 정식3회/진단1회1200step 정상 완료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 정확 일치·human/+2.5m 진입 및 AT Ever 해제 검증·동결 encoder/RMS·checkpoint/운영파일262개 보존·기존3학습 생존 확인. [비교](output_etc/check/fusion_v2_2000_eval_stop_20261104/before_comparison.csv), [정지](output_etc/check/fusion_v2_2000_eval_stop_20261104/stop_comparison.csv), [출발지 분석](output_etc/check/fusion_v2_2000_eval_stop_20261104/spawn_audit/metrics.json), [검증](output_etc/check/fusion_v2_2000_eval_stop_20261104/combined_verification.json).

### penalty_v2 9000 / fusion_v1 4000 성공률 + 정지 거리

- GPU3·4에서고정penalty_v2 9000/fusion_v1 4000을각512환경×3seed×1200step Plane Ever로평가하고,GPU5·6에서같은저장본각64환경×seed0×1200step 정지진단을병행했다. random_scenario/near50·기존평가규약유지,운영코드/학습설정변경없음.
- penalty_v2 8k→9k 전체후행동시/같은쌍선행AT 평균% 34.21/90.48→34.42/91.06, 단독5행동평균94.41→94.13. 후행SIT49.77/선행96.50·CLIMB38.68/87.48·ON_TOP26.98/90.53. SIT증가와CLIMB감소가겹쳐전체정체,seed별변화는혼재했다.
- fusion_v1 3k→4k 전체34.40/94.03→39.39/94.78, 단독평균96.04→94.68. 후행SIT47.52/선행97.68·CLIMB50.59/91.79·ON_TOP31.67/94.95. 전체/ON_TOP은3seed공통상승,SIT평균+8.21%p는seed별편차가크고CLIMB는유지됐다.
- 정지거리중앙값(사람→목표XY,m)SIT/CLIMB/ON_TOP: penalty_v2 9k .97/.13/2.16, fusion_v1 4k 1.95/2.16/2.84. ON_TOP payload는1.37/1.56m. 정지/관찰가능건수각9/10·10/12·21/27,10/10·9/9·23/23. penalty_v2 CLIMB는8.5k1.84m→9k.13m로가까워졌고progress포화반경밖1초정지는8/12→2/12다. root XYZ≤.3m/s·1초연속첫확인기준이며정식성공률과합산하지않는다.
- 성공률6회/정지2회1200step 정상완료·CPU/GPU/JSON재채점일치·float32/64차이0·각규모동일초기상태·native진입/AT Ever해제/정지집계일치·동결encoder/RMS·checkpoint/운영파일보존·기존3학습생존확인. [성공률 비교](output_etc/check/penalty9k_fusion4k_eval_stop_20261104/before_comparison.csv), [정지 비교](output_etc/check/penalty9k_fusion4k_eval_stop_20261104/stop_comparison.csv), [통합 검증](output_etc/check/penalty9k_fusion4k_eval_stop_20261104/combined_verification.json).

### fusion_v2 1000 성공률 평가 + 64환경 정지 진단

- 같은고정fusion_v2 1000 저장본으로GPU3·512환경×3seed×1200step Plane Ever 성공률과GPU4·64환경×seed0×1200step 일반 장면 낙상 종료 정지 진단을 병행했다. 기존sampler/near50 유지, 운영 코드/학습 설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%는0.5k 9.29/74.11→1k 11.30/86.10, 단독5행동 동일 가중치 평균83.77→93.46%다. 후행SIT10.01/선행91.74·CLIMB5.58/77.80·ON_TOP14.08/87.48%. ON_TOP/전체는3seed 공통 상승했다.
- 같은1k의fusion_v1은후행/선행전체18.74/86.56%, 단독평균93.41%. v2는후행SIT−10.76·CLIMB−14.91·ON_TOP−3.23·전체−7.44%p이며세행동모두3seed에서v1보다낮았다. 선행전체는비슷해후행차이를선행성공률만으로설명할수없다.
- 정지거리중앙값(사람→목표XY,m)은SIT .65→1.15·CLIMB .13→1.22·ON_TOP .62→1.81, ON_TOP payload .32→1.46으로변했다. 정지/관찰가능SIT10/12·CLIMB10/13·ON_TOP17/21, progress포화반경밖정지는각10/12·9/13·15/21이다. rootXYZ≤.3m/s·1초연속첫확인기준이며발/팔정지나완료후재실행을보장하는판정은아니다.
- 성공률3회·정지진단1회1200step 정상종료·CPU/GPU/JSON재채점일치·float32/64차이0·각규모의기존초기상태정확일치·native human진입/AT Ever해제/정지집계일치·동결encoder/RMS·가중치/checkpoint/운영파일보존·기존3학습생존확인. [성공률 비교](output_etc/check/fusion_v2_1000_eval_stop_20261104/before_comparison.csv), [정지 비교](output_etc/check/fusion_v2_1000_eval_stop_20261104/stop_comparison.csv), [통합 검증](output_etc/check/fusion_v2_1000_eval_stop_20261104/combined_verification.json).

### 최신 penalty_v2/fusion v1/v2 64환경 정지 거리 및 평가 동반 규칙

- 사용자 요청에 따라 향후 대기 페널티 계열의 eval에는 같은checkpoint의64환경·seed0 정지 거리 진단을 함께 수행하도록 AGENTS/평가 규약에 기록했다. 성공률 정식 평가와 장면 낙상 종료 기반 정지 진단은 구분한다.
- GPU3·4·5에서 최신 고정penalty_v2 8500 / fusion_v1 3000 / fusion_v2 500을각64환경·seed0·4명·8토큰·1200step으로 긴급 진단했다. 선행 미완료·armed 이후 root XYZ≤0.3m/s·1초 연속 첫 확인 시점의 XY거리 기준이다.
- 사람→AT목표 중앙값 SIT/CLIMB/ON_TOP(m): penalty_v2 1.06/1.84/2.01, fusion_v1 1.69/2.08/2.84, fusion_v2 .65/.13/.62. ON_TOP payload→목표는1.28/1.56/.32m. 정지/1초 관찰 가능 건수 SIT/CLIMB/ON_TOP은각7/10·12/12·23/29, 9/9·9/9·23/23, 7/15·13/15·12/25다. 서로epoch가달라v2의반경변경효과를분리해해석하지않는다.
- 진단 driver는 task용payload 거리와 실제wait 진입 거리를분리해fusion_v2의human 기준을반영하고, 매step native armed/released가해당거리·반경·AT Ever와일치하는지검사했다. 기존학습/운영코드 변경 없음.
- 3회1200step 정상 종료·finite·동일 초기 상태·runtime 진입/해제/정지 집계 일치·가중치/원본checkpoint/운영파일260개 보존·기존3학습 생존 확인. [거리/건수 CSV](output_etc/check/latest_three_stop64_20261104/distances.csv), [쌍별 기록](output_etc/check/latest_three_stop64_20261104/pairs.csv), [검증](output_etc/check/latest_three_stop64_20261104/verification.json).

### penalty_v2 8000 / fusion3000 / fusion_v2 500 정식 평가

- GPU2·3·4·5에서 고정penalty_v2 8000·fusion3000·fusion_v2 500 및 동일epoch 비교용penalty_v2 3000을각4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다. 운영 코드/설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 8000 34.21/90.48(직전7000대비+2.97/+0.37%p), fusion3000 34.40/94.03(직전2000대비+4.99/+3.38%p), fusion_v2 500 9.29/74.11. 기존penalty_v2 CLIMB/ON_TOP/전체와fusion CLIMB/ON_TOP/전체는3seed 공통 상승했다. 단독5행동 동일 가중치 평균94.41/96.04/83.77%.
- 동일3000epoch penalty_v2→fusion 후행 SIT31.61→39.32·CLIMB21.74→50.16·ON_TOP20.64→26.12·전체23.16→34.40%, 같은 쌍 선행93.68→95.95·81.63→91.59·83.05→94.32·84.91→94.03%. 세 행동 모두3seed에서fusion우세.
- 동일500epoch fusion→fusion_v2 후행SIT10.03→8.34·CLIMB6.93→4.89·ON_TOP10.08→11.45·전체9.34→9.29%, 각 쌍 선행79.73→82.16·58.74→67.21·70.35→73.97·69.60→74.11%. ON_TOP은3seed 공통 개선했지만 전체 이득은 아직 없고, 단독평균88.56→83.77%로 저하했다. 초기 checkpoint이고 학습seed 반복 비교가 아니므로 개선/악화를 확정하지 않는다.
- 12회1200step 정상 완료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·같은 쌍 분모/상한·동결 encoder/RMS·checkpoint 및운영파일262개 보존·기존3학습 생존 확인. 실제 정지 거리/대기 행동 추가 진단 없음. [전체 추세](output_etc/check/penalty8k_fusion3k_fusionv2_500_eval_20261104/trends.csv), [seed별 변화](output_etc/check/penalty8k_fusion3k_fusionv2_500_eval_20261104/comparison.json), [검증](output_etc/check/penalty8k_fusion3k_fusionv2_500_eval_20261104/verification.json).

### fusion_v2 — ON_TOP 사람 기준 진입 / 대기 반경 +2.5m

- 기존fusion을 보존하고 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2` 환경/학습 YAML·train/test/VNC/eval wrapper를 추가했다. ON_TOP 대기 페널티 진입만 `ontop_distance_source: human`으로 선택하고, 모든 후행의 radius_buffer를3→2.5m로 변경했다. coefficient1.0·root XYZ 감점·진입 latch·선행 AT Ever 해제·Progress payload 기준·Virtual/fusion/bias·current Gate·AMP/PPO·샘플링은 유지한다. 같은 Stage1 26k에서 새로 전이한다.
- `before_wait.py`에 선택적 거리 기준을 연결하고 생략 시 기존payload 동작을 유지했다. validator·실험 등록·launcher·정식평가5개 분류 목록과 config/structure/reference 문서를 갱신했다. 기존/새 checkpoint 계약은 분리한다.
- CPU121개 통과(신규10개 포함):3행동 +2.5m 경계·XY 진입/XYZ 감점·ON_TOP 사람/payload 반대 위치·해제/부분reset·기존 설정/계약 보존. 27개 실험 설정 validator 및5개 shell 문법 검사 통과. 기존fusion2000 strict 복원/finite 확인.
- GPU4·4명/1024환경/8토큰 `CHECK=1 MAX_ITERATIONS=1` 학습exit0(기존 루프 저장epoch2). 저장본 strict 복원·Actor encoder/입력RMS 비트단위 동결·Actor/Critic finite·fusion 학습 확인. 새 eval wrapper64환경×seed0×1200step exit0·CPU/GPU/JSON 재채점 일치·float32/64 차이0·실제 runtime human/2.5/1.0 확인. 기존 운영파일250개 보존(승인된공통코드5개수정 제외), 기존프로세스 임의종료/재시작 없음. 본학습 미시작.
- [검증](output_etc/check/fusion_v2_config_20261104/verification.json), [CPU](output_etc/check/fusion_v2_config_20261104/cpu_tests.log), [학습](output_etc/check/fusion_v2_config_20261104/train.log), [GPU 평가](output_etc/check/fusion_v2_config_20261104/eval/general_seed0/result.json). 실행 명령은 [config](markdowns/config.md)에 추가했다.

### penalty_v2 1/2/4/5/6/7k 및 fusion1/2k·32환경 정지 거리 추세

- GPU2~6에서 요청8개 고정 저장본을각4명·32환경·seed0·1200step으로 진단했다. 기존 root XYZ≤0.3m/s·1초 연속·선행 미완료/armed 이후 첫 정지 기준과 일반 장면 낙상 종료를 유지했다. 운영 코드/설정 변경 없음, ON_TOP 진입은 저장본의 기존payload 기준이다.
- 사람→AT목표 XY거리 중앙값 SIT/CLIMB/ON_TOP(m): penalty_v2 1k .46/.09/.64, 2k .88/.11/.57, 4k 1.48/.07/1.21, 5k 1.55/.13/1.68, 6k 2.02/1.33/2.42, 7k 1.28/1.27/2.35; fusion1k 1.74/.63/2.75, 2k 2.43/3.01/2.79. ON_TOP payload 중앙값은각.20/.15/1.23/1.49/1.62/1.49/1.80/2.28m다.
- 동일32장면의 후행 SIT14·CLIMB6·ON_TOP16쌍 중 정지 사례만 거리 집계, 관찰 가능 분모와정지 건수를 CSV에 별도 기록했다. 소표본이고 root정지는발/팔 정지를보장하지않는다. 64환경 진단과표본을혼합하지않는다.
- 8회1200step 정상 완료·finite·모든 초기 상태 정확 일치·AT Ever 해제/runtime 정지 집계 일치·가중치/원본checkpoint/운영파일254개 보존·기존4학습 생존 확인. [거리/사례 수 CSV](output_etc/check/penalty_fusion_stop32_epochs_20261104/distances.csv), [쌍별 기록](output_etc/check/penalty_fusion_stop32_epochs_20261104/pairs.csv), [검증](output_etc/check/penalty_fusion_stop32_epochs_20261104/verification.json).

### fusion2000 6명/8명·64환경 진단 평가

- GPU3·4에서 같은 fusion2000 고정 저장본을6명/12토큰·8명/16토큰, 각각64환경·seed0·1200step으로 평가했다. 기존 인원 변경 확인 driver에서 Plane Ever·개인 낙상 freeze·random_scenario/near50를 재사용했다. 학습4명에서 변경한 소표본 진단으로512환경×3seed 정식 평가와 구분한다.
- 6명→8명 후행 동시/같은 쌍 선행 AT(%): SIT31.43/91.43(35쌍)→45.95/97.30(37쌍), CLIMB37.50/75.00(24)→32.56/81.40(43), ON_TOP18.37/83.67(49)→16.22/86.49(74), 전체26.85/84.26(108)→27.92/87.66(154). 단독5행동 동일 가중치 평균95.24→90.41%, 개인 낙상5.21→12.11%. 인원별 초기 장면/표본이 다르므로 순수 인원 효과나 통계적 우열을 확정하지 않는다.
- 두 실행1200step·CPU/GPU/JSON 재채점 일치·float32/64 차이0·finite·같은 쌍 분모/상한·동결 encoder/RMS·원본checkpoint/운영파일256개 보존·기존4학습 생존 확인. 8명은 초기 배치 검사에 시간이 더 걸렸지만 정책 rollout은 정상 완료됐다. [행동별 집계](output_etc/check/fusion_6_8_eval64_20261104/metrics.json), [BEFORE CSV](output_etc/check/fusion_6_8_eval64_20261104/before.csv), [검증](output_etc/check/fusion_6_8_eval64_20261104/verification.json).

### penalty_v2 7000 / fusion2000 및 동일2000 비교 평가

- GPU3·4·6에서 고정penalty_v2 7000·fusion2000·동일epoch 비교용penalty_v2 2000을각4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다.
- 최신 전체 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 7000은31.24/90.11(6000대비−2.48/−1.42%p), fusion2000은29.41/90.65(1000대비+10.68/+4.09%p). penalty_v2 전체/ON_TOP은3seed 공통 하락, fusion후행3행동 모두3seed 공통 개선했다. 단독5행동 동일 가중치 평균94.84/94.29%.
- 동일2000epoch penalty_v2→fusion 후행SIT21.18→35.43·CLIMB14.51→40.87·ON_TOP13.90→22.42·전체15.58→29.41%, 각 쌍 선행92.70→91.86·76.71→87.73·78.20→91.36·80.88→90.65%. 세 행동 모두3seed에서fusion우세. SIT는선행 개선 없이 후행 개선이 관찰됐으나 단일학습run 비교로 기전의 인과 효과를 확정하지 않는다.
- 9회1200step 정상 완료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·같은 쌍 분모/상한·동결 encoder/RMS·checkpoint/운영파일256개 보존·기존4학습 생존 확인. [전체 추세](output_etc/check/penalty7k_fusion2k_eval_20261104/trends.csv), [seed별 변화](output_etc/check/penalty7k_fusion2k_eval_20261104/comparison.json), [검증](output_etc/check/penalty7k_fusion2k_eval_20261104/verification.json). 실제 대기 행동은이번에추가측정하지 않았다.

### penalty_v1 64환경 정지 거리 비교

- GPU4·최종6500·64환경·seed0·1200step을 직전v2/fusion과 같은 root XYZ≤0.3m/s·1초 연속 기준 및 같은 초기 장면에서 진단했다. 첫 정지 사람→AT목표 XY거리 중앙값 SIT0.34/CLIMB0.15/ON_TOP0.65m, ON_TOP payload→목표0.23m로 v2/fusion보다 중심에 가까웠다.
- armed 미해제1초 관찰 가능 SIT14/CLIMB14/ON_TOP24쌍 중 root정지9/13/14쌍, progress 포화반경 밖1초 정지는0/1/5쌍이다. root정지 중 발속력까지 유지한 쌍18/36. 거리/행동 동작 후보를 엄밀한 행동 분류로 해석하지 않는다.
- 1200step 정상 종료·finite·동일 초기 상태·AT Ever 해제/runtime 정지 집계 일치·가중치/원본checkpoint/운영파일254개 보존·기존4학습 생존 확인. [집계](output_etc/check/penalty_v1_stop64_20261104/metrics.json), [쌍별 거리](output_etc/check/penalty_v1_stop64_20261104/pairs.csv), [검증](output_etc/check/penalty_v1_stop64_20261104/verification.json).

### penalty_v2 / fusion 64환경 실제 정지 거리 진단

- GPU3·4에서 최신 고정 penalty_v2 6500 / fusion1500을각64환경·seed0·4명·8토큰·1200step, 학습 random_scenario/near50·일반 장면 낙상 종료로 진단했다. 선행 미완료·대기 armed 이후 root XYZ속력≤0.3m/s가30step 연속 유지된 첫 확인 시점의 XY거리를 측정했다. 정식 성공률 평가와 구분한다.
- 첫 정지 사람→AT목표 거리 중앙값 SIT/CLIMB/ON_TOP은 penalty_v2 0.93/1.85/2.10m, fusion1.41/1.92/2.26m. ON_TOP payload→목표는1.28/1.46m. 1초 관찰 가능50/47쌍 중42/43쌍 root정지, 이 첫 구간 발속력도≤0.3으로 유지된 쌍은14/13뿐이다. ON_TOP 정지 시 payload높이가 초기보다0.1m 넘게 상승하지 않은 사례18/19·25/25로 집기 전 정지가 많이 포함된다. +3m는 페널티 진입 반경이며 실제 정지 거리를 고정하지 않는다.
- 두 실행1200step 정상 종료·finite·동일 초기 상태·AT Ever 해제 일치·runtime 정지 집계 일치·가중치/원본checkpoint/운영파일254개 보존·기존4학습 생존 확인. [집계](output_etc/check/penalty_v2_fusion_stop64_20261104/metrics.json), [쌍별 거리/시점](output_etc/check/penalty_v2_fusion_stop64_20261104/pairs.csv), [검증](output_etc/check/penalty_v2_fusion_stop64_20261104/verification.json).

### fusion / penalty_v2 동일1000epoch 평가

- GPU3(penalty_v2)·4(fusion)에서각1000 고정 저장본을4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다. 500epoch는 기존 결과를 재사용했다.
- 1000epoch 전체 후행 동시/같은 쌍 선행 AT 평균%는 penalty_v2 11.25/74.94 → fusion18.74/86.56. 후행 SIT11.99→20.77·CLIMB9.51→20.49·ON_TOP11.67→17.31, 각 쌍 선행81.72→92.79·66.95→81.33·75.72→86.41. 세 행동 모두3seed에서fusion이 우세했다. 단독5행동 동일 가중치 평균90.14→93.41%. 선행 개선도 동반되어 후행만의 인과 효과로 해석하지 않는다.
- 6회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 동일·같은 쌍 분모/상한·동결 encoder/RMS·원본checkpoint 및 운영파일256개 보존·기존4학습 생존 확인. 정지 행동 추가 진단 없음. [동일epoch 비교](output_etc/check/fusion_penalty_v2_1000_eval_20261104/same_epoch.csv), [행동/거리별 결과](output_etc/check/fusion_penalty_v2_1000_eval_20261104/metrics.json), [검증](output_etc/check/fusion_penalty_v2_1000_eval_20261104/verification.json).

### penalty_v2 6000 평가

- GPU3에서 고정6000 저장본을4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다. 전체 후행 동시/같은 쌍 선행 AT는33.73±1.45/91.53±0.25%, 직전5500 대비+3.68/+1.90%p다. 후행 SIT50.39·CLIMB36.90·ON_TOP26.36%, 각 쌍 선행96.34·91.58·89.78%. SIT/ON_TOP/전체는3seed 공통 개선, CLIMB는1seed 상승·2seed 소폭 하락했다. 단독5행동 동일 가중치 평균94.65→93.82%.
- 3회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 동일·동결 encoder/RMS·원본checkpoint 및 운영파일 보존·기존4학습 생존 확인. 정지→해제 행동 추가 진단은 미실시. [추세 CSV](output_etc/check/penalty_v2_6000_eval_20261104/trends.csv), [행동/거리별 결과](output_etc/check/penalty_v2_6000_eval_20261104/metrics.json), [검증](output_etc/check/penalty_v2_6000_eval_20261104/verification.json).

### 실행 중인4종 평가 — 기본25k/asym24.5k/penalty_v2 5.5k/fusion0.5k

- GPU2·3·4·6에서 고정 snapshot4개를4명/8토큰·512환경×3seed×1200step Plane Ever로 평가했다. random_scenario/near50·기존 정식 평가 규약 유지, 중단된 wait/penalty_v1은 재실행하지 않았다. 운영 코드/설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: 기본40.64/88.66, asym29.34/93.62, penalty_v2 30.04/89.64, fusion9.34/69.60. 직전 대비 기본−2.11%p(후행SIT는3seed 공통 하락), asym+2.92%p, penalty_v2+4.74%p. penalty_v2는후행3행동 모두3seed 공통 개선했다.
- 동일500epoch penalty_v2→fusion은 전체 후행10.09→9.34·같은 쌍 선행66.29→69.60, 단독SIT60.46±2.84→83.13±0.68이다. 초기 단독 성능 저하 완화는 관측됐으나 BEFORE 후행 개선은 아직 확인되지 않았다. 실제 정지→해제 행동의 추가 진단은 수행하지 않았다.
- 12회1200step·CPU/GPU/JSON 일치·float32/64 차이0·finite·과거와 동일 초기 상태·같은 쌍 분모/상한·encoder/RMS 동결·원본checkpoint/운영파일 보존·실행 중4학습 생존 확인. [행동별/거리별 수치](output_etc/check/active_four_eval_20261104/before_results.csv), [단독](output_etc/check/active_four_eval_20261104/solo_results.csv), [추세](output_etc/check/active_four_eval_20261104/trends.png), [seed별 변화](output_etc/check/active_four_eval_20261104/comparison.json), [검증](output_etc/check/active_four_eval_20261104/verification.json).

### penalty_v2 BEFORE relation fusion / zero background 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. 기존 penalty_v2에서 `human_relation_fusion: before_zerobg`와 실험 이름만 변경했다. +3m/1.0 속력 페널티·Ever 해제·가상 상자·거리/보상/Gate는 유지한다.
- 후행 query→선행 key의 BEFORE bias와 후행 attention 출력의 잔차 MLP에 같은64차원 embedding을 사용한다. embedding/MLP는4층 공유, projection은층/head별, Actor/Critic은별도다. SELF/NONE/미사용 COUPLED bias는0 고정, fusion은출력 projection 전에 후행에게만 적용하고 성공 후에도 역할 조건을 유지한다. 기본 경로/기존 checkpoint는 보존하고 새 fusion 계약을 분리했다.
- CPU 신규14개 포함131개 통과·3개 기존 skip. 별도 기존 테스트1개는 `checkpoints/Stage1PushFinetune_00021000.pth` 부재로 실패했다. 사용 중인26k 전이·초기 항등·gradient·단방향/두 쌍/셔플·Actor/Critic 가상 관측과 원본 packet·기존 penalty_v2 4500 strict 복원은 확인했다.
- GPU4·4명/1024환경/8토큰 `CHECK=1 MAX_ITERATIONS=2` 학습 exit0(기존 루프 저장 epoch3), Actor encoder 비트 단위 동결·양쪽 fusion/bias 학습·finite 확인. 새 eval wrapper로64환경×1seed×1200step 저장본 재실행 exit0. 기존 YAML/실험 wrapper 보존, 본학습 미시작. [검증](output_etc/check/before_relfusion_zerobg_20261104/verification.json), [학습](output_etc/check/before_relfusion_zerobg_20261104/train.log), [checkpoint 검사](output_etc/check/before_relfusion_zerobg_20261104/checkpoint_verification.json).

### penalty / wait_v2 / wait_v3 실제 정지·거리 비교

- GPU2·3·6에서 penalty6500/wait_v2 10500/wait_v3 7500 각512환경×3seed·일반 장면 낙상 종료로 진단했다. wait 대기 종료ON을 유지하고 반경 밖 정지도 별도 집계했다. penalty_v2 4500 직전3seed 결과를 재사용하고, wait_v2/v3는 같은 seed0의 대기 종료만OFF인 보조 실행을 추가했다(신규11회). 운영 코드/학습 변경 없음.
- 선행 미완료·진입 후1초 이상 관찰 집단의 반경 안1초 root 정지 SIT/CLIMB/ON_TOP: penalty132/235·229/303·327/662, 첫 정지 중심거리 중앙값0.36/0.12/0.19m. wait_v2는1/114·3/192·4/323, wait_v3는0/165·4/208·2/604에 그쳤다. wait 대기 종료 장면616/1536·926/1536. 기존penalty의 동작 후보 전 바깥 정지는9.8/7.9/18.4%, penalty_v2는66.7/39.7/55.0%였다. 기하 후보 검출이며 정확한 행동 분류는 아니다.
- 종료OFF seed0 첫 반경 안 정지 거리 SIT/CLIMB/ON_TOP: v2 0.78/0.17/0.20m, v3 0.31/0.22/0.25m. 종료를 제외해도 너무 접근한 뒤 멈추는 사례가 많다. ON/OFF 초기 상태는 동일하지만 분기 이후 공통 생존 궤적도 달라졌으므로 OFF는 별도 재실행 보조 진단으로 기록하고 프레임 단위 인과 비교로 해석하지 않았다.
- native 종료 판정·raw AT=Plane AT·release=raw AT Ever·초기 상태·finite·원본checkpoint/운영파일249개·기존6학습 보존 확인. [전체/행동별 정지·거리·직접 종료 거리·동일 집단 선행/후행](output_etc/check/wait_stop_comparison_20261104/report.md), [시계열 검증](output_etc/check/wait_stop_comparison_20261104/timing_verification.json), [보존/ON-OFF 차이](output_etc/check/wait_stop_comparison_20261104/verification.json).

### penalty_v2 4500 실제 대기→선행 완료→후행 수행 진단

- GPU2·3·6·512환경×3seed·일반 장면 낙상 종료/1200step에서 +3m/1.0 penalty_v2의 실제 armed/released·root XYZ 속력·발 높이/속력·payload·native/Plane 성공을 기록했다. 기존 정식 평가와 checkpoint·초기 상태 동일, 관측/물리/Gate/정책 변경 없음.
- 미완료 상태를1초 이상 관찰한1239쌍 중 root1초 정지973, 행동 후보 전에 progress 반경 밖 정지664. 선행 완료와 이후3초 생존을 관찰한 SIT159/CLIMB213/ON_TOP620쌍에서 연속 바깥 대기 유지77/49/245, 이어 재출발67/30/220, 이어 실제 동시 Plane 성공40/24/47이었다. 같은 대상 선행 AT는159/213/620 전부 성공. 조건부 진단이며 정식 전체 성공률과 구분한다.
- 선행 끝내 미완료230쌍 중179에서 조기 동작 후보 관측. SIT 높이 하강·CLIMB 높은 발 들기·ON_TOP 가상 배치 구역 진입에 기반한 기하 후보이며 행동 분류 정답은 아니다. 정지→해제→성공 사례는 있으나 일관된 대기는 미달. CLIMB 첫 root 정지 거리 중앙값0.57m로 너무 접근한 뒤 정지가 많다. [상세/같은 분모 순서표](output_etc/check/penalty_v2_wait_sequence_20261104/report.md), [성공 시계열](output_etc/check/penalty_v2_wait_sequence_20261104/ordered_success.png), [미완료 중 시도 후보](output_etc/check/penalty_v2_wait_sequence_20261104/premature_without_leader.png).
- 3seed 실제 종료 판정·초기 상태 일치·native AT와 Plane AT 일치·해제=raw AT Ever/영구 유지·finite·가중치/원본checkpoint/운영파일249개·기존6학습 보존 확인. [검증](output_etc/check/penalty_v2_wait_sequence_20261104/verification.json), [추가 시계열 검증](output_etc/check/penalty_v2_wait_sequence_20261104/sequence_details.json).

### 최신6종 후속 평가 — 기본/asym23.5k 및 wait/penalty 추세

- GPU2·3·6만 사용해 기본23500/asym23500/wait_v2 10500/wait_v3 7500/penalty6500/penalty_v2 4500 고정 snapshot을 4명/8토큰·512환경×3seed×1200step Plane Ever로 평가했다. random_scenario/near50·대기 종료 제외 규약과 이전 평가의 환경 설정·초기 상태를 유지했다. 운영 코드/설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: 기본42.75/88.66, asym26.42/92.96, v2 23.62/88.55, v3 15.49/82.68, penalty17.59/83.67, penalty_v2 25.31/87.85. 기본/asym 직전 하락 후 반등, v2 정체, v3 소폭 개선, 기존penalty 소폭 하락. 새penalty_v2는500→4500에서 단독SIT60.46→92.60으로 회복했으나 중간 checkpoint 추세·실제 정지 행동은 이번에 검증하지 않았다.
- 18회1200step·CPU/GPU/JSON 일치·float32/64 판정 차이0·finite·동일 초기 상태·같은 쌍 분모/상한·거리 분할 합계·동결 encoder/RMS·checkpoint 및 운영파일249개 보존·기존6학습 생존 확인. [행동별/거리별/단독/최근 추세](output_etc/check/needed_eval_followup_20261104/report_plane.md), [그래프](output_etc/check/needed_eval_followup_20261104/trends.png), [seed별 변화](output_etc/check/needed_eval_followup_20261104/comparison.json), [검증](output_etc/check/needed_eval_followup_20261104/verification.json).

### 필요한 최신6종 평가 — GPU0·1 제외

- GPU2·3·6에서 기본20000/asym19500/wait_v2 8000/wait_v3 4500/penalty3500/penalty_v2 500 고정 snapshot을4명/8토큰·512환경×3seed×1200step Plane Ever로 평가했다. random_scenario/near50와 wait 대기 종료 제외 규약 유지. 중간 checkpoint 추가 평가는 생략하고 기존 최근2개 및500epoch 대조 결과를 재사용했다.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: 기본38.93/88.77, asym24.11/93.36, v2 23.86/86.92, v3 13.10/80.66, penalty18.47/83.66, 새penalty_v2 10.09/66.29. 최근 기본/asym 후행 하락·v3 정체, v2/기존penalty 개선. 같은500에서 기존/new penalty 전체9.18/71.88→10.09/66.29이나 단독SIT85.07→60.46으로 저하. 실제 정지 행동의 추가 진단은 수행하지 않았다. [전체·거리별 같은 쌍 선행/후행·단독·추세·std](output_etc/check/needed_eval_20261104/report_plane.md), [그래프](output_etc/check/needed_eval_20261104/trends.png).
- 앞서 추가한 penalty_v2가 `evaluate.py`의5개 실험 분류 목록에서 누락되어 task/motion/PUSH 경로 등록만 보완했다. 평가 CPU18개 통과. 초기 GPU3/6 MPS 상속 설정 불일치는 실행 전 차단됐고, 평가 자식 환경의 상속 pipe를 해제해 각 GPU 검증 daemon으로 재시도했다. 실제 평가 GPU UUID→2/3/6 확인.
- 최종18회1200step·CPU/GPU/JSON 재채점·float32/64 판정 일치·finite·동일 초기 상태·같은 쌍 분모/상한·거리 분할 합계·encoder/RMS 동결·checkpoint 및 등록 보완 후 운영파일249개 해시 보존 확인. 기존6개 학습 생존. [검증](output_etc/check/needed_eval_20261104/verification.json).

## 2026-11-03

### wait_penalty_v2 — 진입 반경 +3m / 속력 가중치1.0

- 기존 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty`를 보존하고 `_v2` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. radius_buffer만2→3m, coefficient만0.6→1.0으로 변경했다. 진입 Ever·선행 성공 Ever 해제, root XYZ 속력, 기존 Progress/current State·Success Gate·AT 높이 제한 해제는 유지하며 대기 조기 종료는 없다. 실험/output 이름을 분리하고 registry/launcher/config/structure를 연결했다.
- CPU103개 통과 및 새 설정의3행동 진입 경계·XYZ 감점·이탈 후 유지·성공 후 해제·부분 reset·기존 checkpoint 계약 분리 확인. GPU0·4명/1024환경/8토큰 `CHECK=1` 짧은 학습 exit0, Actor/Critic 실행·checkpoint 저장·저장된+3/1.0 설정 확인. 기존 penalty YAML/스크립트6개 해시 보존. 본학습은 시작하지 않았다. [검증](output_etc/check/wait_penalty_v2_config_20261103/verification.json), [학습 로그](output_etc/check/wait_penalty_v2_config_20261103/train.log).

### wait_v3 3500 가상 상자 거리 개입 CLIMB 토이

- GPU0·32환경·seed2에서 선택 후행 Alias GTA XY만 root 기준3.5→0.2m로 당겼다가 멀리 보내는12초 주기3회와 동일 초기 상태3.5m 대조군을 실행했다. 대기 종료만 끄고 원래 낙상/시간 종료를 유지했다. 운영 코드/weight/물리 상태 직접 수정 없음.
- 발 중심 Z>0.45m 3step 기준 시도 후보 시작 거리 중앙값1/2/3차0.73/0.99/0.99m, 2회 이상 검출28/32 대 대조군0/32. 선행 AT가 끝내 성공하지 않은 env6에서도0.75/0.66/0.88m에서 높은 발 올리기를 반복했고 실제 영상으로 확인했다. 고정 행동 임계값·정지 성공·학습 성공률로 해석하지 않는다. [결과/한계](output_etc/check/virtual_climb_trigger_20261103/report.md), [거리·발/골반 높이](output_etc/check/virtual_climb_trigger_20261103/distance_and_height.png), [0.5배속 영상2개 ZIP](output_etc/check/virtual_climb_trigger_20261103/virtual_climb_toy_slow.zip).
- 입력 변경 범위·초기 상태 동일·계측 위치/높이 및 선행 성공의 렌더 재현·MP4 decode/ZIP CRC·운영파일241개/checkpoint/기존5학습 보존 확인. 접촉 힘/native done 일부는 렌더 유무에 따라 달라 원본 trace를 집계했으며 대표 env6 생존은 동일하다. [검증](output_etc/check/virtual_climb_trigger_20261103/verification.json).

### source5/asym goalroute 전체 평가 추세 재집계

- 신규 시뮬레이션 없이 현재 goal5_goalroute의0.5/1/1.5/6/7/9/14.5/15.5/17/18k, 두 모델20checkpoint를 통합했다. BEFORE 행동별·거리별 후행/같은 쌍 선행 AT와 단독5행동·표준편차를 [전체 표](output_etc/check/source5_asym_full_trends_20261103/report.md), [BEFORE 그래프](output_etc/check/source5_asym_full_trends_20261103/before_trends.png), [단독 그래프](output_etc/check/source5_asym_full_trends_20261103/solo_trends.png)에 기록했다. 다른 거리 설정·8명·AT9 평가는 섞지 않았다.
- seed별 동일 부분집합 분모·후행 성공 상한·평균/std 재계산 일치를 확인했다. 장기적으로 기본3행동은 개선됐으나 최근 정체, asym은9k 이후 CLIMB 개선이 뚜렷하고 SIT/ON_TOP 개선은 작거나 변동이 크다. 운영 코드/학습 변경 없음.

### wait_v3 3000 +2m 실제 대기 조기 종료 영상

- GPU0·3k 고정 checkpoint·seed2·4명/8물체·행동별 preset으로 SIT/CLIMB/ON_TOP 해당 후행의 실제 대기 종료를 촬영했다. 모두 r_box+2m 반경 밖 접근부터 누적 XYZ 이동 위반1초 종료까지 연속이며 실제 길이3.63/5.27/2.70초. ON_TOP은 자연 실행에서 사례를 먼저 찾아 같은 종료 frame/속력이 재현된 장면을 촬영했다. 처음 고른 부적절한 ON_TOP 촬영만 중단·교체했다.
- 1280×800·0.5배속·시작2초/끝4초 정지, 사람/목표/반경·속력·누적 위반 자막을 표시했다. 초기 미갱신 렌더1프레임만 제외, 물리/정책 상태 조작 없음. [영상 폴더](output_etc/check/wait_v3_3000_videos_20261103/slow_videos/), [ZIP 약4.6MB](output_etc/check/wait_v3_3000_videos_20261103/wait_v3_3000_early_termination_slow.zip).
- 3건 모두 해당 후행 waiting 종료·선행 미해제·누적1초 확인. 전체 MP4 decode·ZIP CRC·시작/종료 화면 확인·checkpoint 및 운영파일241개 해시·기존5개 학습 보존. 선택 사례이며 성능 통계가 아니다. [검증](output_etc/check/wait_v3_3000_videos_20261103/verification.json).

### wait_v2/v3/penalty 동일 epoch 비교 — v2는1k 간격

- 사용자 범위 축소에 따라 v2는1000/2000/3000만, v3는500~3000의500간격, penalty는요청 시점 생성된500~2000의500간격을 비교했다. 진행 중이던 평가 전용 v2 500을 중단하고1500/2500 대기 작업은 취소했으며 본학습은 보존했다. v2 2000/3000·v3 3000·penalty1000만 GPU0·512환경×3seed×1200step 신규 평가12회, 기존9checkpoint는 trace 재집계로 재사용했다.
- 같은3k 전체 후행 동시/같은 쌍 선행 AT 평균%는 v2 15.95/80.91·v3 13.73/77.68. 같은2k는 v2 14.64/77.32·v3 11.42/77.00·penalty15.70/79.04. v2는3k ON_TOP이 높고 CLIMB는v3와 비슷하며, penalty는2k SIT가 높다. [전체/거리별 같은 쌍 선행·후행 및 단독 표](output_etc/check/matched_wait_epochs_20261103/report_plane.md), [같은 epoch 그래프](output_etc/check/matched_wait_epochs_20261103/matched_trends.png).
- 총13checkpoint/39rollout의 CPU/GPU/JSON 일치·finite·float32/64 판정 차이0·동일 초기 상태·seed별 동일 부분집합 분모 확인. 신규 checkpoint/운영파일243개 해시·encoder/RMS 동결·기존5개 학습 보존. wait 조기 종료 제외 규약 유지, 실제 대기 행동 추가 진단 없음. [검증](output_etc/check/matched_wait_epochs_20261103/verification.json).

### source5/asym18000·wait_v2 7000·wait_v3 2500·penalty2000 추세 평가

- 비어 있던 GPU0에서 고정5개 checkpoint·4명/8토큰·512환경×3seed×1200step Plane Ever 평가15회를 완료했다. 학습 random_scenario/near50 및 기존 정식 규약(wait 대기 종료 제외)을 유지했다. 전체 후행 동시/같은 쌍 선행 AT 평균%는 source5 40.33/86.68·asym26.48/90.59·v2 22.34/85.17·v3 13.85/76.66·penalty15.70/79.04.
- 직전 대비 기본/asym 전체 하락, v2/v3/penalty 반등. asym CLIMB는 개선됐지만 SIT는 하락했다. 단독 SIT는 v3 88.15→96.13·penalty80.02→88.02로 회복했다. 학습량·AT 추가 높이 조건 차이를 명시하고 행동별/거리별 같은 쌍 선행·단독·과거 추세·표본std를 [보고서](output_etc/check/next_trends_20261103/report_plane.md), [그래프](output_etc/check/next_trends_20261103/trends.png)에 기록했다. 실제 대기 행동의 추가 진단은 수행하지 않았다.
- 15회1200step·CPU/GPU/JSON 재채점 일치·float32/64 판정 차이0·finite·과거 및 모델 간 초기 상태 동일·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일243개 해시·기존5개 학습 보존. 운영 코드/설정 변경 없음. [검증](output_etc/check/next_trends_20261103/verification.json).

### source5/asym17000·wait_v2 6500·wait_v3 2000 평가 및 추세

- GPU4·4명/8토큰·512환경×3seed×1200step Plane Ever, 학습 random_scenario/near50 유지·wait 대기 종료 제외. 새4개 checkpoint를 고정해12회 평가했다. 전체 후행 동시/같은 쌍 선행 AT 평균%는 source5 42.77/87.42·asym29.21/92.19·v2 20.35/84.10·v3 11.42/77.00. 직전 대비 기본/asym 상승·v2 유지·v3 하락이며 v3 단독 SIT95.03→88.15. penalty1500은 기존 평가를 재사용했다.
- 행동별 같은 쌍 선행/후행·거리 부분집합·단독·checkpoint 추세와 표본std를 [통합 보고서](output_etc/check/combined_trends_20261103/report_plane.md), [추세 그래프](output_etc/check/combined_trends_20261103/trends.png)에 기록했다. 모델별 학습량 및 AT 추가 높이 조건 차이를 명시했다. 과거1500 대기 진단은2000 행동 검증으로 간주하지 않는다.
- 12회 완료·CPU/GPU/JSON 재채점 일치·finite·초기 상태 일치·같은 쌍 상한·거리 분할 합계 확인. checkpoint/운영파일243개 해시 보존·기존5개 학습 생존. 운영 코드/설정 변경 없음. [통합 검증](output_etc/check/combined_trends_20261103/verification.json), [v3 검증](output_etc/check/wait_v3_eval2000_20261103/verification.json).

### penalty1500 정식 평가·wait_v3 동일1500 대기/해제 비교

- GPU4·512환경×3seed×1200step Plane Ever. penalty1500 후행 동시/같은 쌍 선행 AT 평균%는 SIT14.17/87.79·CLIMB8.49/69.29·ON_TOP14.86/80.12, 전체13.22/79.24. penalty500 대비 SIT/ON_TOP 상승·CLIMB 정체, 단독 SIT85.07→80.02. 동일1500 wait_v3와 전체 성능은 비슷하고 행동별 차이가 있다. 기본15500/wait_v2 6000은 학습량이 다른 참고값으로 분리했다. [정식 비교·판단 요약](output_etc/check/penalty_eval1500_20261103/report_plane.md).
- 같은 초기512환경·seed0으로 v3 종료ON/OFF 및 penalty 일반 장면 종료 진단. 진입 후1초 root 정지2/402·184/450·221/461, 동일 진입 집단 선행 Plane Ever9/402·261/450·314/461. v3 ON은364/512장면 대기 종료·평균12.05초(OFF32.77, penalty34.85). OFF/penalty 첫 정지 목표 거리 중앙값 약0.20m, 정지 사례 대부분 이미 누적 이동1초 초과. CLIMB 해제 직전 정지·이후3초 관찰 가능 집단의 이동 재개는 v3 OFF4/20·penalty1/18, 해당 집단 장면 끝까지 후행 동시/선행 Ever는1/20·20/20 및0/18·18/18. 1seed 조건부 표본이며 root 정지가 스킬 시작 전 대기를 보장하지 않는다. [대기 상세](output_etc/check/penalty_eval1500_20261103/report_wait.md).
- 정식3회 CPU/GPU/JSON 일치·float32/64 차이0·finite·penalty500/v3 1500 초기 상태 일치. 진단3건 실제 종료 판정·가중치 보존·정식seed0 포함 초기 상태 정확 일치. checkpoint/운영파일243개 해시·기존5개 학습 보존. 운영 코드/설정 변경 없음. [정식 검증](output_etc/check/penalty_eval1500_20261103/verification.json), [진단 검증](output_etc/check/penalty_eval1500_20261103/native_verification.json).

### wait_v3 1500 정식 평가·500/1000 추세

- GPU4·4명/8토큰·512환경×seed0/1/2×1200step Plane Ever, 학습 random_scenario/near50 유지·정식 규약상 대기 종료 제외. 고정1500 snapshot의 후행 동시/같은 쌍 선행 AT 평균%는 SIT11.20/86.82·CLIMB12.90/62.89·ON_TOP13.48/76.59, 전체12.86/75.55였다. 1000 대비 전체12.43→12.86, CLIMB13.34→12.90으로500의26.09에는 회복하지 못했다. 단독·동일 쌍 거리별 선행/후행·3checkpoint 추세와std는 [보고서](output_etc/check/wait_v3_eval1500_20261103/report_plane.md)에 기록했다.
- 3회1200step·CPU/GPU/JSON 일치·float32/64 누적 판정 차이0·finite·500/1000 초기 상태 정확 일치·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일243개 해시 보존·기존5개 학습 생존. 운영 코드/설정 변경 없음. [검증](output_etc/check/wait_v3_eval1500_20261103/verification.json).

### wait_v2 6000 정식 평가·5500 대비

- GPU4·4명/8토큰·512환경×seed0/1/2×1200step Plane Ever, 학습 random_scenario/near50 유지 및 정식 규약상 대기 종료 제외. 최신 번호6000 snapshot의 후행 동시/같은 쌍 선행 AT 평균%는 SIT22.81/91.93·CLIMB17.87/81.64·ON_TOP20.40/84.61, 전체20.31/85.42였다. 5500 대비 전체 후행17.99→20.31, 세 행동 평균 모두 상승. 단독·거리 부분집합·표본std는 [보고서](output_etc/check/wait_v2_eval6000_20261103/report_plane.md)에 기록했다.
- 3회1200step·CPU/GPU/JSON 일치·float32/64 누적 판정 차이0·finite·5500 초기 상태 정확 일치·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일243개 해시 보존·기존5개 학습 생존. 운영 코드/설정 변경 없음. [검증](output_etc/check/wait_v2_eval6000_20261103/verification.json).

### wait_v3 1000 정식 평가·500 대비

- GPU4·4명/8토큰·512환경×seed0/1/2×1200step Plane Ever, 학습 random_scenario/near50 유지 및 정식 규약상 대기 조기 종료 제외. 고정1000 snapshot으로 후행 동시/같은 쌍 선행 AT 평균%는 SIT10.42/80.38·CLIMB13.34/63.48·ON_TOP12.80/75.31, 전체12.43/73.63이었다. 500 대비 후행 CLIMB26.09→13.34, 전체15.26→12.43. 단독·거리 부분집합·표본std는 [보고서](output_etc/check/wait_v3_eval1000_20261103/report_plane.md)에 기록했다.
- 3회1200step·CPU/GPU/JSON 일치·float32/64 누적 판정 차이0·finite·500 초기 상태 정확 일치·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일 해시 보존·기존5개 학습 생존. 정식 성공률이며 대기 행동 학습 여부를 판정하지 않는다. [검증](output_etc/check/wait_v3_eval1000_20261103/verification.json).

### wait_v3 / penalty 500 실제 장면 0.5배속 묶음

- 직전 평가와 같은 두 모델500 snapshot으로 GPU4·seed2·행동별 preset·4명/8물체에서 SIT/CLIMB/ON_TOP을 촬영했다. wait_v3 대기 종료 ON, penalty 개인 감점 유지. 초기부터 종료/최대20초까지 실제 상태를 기록하고 반경·속력·종료 누적/감점량을 표시했다. ON_TOP 다른 쌍 종료 사례를 포함한7개 MP4를 [한 폴더](output_etc/check/wait_v3_penalty_videos_20261103/slow_videos/) 및 [ZIP](output_etc/check/wait_v3_penalty_videos_20261103/wait_v3_penalty_500_slow_videos.zip)에 모았다.
- 1280×800·0.5배속·시작2초/끝4초 정지 화면, reset 직후 미갱신 렌더1프레임만 제외. MP4 전체 decode·ZIP CRC·checkpoint 해시 확인, 기존5개 학습 생존. 촬영 프로세스 중단 후 미완료 penalty2건만 재실행했다. 선택 장면은 성능 통계가 아니며 episode/종료 사유는 README와 verification.json에 기록했다. 운영 코드/설정 변경 없음.

### 실행 중5개 최신 저장본 GPU4 Plane Ever 평가

- 요청 시점 기본15500·asym15500(15k resume run)·wait_v2 5500·wait_v3 500을 고정하고, 아직 저장본이 없던 penalty는 첫500 생성 후 포함했다. 모든 입력을 검증 경로에 snapshot으로 보존해 GPU4·4명/8토큰·512환경×seed0/1/2×1200step로 평가했다. 학습 random_scenario/near50 유지, wait_v2/v3 대기 종료는 정식 규약상 제외했다.
- 전체 후행 동시/같은 쌍 선행 AT Ever 평균%는 기본40.59/86.86·asym22.43/89.24·wait_v2 17.99/84.77·wait_v3 15.26/74.37·penalty9.18/71.88이었다. 행동별·단독·후행/선행 근거리·직전 평가 추세와 표본std는 [보고서](output_etc/check/latest_five_eval_20261103/report_plane.md)에 기록했다. 모델epoch 및 기존 AT 추가1mm 학습 조건 차이를 명시했다.
- 15회 모두1200step·CPU/GPU/JSON 재채점 일치·float32/64 누적 판정 차이0·finite·초기 상태 동일·같은 쌍 상한·거리 분할 합계 검증. encoder73tensor/입력 RMS 동결·checkpoint 해시·운영파일240개 보존·기존5개 학습 생존 확인. 평가 중 요청받은 viewer 토글3개 파일 변경은 별도 해시로 추적했으며 정식평가 동작에는 영향 없다. [검증](output_etc/check/latest_five_eval_20261103/verification.json).

### BEFORE viewer 대기 조기 종료 선택 옵션

- 요청에 따라 wait/wait_v2/wait_v3 test/VNC의 `BEFORE_WAIT_TERMINATION=0`을 추가했다(기본1). `--disable_before_wait_termination`으로 추론 중 대기 위반 종료만 건너뛰며, 낙상/시간 종료·checkpoint 설정·학습·penalty 보상은 유지한다. 학습 직접 CLI 사용은 거부한다. USAGE/config에 명령을 기록했다.
- GPU4·wait_v3 500에서8환경16step 및 상태 주입 ON/OFF 검사로 기존 대기 종료·옵션OFF 시 낙상/시간 종료 버퍼 보존·부분 reset·정식평가 우회를 확인했다. 실제 test wrapper도1환경8step 정상 완료했고 플래그 전달/학습 거부·shell 문법을 확인했다. [물리 검증](output_etc/check/viewer_wait_toggle_20261103/physics/physics_validation.json), [wrapper 로그](output_etc/check/viewer_wait_toggle_20261103/wrapper.log).
- 진행 중 최신5개 정식평가는 원래 대기 종료를 우회하므로 이 추론 옵션 추가는 채점 동작을 바꾸지 않는다. 해당3개 파일의 승인된 변경 해시는 평가 검증에서 별도 추적한다.

### Virtual wait_penalty: 개인 연속 XYZ 속도 감점

- virtual_goal5_goalroute 기반 `_wait_penalty` env/train·train/test/VNC/eval·registry·평가 목록을 추가했다. AT 추가1mm 높이 제한 해제(기존3D20cm 유지), 목표 XY r_box+2m 최초 진입 후 후행 개인 task 보상에서 `0.6×root XYZ 속력`을 매 step 차감한다. 선행 raw AT 최초 성공 시 같은 step부터 영구 해제하며 기존 state/success current gate·progress·HOLDING·AMP는 유지한다. 대기 위반 종료는 없다.
- 기존 BeforeWait의 진입/해제 기록을 공유하고 새 감점을 별도 reward term에 기록했다. 기존 설정/계약은 보존하고 새 계약/출력은 분리했다. config/USAGE/structure/상세 문서에 실행법과 의미를 추가했다.
- CPU183통과·3skip. GPU4·4명/1024환경 짧은 학습2epoch/262144frame 완료·finite·actor encoder73tensor/입력 RMS 동결, wait_penalty 로그 확인. 8환경×6preset 각16step 및 상태 주입 검사에서 개인 감점량·타 보상 항 보존·종료 버퍼 불변·진입/성공 기록·부분 reset·Virtual 관측 확인. 기존 YAML46개 해시 보존. 본학습은 시작하지 않았다. [검증 보고서](output_etc/check/before_wait_penalty_20261103/report.md).

### wait_v3: 대기 진입 반경 +2m

- 사용자 요청으로 wait_v2를 보존하고 `_wait_v3` env/train YAML·train/test/VNC/eval wrapper·registry·실행/평가 목록을 연결했다. 동작 설정 차이는 `waiting_termination.radius_buffer: 1.0 → 2.0` 하나이며 progress·XYZ 속력0.3m/s·유예0초·누적1초 종료·선행 성공 시 영구 해제·AT 기준을 유지했다. 기본 Stage1 26k 전이로 별도 새 학습한다. config/USAGE/structure에 경로와 명령을 추가했다.
- 기존 대기 CPU87개 통과. SIT/CLIMB/ON_TOP별 +2m 경계 안팎 및 +1.5m 진입·30step 누적 종료·선행 성공 즉시/영구 해제9사례, v2 설정/스크립트 해시 보존·반경 외 YAML 동일·checkpoint 계약 분리·shell 문법 통과. GPU4에서4명·1024환경 짧은 학습2epoch/262144frame 정상 종료·저장, 실제 저장 설정+2m·모델 finite 확인. 본학습 시작/기존 프로세스 종료 없음. [설정 검증](output_etc/check/before_wait_v3_20261103/config_check.json), [실행 검증](output_etc/check/before_wait_v3_20261103/training_check.json).

### wait_v2 4500 행동 전 정지 진단·실제 장면

- GPU4·512환경·seed0 첫 episode에서 일반 낙상/대기 종료를 유지했다. 선행 성공 전 반경 진입 후행298건 중 반경 안 root XYZ≤0.3m/s 1초 연속 정지는 SIT0/65·CLIMB2/82·ON_TOP5/151이었다. 대기 위반 종료267/512, 평균18.91초. 같은 쌍 선행 AT Ever를 진입 집단/전체별 함께 기록했다.
- 실제 선택 장면에서 CLIMB·ON_TOP은 반경 진입 후 정지하지 않고 발 들기/상자 내리기 동작을 시작한 뒤 누적 위반1초로 종료됐다. 세 영상 모두 선행 AT 성공 전이며 SIT는 계속 접근하다 종료됐다. 0.5배속 접근부터 종료까지 영상 및 [보고서](output_etc/check/wait4500_stop_20261103/report.md). 정지 지표만으로 행동 전 대기 성공을 뜻하지 않으며 정식평가와 구분한다.
- 1k 진단과 초기 상태 정확 일치·실제 종료 판정 일치·checkpoint/운영파일229개 해시 보존·기존4개 학습 생존 확인. 운영 코드/설정 변경 없음.

### 기본/virtual/asym/wait 최신4개 GPU4 Plane Ever 평가

- 요청 시점 최신 저장본 source5 14500·virtual13500(재개 run)·asym14500·wait_v2 4500을 GPU4에서 각512환경×seed0/1/2×1200step·4명/8토큰으로 평가했다. 사용자 명칭에 맞춰 이번 보고서의 wait는 wait_v2이며 기존 wait 실험과 파일명은 보존했다. 정식평가에서는 대기 종료를 제외했다.
- BEFORE 후행 SIT/CLIMB/ON_TOP 평균%는 기본60.87/53.69/28.09, virtual34.87/7.14/20.08, asym16.14/40.94/17.96, wait23.99/23.52/20.42였다. 전체 후행/같은 쌍 선행AT는40.80/86.75·20.13/83.51·22.89/91.49·21.85/83.31%. 각 행동·전체/후행근거리/선행근거리마다 같은 쌍 AT Ever 및std를 함께 기록했다. 단독5행동·직전 평가 추세도 [보고서](output_etc/check/latest_four_eval_20261103/report_plane.md)에 정리했다.
- 12회 모두1200step·CPU/GPU/JSON 재채점 일치·float32/64 누적 차이0·finite·seed별 초기 상태 동일·같은 쌍 AT 상한·거리 분할 합계 통과. 각 checkpoint의 encoder73tensor/입력 RMS 동결, checkpoint 및 운영파일229개 해시 보존, 기존4개 학습 프로세스 생존을 확인했다. 운영 코드/설정 변경 없음. [검증](output_etc/check/latest_four_eval_20261103/verification.json).

### wait_v2 1000 GPU4 정식 평가·실제 대기 종료 진단

- 사용자 요청으로1k를 GPU4에서4명·8토큰·512환경×seed0/1/2×1200step Plane Ever 평가했다. 정식평가에서 대기 종료는 제외한다. 전체 BEFORE 후행 동시 Ever/같은 쌍 선행 AT Ever는 SIT12.91±0.82/82.31±3.74%, CLIMB21.52±1.44/60.72±4.19%, ON_TOP13.78±1.00/74.26±4.09%였다. 전체15.39±0.94/72.78±4.11%, 기존wait1k 후행9.17% 대비 높았다. 단독 SIT91.55·CLIMB98.59·AT85.38·ON_TOP86.41·PUSH95.65%. 전체/후행근거리/선행근거리·동일1k 비교는 [정식 보고서](output_etc/check/wait_v2_eval1000_20261103/report_plane.md).
- 같은1k·seed0·512환경 첫 episode로 일반 장면 낙상 종료를 유지하고 대기 종료 ON/OFF를 비교했다. 초기 상태 정확 일치. ON은297/512장면이 대기 위반 포함 종료, 평균16.67초(OFF31.99초). 진입327후행 중 AT 성공 전 반경 안에서 root XYZ≤0.3m/s로1초 연속 정지1명. 위반9432프레임 중 XY만으로는 위반하지 않고 XYZ로만 추가된 경우117(1.24%). 같은 쌍 선행 AT Ever는 ON26.72%/OFF62.07%로 종료에 따른 수행 기회 감소를 확인했다. 이는1seed 진단이며 정식3seed 성능표와 구분한다. [대기 보고서](output_etc/check/wait_v2_eval1000_20261103/report_native.md).
- 정식 wrapper 실행 중 `evaluate.py`의 wait/wait_v2 등록 누락으로 task 선택이 실패했다. 반복된 기존 실험 목록5곳에 두 이름만 추가해 task·모션·PUSH 집계를 연결했으며 재실행3seed 완료·CPU18테스트 통과. 학습 코드/보상/종료 설정은 수정하지 않았다.
- CPU float64/GPU/JSON 채점 일치·float32/64 누적 차이0·1200step/finite·기존virtual과seed별 초기 상태 동일·encoder73tensor/입력 RMS 동결·checkpoint/운영파일 해시 보존·기존4개 학습 생존을 확인했다. [검증](output_etc/check/wait_v2_eval1000_20261103/verification.json).

## 2026-11-02

### Virtual wait_v2: 유예 제거·누적 XYZ 위반·AT 높이 제한 해제

- 기존 wait를 보존하고 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2` env/train·train/test/VNC/eval·registry·GPU4 명령을 추가했다. XY 대기 반경을 r_box+1m로 늘리고 유예0초·root XYZ 속력0.3m/s 초과 누적1초 종료를 적용했다. 정지/반경 이탈은 누적값을 지우지 않으며 선행 AT 성공은 즉시 에피소드 끝까지 해제한다. Progress는 그대로다.
- 새 설정의 단독/선행/PUSH AT는3D20cm를 공통 사용한다. `at_z_tolerance=.2`로 추가1mm 제약을 없앴고 ON_TOP 자체 높이 조건은 유지했다. 기존 성공 함수를 재사용하며 AT 설정을 checkpoint 계약에 반영했다. 기존 설정은 XY·유예·연속 모드를 그대로 사용한다.
- CPU174통과·1skip. GPU4에서4명·1024환경 짧은 학습2epoch/262144frame 완료, Stage1 26k165tensor 전이·encoder73tensor/입력 RMS 동결·head6tensor 갱신·finite 확인. 저장본8환경×6preset 실제16step 및 reset 주입 검사에서 수직 이동/누적 종료·AT 성공 즉시/영구 해제·부분 reset·독립 행동·정식평가 bypass 통과. 기존 YAML 해시 및4개 학습 프로세스를 보존했다. [검증](output_etc/check/before_wait_v2_20261102/report.md).

### wait1500 실제 대기 종료 장면 촬영

- 앞부분 부족·카메라 혼잡에 대한 사용자 피드백으로 같은 GPU5·wait1500에서 한 후행의 에피소드 시작부터 연속 재촬영했다. 선행/후행 색 구분·목표/대기 반경의 설명용 표시·추적 카메라를 적용했으며 영상은0.5배속이다. 실제 시간 SIT3.97초·CLIMB12.30초 후 대기 종료, ON_TOP15초 동안 대기 종료 없음을 기록했다. 세 장면 모두 반경 진입 전부터 시작하고 에피소드가 이어짐·checkpoint/기존4개 학습 보존을 확인했다. [연속 영상 검증](output_etc/check/wait_capture_context_20261102/verification.json).
- 사용자 요청으로 GPU5에서 wait1500을 행동별4명·8환경·8토큰·seed2 deterministic 일반 실행으로 촬영했다. 상태/속도 주입 없이 실제 카메라 화면에 거리·root XY 속도·유예/연속 위반 시간을 표시했다. SIT·CLIMB는 연속 위반1초와 장면 reset을 함께 확인했고, ON_TOP은900step 동안 대기 종료가 없었으며 최대 연속 위반0.90초였다. [영상·판정 기록](output_etc/check/wait_capture_20261102/verification.json).
- 촬영기만 output_etc/check 아래 추가했으며 카메라 월드 좌표를 보정했다. 최초 촬영의 시뮬레이터 정리 단계 오류 후 재촬영을 정상 완료했다. 체크포인트 해시와 기존4개 학습 PID 생존을 확인했으며 운영 코드·대기 조건은 변경하지 않았다.

### Virtual wait1500 GPU5 Plane Ever 평가

- 사용자 요청으로 GPU5에서 wait1500을4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode로 평가했다. 정식 규약상 대기 위반 종료는 제외하고 기존 개인 낙상을 적용했다. wait500/1000·virtual1500과 비교하고 전체/후행 경로 짧음/선행 경로 짧음 모두 행동별 동일 쌍 선행 AT를 함께 집계했다.
- 전체 후행 동시 Plane Ever/같은 쌍 선행 AT는 SIT11.67/86.98%, CLIMB8.45/63.22%, ON_TOP10.82/72.89%, 전체10.48±0.40/73.58±1.38%였다. 후행 경로가 짧은 집단은11.62/86.07·4.28/62.38·10.18/69.13%, 선행 경로가 짧은 집단은12.47/89.80·25.10/66.49·13.19/86.74%였다. 단독 SIT90.40·AT92.39·ON_TOP93.00%였다. [추세·virtual1500·거리별 표](output_etc/check/wait_eval1500_20261102/report_plane.md).
- 3회 모두1200step·CPU float64/GPU/JSON 일치·trace finite·float32/64 누적 차이0, 모든seed 기존 virtual과 초기 상태 정확 일치, 같은 쌍 선행 AT 상한·거리 분할 합 재현·encoder73tensor/입력 RMS 동결·checkpoint/운영 파일223개 해시 보존을 통과했다. 기존4개 학습 PID도 유지했다. [검증](output_etc/check/wait_eval1500_20261102/verification.json).

### BEFORE 실제 근거리 추세·wait와 같은 쌍 선행 AT 보고

- 반대 집단인 선행 접근+운반 경로 < 후행 수행 경로도367쌍(21.73%, SIT83·CLIMB80·ON_TOP204)으로 집계했다. 동거리0건이며 기존 후행 근거리 집단의 정확한 여집합이다. 최신 CLIMB 후행/같은 선행AT는 source5 45.18/84.54%, virtual23.30/85.93%, asym25.77/89.32%, wait1000 23.74/64.30%였다. 실제 완료 순서가 아닌 초기 XY 경로 합 분류다. [전체 checkpoint·평균/std·분모](output_etc/check/goalroute_leader_near_trends_wait_20261102/report.md).
- 기존 goalroute 세 모델의500/1000/1500/중간6000·5500/7000/최신9000·8500 저장본18개와 wait500/1000의 평가2개를 새 시뮬레이션 없이 CPU 집계했다. 초기 후행 수행 경로 < 선행 접근+운반 경로인1322/1689쌍(78.27%)을 동일하게 분리했으며 SIT265·CLIMB311·ON_TOP746쌍이다. 강제 near50 선택 플래그가 아닌 실제 거리 기준이다.
- 최신 근거리 후행 동시 Plane Ever는 source5 SIT44.98·CLIMB40.17·ON_TOP21.86%, virtual35.13·2.91·20.23%, asym11.16·9.34·16.77%였다. 같은 쌍 선행 AT는 각각83.06·73.97·85.93%,93.19·69.90·85.00%,94.76·85.22·89.55%였다. wait1000은7.55·4.27·9.23%, 같은 선행 AT81.60·57.04·64.98%였다. [전체 추세·평균/std·분모](output_etc/check/goalroute_near_trends_wait_20261102/report.md).
- 모든 평가/seed의 거리·분류 일치, near+나머지의 건수/성공 합과 전체 평가 재현, 기존 근거리 보고서의6개 모델·checkpoint 결과 재현, 후행 동시 Ever≤같은 쌍 선행 AT Ever를 확인했다. 사용자 요청으로 AGENTS/evaluation 보고 규칙에 전체·행동별·거리별·추세의 후행과 같은 분모 선행 AT 동시 보고를 추가했다. 학습·관측/보상·평가 지표 코드는 변경하지 않았다. [검증](output_etc/check/goalroute_near_trends_wait_20261102/verification.json).

### Virtual wait 500/1000 GPU5 Plane Ever 평가

- 사용자 지정 GPU5에서 `_wait`의500·1000 고정 checkpoint를 각각4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode로 평가했다. 기존 정식 규약에 따라 대기 이동 종료를 제외하고 개인 낙상만 적용했으며 기존 virtual500/1000 결과를 비교용으로 재사용했다.
- wait500→1000 후행 동시 Plane Ever는 SIT6.83→7.79%, CLIMB6.83→8.22%, ON_TOP7.79→10.07%, 선행 AT64.85→69.13%였다. 단독 SIT89.64→86.87%이며 기존 virtual1000의79.63%보다는 높았다. virtual1000 대비 후행 SIT/ON_TOP·선행 AT는 낮았다. [500/1000·기존 virtual 비교표](output_etc/check/wait_eval500_1000_20261102/report_plane.md).
- 6회 모두1200step·CPU float64/GPU/JSON 일치·trace finite·float32/64 누적 판정 차이0, 모든seed의 기존 virtual 초기 상태 정확 일치, encoder73tensor/입력 RMS 동결·checkpoint·운영 파일223개 해시 보존을 통과했다. 기존4개 학습 PID도 유지했다. 정지 행동 자체의 진입/속도 지표는 이번 성공률 평가에 포함하지 않았다. [검증](output_etc/check/wait_eval500_1000_20261102/verification.json).

### goalroute 세 모델 최신9000/8500/9000 GPU2 Plane Ever 평가

- 사용자 지정 GPU2만 사용해 시작 시 최신 고정 저장본 source5 9000·virtual8500·virtual_asym9000을4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode로 정식 평가했다. 각 checkpoint의 기존 random_scenario/near50·크기·거리 설정을 유지했으며 AT9 override와 새 wait 실험은 적용하지 않았다.
- 후행 동시 Plane Ever는 SIT45.17/32.29/12.91%, CLIMB41.16/7.15/12.82%, ON_TOP22.31/20.31/18.24%(source5/virtual/asym 순서)였다. 전체31.39±1.11/19.72±0.77/15.88±0.91%, 단독 SIT93.85/92.36/86.48%, 선행 AT84.20/85.21/90.67%였다. [행동별·단독·선행 AT 표](output_etc/check/goalroute_eval_latest_gpu2_20261102/report_plane.md). 모델epoch는 서로 다르다.
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0, 같은seed 세 모델 및 이전7000 평가의 초기 상태 정확 일치, 입력 checkpoint·운영 파일 해시 보존을 통과했다. 기존 세 학습 PID 생존·학습 scalar finite도 확인했다. [입력 조건](output_etc/check/goalroute_eval_latest_gpu2_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval_latest_gpu2_20261102/comparison.json), [검증](output_etc/check/goalroute_eval_latest_gpu2_20261102/verification.json).

### source5·virtual AT 지정9종·64환경 진단 평가

- GPU4에서 source5 9000·virtual8500을 각각4명·8토큰·64혼합환경·seed2·1200step·deterministic Plane Ever로 평가했다. 평가 프로세스에서9종 제안 크기로 기존 크기 호환 역할 sampler를 한 번 실행해 첫 episode 역할을 고정하고, 실제 단독/선행 AT 물리 자산만9종으로 교체했다. ON_TOP payload 등 비AT 크기·기존 near50/거리/충돌 배치를 유지하며 운영 코드·checkpoint는 변경하지 않았다.
- 단독 AT96.30/96.30%(각26/27), BEFORE 선행 AT87.67/84.93%(64/73·62/73)였다. 후행 SIT/CLIMB/ON_TOP별 선행 AT는76.19/82.35/97.14%와85.71/76.47/88.57%였다. 전체 AT100건에서9종 모두 포함됐고 단독0.42³는 표본0건이다. 작은1seed 진단이며 모델epoch가 다르다. 두 모델 초기 상태 정확 일치·실제 AT 자산/관측 크기 일치·비AT 크기 보존·CPU/GPU/JSON 채점 일치·trace finite·checkpoint/운영 파일 해시 보존을 확인했다. 최초 공용 bank 전체 교체 예비 결과는 제외했다. [행동별·크기별 결과](output_etc/check/at9_eval64_20261102/report.md), [검증](output_etc/check/at9_eval64_20261102/verification.json).

### Virtual goalroute 후행 대기 이동 종료 실험

- 기존 virtual_goal5_goalroute를 보존하고 `_wait` 환경/학습 YAML·train/test/VNC/eval·출력/checkpoint 계약을 추가했다. 새 설정만 BEFORE 후행의 XY 거리(root 또는 ON_TOP payload)가 공유 상자 반대각선+0.5m 이내면 진입을 유지하고,1초 유예 후 root XY 속도>0.3m/s가1초 연속이면 후행 terminate/장면 reset을 적용한다. 선행 AT의 같은 스텝 raw_own_success로 episode 내 영구 해제하며 부분 reset은 해당 환경만 초기화한다. 기존 Progress·보상/Gate·낙상·샘플링·모델을 유지하고 정식 Plane 평가는 기존 규약상 새 종료를 제외한다. config/structure/USAGE/상세 문서를 갱신했다.
- GPU4만 사용해4명·1024환경·8토큰 CHECK를 epoch3/frame393216까지 실행했다.26k165tensor 전이·encoder73tensor/입력 RMS 동결·head6tensor 갱신·모델/optimizer/지표 finite, CPU212개 통과(기존1개 skip·전이3개 제외),6개preset의8환경16step actor/critic·Virtual·종료 버퍼 연결·동시 성공 해제·latch·부분 reset을 확인했다. 강제 SIT/CLIMB 검사는 호환 bank로 별도 초기화했다. 기존 세 goalroute YAML6개 해시·문서 링크/21개 설정 목록·wrapper 구문/diff 검사도 통과했다. 본학습·학습된 대기 성능·장기 속도는 미검증이다. [검증 보고서](output_etc/check/before_wait_20261102/report.md).

### goalroute 세 모델 7000 checkpoint 4명 Plane Ever 평가

- 사용자 지정 GPU5·6에서 세 실험 모두 epoch7000/frame917504000 저장본으로4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode를 정식 평가했다. random_scenario/near50을 포함한 저장 설정을 유지하고 기존 학습·checkpoint·운영 코드는 보존했다.
- BEFORE 후행 동시 Plane Ever는 SIT34.55/29.61/12.03%, CLIMB31.40/6.36/7.67%, ON_TOP18.82/19.03/16.48%(source5/virtual/asym 순서)였다. 전체25.02±1.35/18.29±2.13/13.54±0.79%, 단독 SIT96.94/92.34/82.69%, 선행 AT84.63/83.29/87.65%였다. [행동별·단독·선행 AT 표](output_etc/check/goalroute_eval7000_20261102/report_plane.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0·세 모델의 같은seed 초기 상태 정확 일치·입력 checkpoint 및 운영 파일281개 해시 보존을 통과했다. 기존 세 학습 PID 생존도 확인했다. [입력 조건](output_etc/check/goalroute_eval7000_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval7000_20261102/comparison.json), [검증](output_etc/check/goalroute_eval7000_20261102/verification.json).

### BEFORE 세 모델 8명·16토큰·64환경·1회 평가

- 추가 GPU/시뮬레이션 없이 기존4명(6000/5500/6000·512환경×3seed 평균)과8명(7000/6500/7000·64환경×seed2)의 Plane Ever 차이를 집계했다. checkpoint·평가 규모가 달라 순수 인원 수 효과로 해석할 수 없음을 명시했다. [4명→8명 행동별 차이](output_etc/check/before8_eval64_20261102/comparison_4_to_8.md), [집계 JSON](output_etc/check/before8_eval64_20261102/comparison_4_to_8.json).
- 사용자 지정대로 GPU5·6만 사용해 source5 7000·virtual6500·asym7000 저장본을8명·16토큰·64환경·seed2·첫 episode1회·1200step·deterministic으로 평가했다. 저장된 random_scenario/near50 설정을 유지하고 기존 Plane Ever 기하 판정·개인 낙상 freeze를 재사용했다. 학습4명에서 인원을 바꾼 진단 평가이며, 정식 general의 인원 변경 금지 규칙은 운영 코드에서 유지하고 확인용 driver만 별도 경로에 뒀다.
- 후행 동시 Plane Ever는 SIT21.05/15.79/13.16%(각38쌍), CLIMB30.77/2.56/2.56%(각39쌍), ON_TOP10.11/13.48/8.99%(각89쌍), 전체17.47/11.45/8.43%(각166쌍)였다. 선행 AT81.33/77.71/79.52%이며 단독 행동은 양쪽 BEFORE 역할을 제외해 집계했다. 1seed이므로 표준편차를 계산하지 않았다. [행동별·단독·선행 AT 표](output_etc/check/before8_eval64_20261102/report_plane.md).
- 세 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·모델 간 초기 상태 정확 일치·checkpoint 및 운영 파일281개 해시 보존을 확인했다. 기존 학습 프로세스도 유지됐다. 8명 초기화에서 기존 PUSH sequential 경로 배치가 오래 걸렸으며 이번 예산 수정에서 해당 물리 sampler를 최적화하거나 검사를 완화하지 않았다. [입력 조건](output_etc/check/before8_eval64_20261102/plan.json), [전체 수치](output_etc/check/before8_eval64_20261102/comparison.json), [검증](output_etc/check/before8_eval64_20261102/verification.json).

### BEFORE PUSH viewer의 가변 인원·물체 예산 지원

- 추론 validation을 CLI와 task 생성까지 전달해 BEFORE PUSH viewer/test/VNC에서 N≥1·M≥N을 허용했다. Alias 포함 `N+ON_TOP 수`가 예산을 초과한 과제 조합만 최대1024회 재샘플링하며 불가능한 고정 ON_TOP preset은 즉시 거절한다. 학습/resume의 M≥2N·기존 충분한 예산의 샘플링/RNG·모델/보상/Gate/계약과 정식 general 규약은 유지했다. config/USAGE/structure를 갱신했다.
- CPU132개 통과(기존 checkpoint 전이1개 제외), 수정 전후 CPU5개·CUDA4개 사례의 그래프/역할/template/RNG 정확 일치, GPU5의 작은 토큰 예산2560개 장면 검사를 통과했다. GPU6에서 기존 asym6000을8명·1환경·12토큰·seed2·32step으로 실제 VNC 실행/종료하고 CUDA/Vulkan GPU 일치를 확인했다. 같은 GPU에서 실제 test wrapper의1명·1토큰/source5 7000과8명·8토큰/virtual6500을 각각1환경·seed2·16step으로 완료했다. 보조8환경 다중preset 검사는 random_scenario16step·부분reset 이후 all_solo의 기존 PUSH 배치 지연으로 해당 보조 프로세스만 중단했으며 전체 preset 통과로 간주하지 않는다. 기존 학습·참고 jointpush 레포는 보존했다. [검증/제한](output_etc/check/before_viewer_budget_20261102/report.md), [이번 구현 diff](output_etc/check/before_viewer_budget_20261102/implementation_diff.patch).

### 최신 goalroute 세 BEFORE 실험 Plane Ever 평가

- 실제 초기 후행 수행거리가 선행 접근+운반거리보다 짧은1322/1689쌍(78.27%)을 CPU로 분리 집계했다. 후행 SIT45.36/22.51/10.51%, CLIMB24.15/2.60/1.00%, ON_TOP19.66/19.57/17.55%였다. 강제 near 표시는 저장되지 않아 실제 거리로 분리했으며, 모델별 분류 동일·near/far 합의 기존 행동/seed별 건수 정확 재현을 확인했다. [가까운 쌍/나머지 표](output_etc/check/goalroute_eval_latest_20261102/report_near_plane.md).
- 평가 시작 시 최신 고정 저장본인 source5 6000/frame786432000, virtual 5500/frame720896000, virtual_asym 6000/frame786432000을 각각512환경×4명×seed0·1·2×1200step deterministic 첫 episode로 평가했다. 직전 허용 GPU0·3 중 GPU0은 여유 메모리 약2GB라 GPU3만 사용했고 기존 프로세스·학습 설정은 유지했다. near50을 포함한 checkpoint의 학습 샘플링을 사용하며 모델별 epoch가 다르다.
- BEFORE 후행 동시 Plane Ever는 SIT44.68/21.81/11.46%, CLIMB25.96/6.17/6.09%, ON_TOP21.31/20.69/18.27%였다. 단독 SIT96.55/92.31/85.80%, 선행 AT82.84/83.00/84.72%였다(source5/virtual/asym 순서). [행동별 표·선행 AT](output_etc/check/goalroute_eval_latest_20261102/report_plane.md), [1500→최신 비교](output_etc/check/goalroute_eval_latest_20261102/comparison_1500_to_latest.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0·입력 checkpoint 해시 보존을 통과했다. 세 모델과1500↔최신의 같은seed 초기 상태가 정확히 같았고 운영 코드/설정281개 해시도 같았다. 현재 세 학습 PID 생존·누적 학습 지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval_latest_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval_latest_20261102/comparison.json), [전체 검증](output_etc/check/goalroute_eval_latest_20261102/verification.json).

## 2026-11-01

### 최신 goalroute 세 실험 1500 checkpoint Plane Ever 평가

- 추가 CPU 분리 집계에서 초기 후행 수행거리가 선행 접근+운반거리보다 짧은 쌍은1322/1689개(78.27%)였다. near50은 약50%에 짧은 거리를 강제하며 나머지 무제약 표본도 가까울 수 있다. 저장되지 않은 강제 near 선택 표시 대신 실제 초기 거리로 분리했고 세 모델/seed별 분류 동일·near/far 건수 합의 기존 결과 정확 재현을 확인했다. [가까운 쌍/나머지 Plane 표](output_etc/check/goalroute_eval1500_20261101/report_near_plane.md).
- 사용자 지정 GPU0·3에서 세 goalroute 실험의 epoch1500/frame196608000 저장본을 각각512환경×4명×seed0·1·2×1200step deterministic 첫 episode로 평가했다. BEFORE는 선행AT·후행 동시 Plane Ever, 단독은 혼합 장면에서 BEFORE 양쪽 역할을 제외한 개인 Ever다. 기존 프로세스·학습 설정은 유지했다.
- BEFORE 동시 전체 평균±표본std는 source5 `21.42±0.77%`, virtual `12.27±1.03%`, asym `12.22±1.57%`였다. 후행 SIT38.07/13.79/11.27%, CLIMB19.91/5.13/6.64%, ON_TOP15.86/14.64/14.86%였다. 단독 SIT94.64/82.35/84.62%로1000 대비 각각+3.48/+2.72/-3.82%p였다. [행동별 Plane 표](output_etc/check/goalroute_eval1500_20261101/report_plane.md), [1000→1500 비교](output_etc/check/goalroute_eval1500_20261101/comparison_1000_to_1500.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·checkpoint 해시 보존을 통과했다. 세 모델과1000↔1500의 같은seed 초기 상태가 정확히 같았고 운영 코드/설정281개 해시도 보존됐다. 모든seed GPU0/3 실행과 현재 세 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval1500_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval1500_20261101/verification.json).

### 최신 goalroute 세 실험 1000 checkpoint Plane Ever 평가

- 동일 세 goalroute 실험의 epoch1000/frame131072000 저장본을 GPU5·6만 사용해 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. BEFORE는 선행AT·후행 동시 Plane Ever, 단독은 혼합 장면에서 BEFORE 양쪽 역할을 제외한 개인 Plane Ever다. 기존 학습·checkpoint·운영 코드는 유지했다.
- BEFORE 동시 전체 평균±표본std는 source5 `18.93±0.91%`, virtual `13.39±0.44%`, asym `12.29±1.66%`였다. 후행 SIT25.65/15.07/9.99%, CLIMB22.50/6.75/5.91%, ON_TOP14.90/15.54/15.72%였다. 단독 SIT는91.16/79.63/88.44%로500 대비 각각-5.34/+1.99/+0.03%p였다. [행동별 Plane 표](output_etc/check/goalroute_eval1000_20261101/report_plane.md), [500→1000 비교](output_etc/check/goalroute_eval1000_20261101/comparison_500_to_1000.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·입력 checkpoint 해시 보존을 통과했다. 세 모델 및500↔1000의 같은seed 초기 상태가 정확히 같았고 두 평가의 운영 코드/설정281개 해시도 같았다. 모든seed GPU5/6 실행과 현재 세 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval1000_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval1000_20261101/verification.json), [seed별 변화](output_etc/check/goalroute_eval1000_20261101/delta_from_500.json).

### 최신 goalroute 세 실험 500 checkpoint Plane Ever 평가

- source5_goal5_goalroute·virtual_goal5_goalroute·virtual_asym_goal5_goalroute의 epoch500/frame65536000 저장본을 GPU5·6만 사용해 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. 시작 시 virtual은466epoch여서 해당500 저장본이 생성된 뒤 평가했으며, 각 checkpoint의 goal_route_sum·목표 거리·Virtual/비대칭 설정을 유지했다.
- BEFORE 동시 Plane Ever 전체 평균±표본std는 source5 `14.26±1.15%`, virtual `11.60±0.90%`, asym `10.71±1.23%`였다. 후행 SIT22.22/13.83/9.99%, CLIMB19.76/8.45/9.24%, ON_TOP9.14/12.09/11.57%이며 단독은 BEFORE 양쪽 역할을 제외해 따로 집계했다. [행동별 Plane 표](output_etc/check/goalroute_eval500_20261101/report_plane.md), [seed별 건수·통계](output_etc/check/goalroute_eval500_20261101/plane_tables.json).
- 9개 rollout 모두1200step을 완료하고 CPU float64 재채점·GPU 누적·JSON 건수가 일치했다. trace finite·float32/64 누적 판정 차이0·같은seed 초기 상태 정확 일치·입력 checkpoint 해시 보존과 모든seed GPU5/6 실행 기록을 확인했다. 운영 코드/설정281개 해시와 기존 세 학습 PID·지표 finite도 유지됐다. [입력/실행 조건](output_etc/check/goalroute_eval500_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval500_20261101/verification.json).

### 이전 source5·virtual·virtual_asym 5000 checkpoint Plane Ever 평가

- 이전 세 run의 epoch5000/frame655360000 저장본을 GPU5·6에서 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. checkpoint에 저장된 기존 relative_sum 배치를 사용했고 현재 goalroute 학습은 유지했다. BEFORE는 선행AT·후행 같은 frame의 동시 Plane Ever, 단독은 BEFORE 선행·후행을 모두 제외한 개인 Plane Ever다.
- BEFORE 동시 전체 평균±표본std는 source5 `15.12±1.78%`, virtual `8.12±0.68%`, asym `9.10±1.16%`였다. 행동별 SIT33.85/7.59/9.72%, CLIMB13.82/2.07/2.25%, ON_TOP8.85/10.85/11.66%로 source5의 SIT/CLIMB 우위가 유지됐다. [Plane 행동별 표](output_etc/check/eval5000_20261101/report_plane.md), [세부 수치](output_etc/check/eval5000_20261101/plane_tables.json).
- 9개 rollout의 CPU float64 재채점·GPU 누적·JSON 건수가 일치했고 trace finite·float32/64 누적 판정 차이0·동일seed 초기 상태 정확 일치·입력 checkpoint 해시 보존을 확인했다. 비교 코드 삭제 후 운영 코드/설정281개 해시도 유지됐고 현재 세 goalroute 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/eval5000_20261101/plan.json), [전체 검증](output_etc/check/eval5000_20261101/verification.json).

### 사용하지 않는 최적화 비교·계측 코드 정리

- 사용자 요청으로 `output_etc/check/push_opt123_20261101`·`reset_share_20261101` 아래 일회성 Python17개와 bytecode9개만 삭제했다. 운영 경로의 참조·실행 프로세스가 없음을 확인했고 코드/설정281개 해시를 전후 비교해 전부 보존했다. 현재 학습·원본·모든 checkpoint·보고서·JSON·로그는 유지했다. 보고서의 삭제 코드 링크를 정리하고 삭제 표시를 추가했다. [삭제 목록·검증](output_etc/check/eval5000_20261101/cleanup_result.json).

### source5 goalroute 실제 학습의 reset 시간 비중 계측

- GPU5·원본 source5_goal5_goalroute·4명/1024환경/8토큰·seed42·Stage1 26k에서 CHECK40epoch(1280step)를 실행했다. 계측은 별도 프로세스에서 reset 앞뒤 동기화와 후보 sampler CUDA event로 추가했고 운영 코드·모델/보상/샘플링 설정·기존 프로세스는 유지했다. 시작 전체 초기화와 첫2epoch는 집계에서 제외했다.
- 측정38epoch275.584초 중 reset39.994초(14.51%), 나머지 rollout80.372초(29.16%), PPO/AMP 업데이트·배치 준비 등155.217초(56.32%)였다. reset 내부 PUSH 후보 sampler는7.369초로 전체2.67%·reset18.42%였다. reset465회 중464회는1–5환경,1회는550환경이었고 호출당 환경 수 중앙값1이었다. 큰 전체 초기화의63–66% 단축을 학습 중 작은 부분 reset에 그대로 대입할 수 없으며, 이번 후보 sampler 최적화가 줄일 수 있는 전체 시간 비중은 제한적이다.
- 40epoch/frame5242880 완료·학습 지표/모델/optimizer/정규화 finite·encoder73 tensor/입력 RMS 동결·운영 sampler 해시 보존을 확인했다. 단일seed 초기 학습과 동기화 계측의 근사 비중이며 성숙한 정책·세 모델 동시 학습이나 최적화 적용 후 처리량을 보장하지 않는다. [보고서](output_etc/check/reset_share_20261101/report.md), [전체 집계](output_etc/check/reset_share_20261101/verification.json), [checkpoint 검사](output_etc/check/reset_share_20261101/checkpoint_validation.json).

### source5 goalroute PUSH sampler 최적화1·2·3 분리 비교

- 사용자 요청으로 GPU5만 사용해 source5_goal5_goalroute 하나에서 원본/실제 PUSH 경로 벡터화/반경 재사용/후보 메타데이터 공유를 누적 비교했다. 구현은 `output_etc/check/push_opt123_20261101/variants.py`의 프로세스별 교체로 격리했고 운영 코드·세 설정·기존 프로세스는 보존했다. 최종 전체 장면 validator는 원본을 유지한다.
- CPU/GPU 각각 경로147,456건·경계30건·sampler54건의 판정/후보/미해결 index/입력/RNG 동일성을 통과했다. 교차32,768회 reset의 상태·관측이 정확히 같았고, 4명/1024환경/8토큰 스트레스65,596회에서 거리/near/충돌/경로/각도/성공 환경 보존·부분 reset을 통과했다. 최종 실패0·최대10라운드·버전별 fallback2,216건 전부 해결·집계 동일이었다. 네 버전의32환경5preset×16step Actor/Critic·alias·보상/관측 finite·부분 reset·모델 보존과 각5epoch CHECK 학습의26k165 tensor 전이·encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.
- 같은 GPU에서 순서를 교차한6개 측정 seed의 전체 reset 중앙값은 원본1.749초, 1번0.645초, 1+2번0.629초, 1+2+3번0.589초였다. 순서를 뒤집은 최종 단일seed 학습에서 첫 epoch 제외 중앙값은 각각6.890/7.003/7.239/7.130초로 전체 PPO 속도 개선은 확인되지 않았다. 후보 중간 메모리는108.0→179.5/162.8/146.5MiB로 늘었고 전체 PyTorch 학습 peak는 약13.17GiB였다. 장기 처리량·정책 성능은 미검증이므로 정식 반영은 보류한다. 기존 mix226 전용 물리 검사 도구가 goalroute 설정을 거절한 것은 비교용 검사만 현재 거리 검증으로 교체해 재검증했다. [보고서](output_etc/check/push_opt123_20261101/report.md), [전체 결과](output_etc/check/push_opt123_20261101/verification.json).

### goal5 기반 목표 중심 BEFORE 배치·수행 거리 near50 실험3개

- 세 goal5 설정에 `_goalroute`를 붙인 환경/학습 YAML과 train/test/VNC/eval을 추가했다. 선행 A→O·O→G 각각1–5m, SIT/CLIMB 후행 G→B 1–10m, ON_TOP G→P·P→B 각각1–5m와 B→G≥1m를 적용한다. near50은 생성된 쌍 절반에 후행 수행 거리 합 < 선행 접근+운반 거리 합을 강제한다. 기존 goal5 여섯 YAML 해시·단독 거리·모델/보상/Gate/Virtual/비대칭 설정을 보존하고 새로운 before 계약·출력·20개 실험의 문서/명령 목록을 분리했다. 기본은 동일 Stage1 26k 전이이며 기존 goal5 저장본과 resume/추론을 혼용하지 않는다.
- near 거리 합의 조건부 CDF와 조건부 각도로 반복 rejection을 피한다. 새 설정에서만 AT 목표·후행 배치를 PUSH 후보 검사보다 먼저 확정하고, PUSH/distractor/단독 ON_TOP 받침처럼 나중에 배치할 물체를 제외한 고정 장면을 선검사한다. 최종 전체 충돌·경로·각도·strict near 판정과64회 외부 재시도는 유지한다. 이 선검사는 중간 검사에서 발견한 불가능한 장면의 긴 PUSH 순차 fallback을 줄이기 위한 것이다.
- GPU5/6에서 최종 코드의 각4명·1024환경·8토큰을16,399회씩, 총49,197회 reset했다. 혼합·전원PUSH·BEFORE 두쌍/한쌍·독립과 부분 reset, 모든 거리·near50·충돌·경로·각도·graph/성공 환경 보존을 통과했고 최종 실패0·최대10라운드·순차 fallback 미해결0이었다. 설정별 후행 SIT2645·CLIMB2461·ON_TOP11594건을 포함했고 강제 near5739/11571쌍(49.60%)이었다. 세 설정의 기하 검사 집계도 동일했다.
- CPU213개 통과(없는 기존 checkpoint4개 skip·기존21k 전이1개 제외), 세 새 wrapper CHECK 학습 epoch2/frame262144·26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·finite를 확인했다. 최종 선검사 포함 asym CHECK 재확인과32환경5preset×16step의 actor/critic·Virtual XYZ/실제 회전·같은 스텝 AT 성공 전환·State/Success/HOLDING/원본 상태 보존, 세 wrapper의4환경32step headless 재로딩을 통과했다.12개 wrapper 구문·20개 config 표/세 명령 목록·문서 링크·diff 검사도 통과했다.
- 같은 GPU6에서 순서를 교차한8쌍 중 warmup2쌍을 제외한 혼합1024환경 reset 중앙값은 기존 goal5 5.246초→goalroute 3.827초(27.1% 감소)였다. 기존 본학습과 함께 실행된 reset 측정이며 장기 PPO 전체 처리량 개선을 보장하지 않는다. 기존 사용자 프로세스를 유지했으며 본학습·새 정책 성능 평가·GUI/VNC 실화면은 실행하지 않았다. [검증 보고서](output_etc/check/before_goalroute/report.md), [전체 수치/실행 조건](output_etc/check/before_goalroute/verification.json).

### 세 모델4k BEFORE·단독 Plane Ever 재집계

- 사용자 요청으로 기존 source5/virtual/virtual_asym epoch4000·512환경×4명×3seed·1200step 결과에서 `region`을 Plane Ever 주 지표로 정리했다. BEFORE는 선행 AT와 후행의 같은 프레임 동시 Ever, 단독은 선행·후행 모두 제외한 개인 Ever다. 기존 Point 표는 보존하고 [Plane 표](output_etc/check/latest3_eval_20261101_132836/report_plane.md)·[seed별 건수/통계](output_etc/check/latest3_eval_20261101_132836/plane_tables.json)를 추가했다.
- 기존9개 CPU/GPU 재채점 결과와 finite를 확인하고 저장 trace의 Plane 성공 건수/JSON 일치·기존 Point 단독 표 재현을 통과했다. 새 시뮬레이션·GPU 작업 없이 재집계했다. 후행 SIT 동시 Ever29.07/5.82/14.12%, 단독 SIT95.37/79.60/93.07%이며 값은 source5/virtual/asym 순서의 seed 평균이다.

### 원본 Stage1 26k 단독 SIT·CLIMB 거리별 평가

- 사용자 지정 주 지표는 **Plane Ever**(기존 코드 `region`)다. 저장·CPU 검증한 결과를 다시 정리했으며 추가 시뮬레이션은 하지 않았다. Stage1 26k는 SIT1–5m83.75%·6–10m90.31%, CLIMB100.00%·99.69%다. [Stage1 Plane 표](output_etc/check/stage1_26k_distance_eval_20261101/report_plane.md), [source5 Plane 표](output_etc/check/source5_distance_eval_20261101/report_plane.md). 아래 Point 수치는 별도 지표로 보존한다.
- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`의 저장 env/sim을 복원하고 Stage1 단독 기준1명·1상자·loco·deterministic·900step·seed0으로 평가했다. 사람 root→자기 상자 중심 초기 XY 거리1·2·…·10m마다64환경, 행동별640시도이며 GPU5(SIT)·6(CLIMB)만 사용했다. 기존 학습 프로세스·코드·설정·원본 checkpoint는 보존했다.
- Point Ever는 SIT1–5m70.00%·6–10m71.88%, CLIMB98.75%·98.12%였다. 앞선 source5는4명·8토큰·1200step이므로 모델만 바꾼 통제 비교와 구분한다. 두 rollout900step·총1280시도의 CPU float64 재채점/GPU 누적/JSON 집계 일치, float32/64 누적 판정 차이0·trace finite·초기 거리 오차 최대0.067mm·checkpoint SHA256 보존을 확인했다. [거리별 결과](output_etc/check/stage1_26k_distance_eval_20261101/report.md), [실행 조건/검증](output_etc/check/stage1_26k_distance_eval_20261101/comparison.json).

### source5 5k 독립 SIT·CLIMB 거리별 평가

- 요청 시점 최신 source5 epoch5000 저장본을 사람 root→자기 상자 중심 초기 XY 거리1·2·…·10m에서 평가했다. 행동·거리별64환경×4명=256시도, seed0·loco·deterministic·1200step이며 전원 독립 SIT 또는 CLIMB로 배치했다. 별도 평가 프로세스에서 거리만 고정하고 기존 학습·설정·보상·충돌 검사를 유지했다. 단일 seed 거리 진단이며 일반 혼합 과제 평가와 구분한다.
- Point Ever는 SIT1–5m 평균83.05%·6–10m82.89%, CLIMB95.00%·95.63%였다. 거리별 결과는 [결과표](output_etc/check/source5_distance_eval_20261101/report.md)에 있다. 두 rollout의1200step·전체5120시도 CPU float64 재채점과 GPU 누적/JSON 집계 일치, float32/64 누적 판정 차이0·trace finite·최대 초기 거리 오차0.168mm·checkpoint SHA256 보존을 확인했다.
- 평가는 GPU0/1에서 완료됐고 해당 프로세스는 종료됐다. 완료 직후 사용자 지시에 따라 이후 GPU 작업은5/6만 사용한다. [실행 조건/검증](output_etc/check/source5_distance_eval_20261101/comparison.json).

### source5 계열 goal5 실험3개 추가

- `stage2_before_push_near50_pair50_source5_goal5`·`stage2_before_push_near50_pair50_source5_virtual_goal5`·`stage2_before_push_near50_pair50_source5_virtual_asym_goal5` 환경/학습 YAML과 train/test/VNC/eval을 추가했다. `before.goal_distance: [1.0, 5.0]`으로 단독 AT·BEFORE 선행 AT의 현재 상자→목표와 단독 ON_TOP의 들 상자→받침을 반경 샘플링하고, PUSH 상자→목표는2–5m로 바꾼다. 사람과 목표 최소1m·기존 Z 계산은 유지하고 새 목표에는 원점±4.5m 제한을 적용하지 않는다. 기존 세 실험의 환경/학습 YAML6개 해시를 보존했다.
- 새 설정에서만 후보 상자 간격·AT 목표와 다른 상자 간격에1mm 여유, PUSH 후보 cone89.98°를 적용한다. 최종 충돌·90° cone 기준은 유지한다. 사용자 선택대로 BEFORE 후행 최종 수행 거리에는 상한을 추가하지 않았고 현재 공유 상자/들 상자 배치·near50·보상·Gate·Virtual 관측·lambda10·모델·AMP/PPO는 유지한다. 새 계약/출력을 분리하고 기본 전이는 모두 기존 실험 실행과 같은 Stage1 26k다. config/structure/USAGE/README·거리 상세를 갱신했다.
- CPU191개 통과(없는 기존 가중치4개 skip·21k 전이1개 제외), 기존 정사각형 sampler 출력/RNG 회귀를 통과했다. GPU0/3/1·각4명/1024환경/8토큰에서 혼합·전원PUSH·BEFORE 두쌍/한쌍·독립과 부분 reset을 각각16,399회, 총49,197회 검사해 최종 실패0·장면 검사 최대9라운드·거리/near/충돌/경로/각도/성공 환경 보존을 확인했다. 각 순차 fallback2건은 기존 장면 재시도로 해결했다.
- 새 wrapper 세 개의 CHECK 학습을 각각 epoch2/frame262144까지 저장했다.26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite와 각4환경32step headless 재로딩을 통과했다. 새 asym 저장본의32환경5preset×16step·부분 reset에서 Virtual XYZ/실제 회전·같은 스텝 AT 성공 전환·State/Success/HOLDING/원본 상태 보존을 확인했다.17개 컨피그 표·세 명령 목록·12개 새 wrapper·문서 링크/구문 검사를 통과했다. 본학습·정규 성능 평가·GUI/VNC 실화면은 실행하지 않았고 기존 학습을 유지했다. [전체 검증](output_etc/check/before_goal5/verification.json).

### PUSH 목표 거리2–5m 검증과 reset 수치 경계 확인

- 기존 source5/virtual/asym 설정·코드와 학습 프로세스는 보존하고, 별도 검증 프로세스의 메모리에서 PUSH 상자→목표만2–5m로 바꿨다. 기존2–10m는 변경 범위를 접근 거리로 한정하면서 유지한 값이며,5m 축소 검증 실패로 되돌렸다는 기록은 확인되지 않았다. 현재 validator는 기존 Stage1 PUSH의2–10m 계약을 요구하므로 새 범위는 별도 계약 지원이 필요하다.
- 막힌4명 PUSH fixture는 기존/축소 각각20seed에서 통과했다. GPU0/3·4명·1024환경·8토큰에서 기존2–10m는16,399회, 무여유2–5m는 혼합/전원PUSH/BEFORE 두쌍·한쌍/독립의 다섯 preset24,591회 reset을 완료했다.64회 한도 소진0·최대 장면 검사12/10라운드였지만, 축소 검사에서는 PhysX 반영 뒤 상자 간격 경계가 약0.007–0.008mm 뒤집힌 두 장면을 발견했다. 탐색 검사의 추가 강제SIT preset은 random asset bank와 호환되지 않아 제외하고 기존 범위는 다섯 preset으로 재검증했다.
- 후보 간격1mm 여유만 준 중간 검사에서는 PUSH 방향44.99903°가 반영 뒤45.00025°로 바뀌는 각도 경계도 발견했다. 검증용 프로세스에서 후보 간격1mm·후보 cone89.98°를 함께 적용하자16,399회 reset과 부분 reset·그래프/성공 환경 보존·거리/near/경로/충돌/각도 최종 검사를 통과했다. 최종 간격50mm·cone90° 및 나머지 판정은 유지했고 최대 장면 검사11라운드였다. 보완안은 실제 코드·config에 적용하지 않았다. [검증 요약](output_etc/check/push_goal5_reset/report.md), [전체 결과/제한](output_etc/check/push_goal5_reset/verification.json).

### source5·virtual·virtual_asym 최신4k 정식 평가

- 재시작한 세 run의 요청 시점 최신 저장본은 모두 epoch4000/frame524288000이었다. 전용 eval wrapper로 각512환경×seed0·1·2·4명·1200step·deterministic 첫 episode를 GPU0/2/3에서 평가했다. 기존 GPU5/6 학습과 checkpoint를 보존했다.
- 점 기준 개인 macro5는 source5 `74.54±0.75%`, virtual `56.00±2.13%`, virtual_asym `61.26±2.28%`였다. BEFORE 동시 성공은 각각 `15.82±0.92%`, `8.41±3.26%`, `11.70±1.92%`, 후행 개인 성공은 `46.15±1.70%`, `9.24±3.73%`, `12.11±1.99%`였다. 실험당1689쌍이며 값은 seed 평균±표본std다. 비대칭은 virtual보다 개선됐지만 현재 시점에서는 원본이 우세하며, 후행 SIT/CLIMB 동시 성공에서 차이가 컸다. 선행 AT는 `67.62/67.00/75.35%`였다.
- 모든 seed에서 세 실험의 저장 초기 상태가 정확히 동일했다.9개 rollout이1200step을 완료하고 CPU float64 재채점·GPU 누적 성공·JSON 집계가 정확히 일치했다. float32/64 판정 차이0·trace finite·입력 checkpoint SHA256 보존을 확인했다. [결과표](output_etc/check/latest3_eval_20261101_132836/report.md), [전체 수치/검증](output_etc/check/latest3_eval_20261101_132836/comparison.json), [입력/실행 조건](output_etc/check/latest3_eval_20261101_132836/plan.json).

### source5 계열 PUSH 초기 후보 검사 배치화

- source5·virtual·virtual_asym의 초기 PUSH 후보128개를16개씩 최대8묶음으로 검사하도록 변경했다. 기존 `sample_layout(..., max_attempts=1)`의 유효성 검사를 재사용하며 환경별 첫 유효 후보·성공 환경 보존·후보 예산·순차 fallback·장면 재시도를 유지한다. 보상·Virtual 관측·lambda10·모델·checkpoint 계약·실행 명령과 다른 설정은 유지한다. 난수 소비 순서가 바뀌므로 같은 seed의 배치는 달라질 수 있다. 상세/structure/USAGE를 갱신했다.
- 관련 CPU173개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU6·1024환경 reset16,399회에서 최종 실패0, 순차 fallback2,658건 해결·성공 환경/graph/near/거리/경로 조건 보존을 확인했다. GPU5·1024환경 CHECK 학습 epoch4/frame524288와 재확인 학습에서 actor encoder73 tensor·입력 RMS 동결, head6 tensor 갱신, 모델/optimizer/지표 finite를 확인했다.32환경5preset의 물리·Virtual XYZ/실제 회전·현재 AT 성공 전환·기존 State/Success/HOLDING 보존과4환경32step headless 재로딩을 통과했다.
- 후보 검사만 비교하면 어려운1환경은 중앙값1.290→0.084초(15.28배), 일반1024환경은0.0259→0.0109초(2.37배)였다. GPU5에 다른 학습이 없는 상태에서 같은 설정·seed42·4epoch의 평균 rollout+update는8.735→7.416초로15.1% 감소했다. 난수 순서가 다른 짧은 단일 seed 측정이며 장기/동시 학습 속도 보장은 아니다. 기존 프로세스를 종료하지 않았고 본학습은 시작하지 않았다. [전체 검증](output_etc/check/push_batched_sampling/verification.json).

### Virtual BEFORE 비대칭 progress 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual_asym` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 virtual에서 `before.asymmetric_progress.lambda_inner: 10.0`만 추가하고 출력·checkpoint 계약을 분리했다. 기본26k 전이·Virtual GTA·물리 배치·모델·PPO/AMP·가중치·State/Success Gate·성공 saturation·HOLDING/독립/PUSH 보상은 유지한다. config/structure/USAGE·상세 설계를 갱신했다.
- `predecessor_at_progress()`에 현재 AT raw 성공과 lambda를 전달해 미완료 시 반경 안쪽 거리만10배 감쇠한다. 거리/root·payload/반경·바깥 감쇠는 유지하고 성공 시 기존 식으로 즉시 복원한다. 호출부는 Runtime과 같은 `interaction_own_success()`로 현재 phi/오차를 판정하며 이전 own_success/achieved를 읽지 않는다. 기존 설정은 lambda0 경로로 동일 계산을 유지하고 새 옵션은 source5 virtual에 한정해 검증한다.
- CPU165개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU5·1024환경 CHECK epoch2/frame262144 저장, actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.32환경7preset의 각16step·부분 reset과 성공/실패3회 전환을 검사해 같은 스텝 반영·성공 이력 무시·State/Success/HOLDING/독립 항 보존·Virtual XYZ/실제 회전 유지를 확인했다.4환경32step headless 재로딩과 eval wrapper help·구문·문서 링크를 통과했다. 기존 본학습은 유지했고 새 본학습·정규 성능 평가·실제 GUI/VNC는 실행하지 않았다. [전체 검증](output_etc/check/before_asymmetric_progress/verification.json).

## 2026-10-31

### source5 Goal-Aligned Virtual Object 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 환경 차이는 `relationGraph.before_virtual_object: at_goal_xyz_v1` 하나이며 source5 배치 보완·보상·Gate를 유지한다. 기본 전이는 연결된26k이고 비교 원본도 같은26k를 지정한다. 출력과 checkpoint 계약을 분리하고 기존 계약은 보존했다.
- `before_object_alias.py`에서 packet의 actor/payload/target으로 선행 AT Goal을 찾아 복사한 scene의 후행 Alias GTA XYZ만 변경한다. `stage2_push_network.py`의 Actor/Critic Encoder 직전에 함께 적용하고 Quaternion·intrinsic·원본 관측/물리 상태/graph·coordination packet은 유지한다. 기존 목표 XY progress가 곧 가상 상자 접근 progress이며 추가 보상이나 구조 변경은 없다. config/structure/USAGE·BEFORE 상세를 갱신했다.
- 관련 CPU140개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU5·1024환경 CHECK epoch2/frame262144 저장,26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.32환경7preset에서 각16step과 부분 reset, SIT/CLIMB/ON_TOP Alias XYZ·실제 회전·원본 상태/graph 보존·정책/critic/AMP finite를 통과했다.4환경32step headless 재로드와 전용 eval의4환경·seed0·1200step을 완료했고 CPU 재채점/JSON 집계 일치·float32/64 차이0·finite·입력 저장본 해시 보존을 확인했다. 소표본 실행 검사이며 실제 GUI/VNC·장기 성능은 검증하지 않았다. 기존 본학습은 유지했다. [전체 검증](output_etc/check/before_virtual_object/verification.json).

### source5 PUSH 후보 소진·reset 좌표 반올림 보완

- 4명 독립 PUSH에서128개 동시 후보가64회 모두 소진되는 종료를 재현했다. `stage2_before_push_near50_pair50_source5`에서 기존 후보가 소진된 환경에만 순차 경로 배치를 적용하고, 유효한 경로를 보존하며 막히면 순서를 다시 뽑는다. 그래프·near 선택·성공 환경은 보존하고 소진된 환경만 기존 장면 재시도로 넘긴다.
- 상태 반영 뒤 좌표가 최대0.061mm 달라져 PUSH 경로·near 경계가 뒤집히는 사례를 확인했다. 해당 설정의 후보 생성에만 경로 여유와 near 거리 차이1mm를 추가했다. 거리 범위·최종 부등식/충돌 조건·기존 near50/mix226 기본 샘플링은 유지하며, 보완된 장면의 공간적 배치 분포는 달라진다.
- 관련 CPU92개 통과(없는 기본21k 전이 검사1개 skip·1개 제외). GPU5·1024환경에서 일반51,200·4명 PUSH5,120·BEFORE 두 쌍/한 쌍 각5,120·독립5,120과 부분 reset15개, 총71,695회를 통과했다. 순차 보완7,951건 중7,949건을 해결하고2건은 기존 장면 재시도로 통과했다. 최종 reset 실패0·그래프/near/성공 환경 보존·기존 거리/경로/충돌 조건을 확인했다. 거리 경계 측정에는 float32 허용0.2mm를 사용했으며 최대 이탈은0.038mm였다.
- CHECK 학습 epoch11/frame1441792 저장·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다. 저장본의4환경32step headless 재로딩과32환경5preset의 물리·alias·AMP·부분 reset 검사를 통과했다. [전체 검증](output_etc/check/reset_1031/verification.json). 본학습은 재시작하지 않았으며 장기 수렴 성능은 검증하지 않았다.

### 1031 checkout 모션·26k 전이 연결 복구

- tmux `1031`의 SIT 모션 `FileNotFoundError`를 확인했다. `stage1_2_base_jointpush`를 참고해 누락된 SIT/CARRY/LOCO/CLIMB 네 모션 링크를 공유 원본에 연결하고, `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 참고 레포의 파일에 연결했다. 기존 PUSH 링크와 외부 원본은 유지했다.
- PUSH 모션 YAML161개·기존 모션 YAML136개 파일 참조가 모두 존재하며, 오류가 난 SIT 배열과26k checkpoint(epoch26000·model165 tensor)의 CPU 로딩을 확인했다. [검증](output_etc/check/data_links_1031/verification.json). 학습은 실행·재시작하지 않았다.

### BEFORE near50 쌍 선택50·접근 최대5m 설정 추가

- `stage2_before_push_near50_pair50_source5` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 near50에서 쌍 생성 확률0.5·일반 사람→상자1–5m·독립 PUSH 접근2–5m만 적용한다. near50·AT/ON_TOP 목표 배치·PUSH 이동2–10m·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지하고 출력·저장본 계약을 분리했다.
- 1024환경 첫 검사에서 고정 주변 배치에 대한 PUSH 후보 소진을 확인했다. 새 설정에서 성공 환경을 보존하고 실패 환경만 기존 장면 재샘플링으로 넘기도록 연결했으며 거리·경로·충돌 조건과 기존 실험 기본값은 유지했다. 전원 독립 PUSH32환경 검사는 재샘플링에 수분이 소요됐다.
- 최종 CPU196개 통과(없는checkpoint4개 skip·21k 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신을 확인했다. 32환경7preset의 거리·near·alias·PUSH 경로·부분 reset과4환경32step headless 재로드를 통과했다. AT 목표 최대14.04m·BEFORE ON_TOP 운반 최대9.08m를 관측해 별도 목표 거리 상한을 추가하지 않았음을 확인했다.
- 일반8환경·seed0·1200step 검사와 CPU 재채점의 건수 일치·float32/64 차이0·finite를 확인했다. 기본21k 파일이 없어 검증과 문서 학습 명령에는26k를 명시했다. [전체 검증](output_etc/check/before_pair50_source5/verification.json). config/structure/USAGE/README·BEFORE 상세를 갱신했다. 본학습·GUI/VNC 실화면·정규512환경×3seed 성능 평가는 실행하지 않았다.

### 개인 목표 solo5를 별도 컨피그로 분리

- 요청에 따라 `stage2_coupled_push_individual_goal`의 단독 거리 상한 변경만 되돌렸다. 공동 확률0.5와 개인 목표·접촉 설정은 유지한다. 단독 접근/PUSH 이동 최대5m는 새 `stage2_coupled_push_individual_goal_solo5` 환경·학습 YAML과 train/test/VNC/eval로 분리하고 출력·저장본 계약도 구분했다. 단독 CARRY AT 목표 높이 보존 수정은 유지한다.
- CPU71개 통과(기존21k 전이1개 제외), 원본/변형 환경 차이는 단독 상한 두 값뿐임을 확인했다. GPU6에서 기존5m 검증 저장본을 새 wrapper로 재로딩하여4환경32step 실행을 완료했다. [CPU 로그](output_etc/check/coupled_individual_solo5_split/cpu.log), [실행 로그](output_etc/check/coupled_individual_solo5_split/reload.log). shell 구문·diff 검사 통과. 본학습은 실행·재시작하지 않았다.

### 개인 목표 실험의 단독 접근·PUSH 이동 최대5m

- 새 개인 목표 컨피그에 `solo_source_max_distance: 5.0`·`solo_goal_max_distance: 5.0`을 추가했다. SIT/CLIMB/CARRY AT/ON_TOP의 사람→상자는1–5m, 단독 PUSH 접근·이동은2–5m다. 공동 접근2–10m·이동2–5m와 CARRY 목표 XY 샘플링은 유지한다. 공통 샘플러의 기본값은 유지하고 현재 설정에만 상한을 전달한다.
- 개인 목표 변환을 공동 구성원에게만 적용하여 단독 CARRY AT의 목표 높이 덮어쓰기를 수정했다. 기존 검사의 목표 벡터 비교를 PUSH에 한정하고 단독 접근 범위·AT 목표/받침 높이 검사를 추가했다.
- CPU53개 통과(기존21k 전이1개 제외). GPU6·확률0.5·1024환경 CHECK epoch2/frame262144 저장, finite·encoder/RMS 동결·head 갱신 확인.32환경8preset의 거리·고정 목표·부분 reset 검사를 통과했고 holding_at의 높은 목표61개를 포함해 받침과 목표 높이 일치를 확인했다. [검증](output_etc/check/coupled_individual_solo5/verification.json). 기존 본학습은 재시작하지 않았다.

### 개인 목표 실험의 공동 선택 확률50%

- `stage2_coupled_push_individual_goal`의 `coupled_probability`를0.8→0.5로 변경했다. 그룹 선택 확률 기준이며4명 기대 인원 비율은 공동63.54%·단독36.46%다. 확률 validator와 설정 비교 검사를 갱신했고 기존 coupled/goal5는0.8을 유지한다.
- 관련 CPU44개 통과(기존21k 전이1개 제외), diff 검사 통과. [로그](output_etc/check/coupled_individual_goal/probability50_cpu.log). 아래 GPU 검증은 변경 전0.8 설정의 결과이며 이번에는 학습을 실행·재시작하지 않았다.

### 개인 목표·공통 4면 coupled 실험 연결

- `stage2_coupled_push_individual_goal` 환경/학습 YAML·train/test/VNC/eval을 추가했다. 공동2–5m 이동 벡터를 초기 분할 토큰마다 적용하고 개인 AT 연결·보상·도착 예외·현재 성공 포화를 사용한다. 공유 중심 목표는 물리 경로와 공통 외곽4면 선택에만 사용한다. 목표 방향60° 후보 중 담당 구역까지 그룹 최대 손 거리가 가장 작은 면을 선택하고 전원 gate도 같은 면으로 검사한다. 별도 `aligned` 조건은 제거했다. 기존 coupled/goal5·독립 행동은 유지하고 저장본 계약을 분리했다.
- CPU83개 통과(없는 기본21k 저장본을 요구하는 기존 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신 확인.32환경8preset의 거리/개인 목표 연결·고정 목표·부분 reset·정책/AMP finite 검사와4환경32step 실제 렌더를 통과했다. [학습 검사](output_etc/check/coupled_individual_goal/training_validation.json), [장면 검사](output_etc/check/coupled_individual_goal/scene/physics_validation.json), [목표 표시](output_etc/check/coupled_individual_goal/viewer/frame_32.png).
- GPU6·32환경·seed0·1200step 평가를 완료하고 개인/그룹 동시 성공의 ever/final 집계를 연결했다. CPU 재채점·JSON 건수 일치, float32/64 개인/BEFORE 지표 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. [재채점](output_etc/check/coupled_individual_goal/eval/general_seed0/rescore.json). config/structure/USAGE/README·상세 설계를 갱신했다. 본학습은 시작하지 않았으며 기존 학습 프로세스는 유지했다.

### PUSH·CARRY AT 목표구 반지름20cm 통일

- 요청에 따라 모든 설정의 AT 목표구를 생성 시 반지름20cm로 통일하고 PUSH 와이어 구를25cm에서20cm로 줄였다. 바닥 링은35cm로 유지하며 reset 중 scale 변경은 추가하지 않았다.
- URDF 시각 반지름0.1m×생성 scale2.0=0.2m·Python 구문·diff 검사를 통과했다. 이번 크기 조정의 viewer 실행 검증은 하지 않았다.

### PUSH viewer 표시 롤백 및 선 표시 재구현

- 사용자가 viewer에서 물체가 날아가는 비정상 동작을 보고해 직전 PUSH 표시 이식 전체를 롤백했다. reset 시 marker scale을 변경하는 코드·표시 모듈·호출부를 제거하고 기존 목표구 표시와 문서를 복구했다. 짧은 관측/렌더 검사만으로 정상 동작을 판단한 기존 검증은 불충분했다. 기존 학습·컨피그 변경은 보존했다.
- 이후 기존 목표구·actor·scale·reset을 유지하고 `add_lines`로 PUSH 목표의 바닥 링·세로선·외곽 와이어 구만 덧그리도록 다시 구현했다. 공유 목표 선은 노란색이며 일반 AT에는 추가 선이 없다. 공통 reset 파일은 변경 전과 동일하다.
- GPU5의 실제 viewer에서5500 정책·4환경·동일 seed의 기존/선 표시 실행을 각각1200step 비교했다. 전체 actor root state·정책 관측·행동이 bitwise 동일하고 최대 차이0·finite다. [궤적 비교](output_etc/check/push_goal_viewer_safe/trajectory_comparison.json). 일반 AT·독립 PUSH·2명 두 그룹·4명 공동의 선 표시만 호출하는 것도 확인했다. [표시 검사](output_etc/check/push_goal_viewer_safe/draw_only_verification.json).

### Coupled goal5 5500 간단 평가

- 최신 정기5500 저장본을 GPU5·32환경·seed0·1200step 첫 episode1회로 평가했다. 공동 PUSH의 실제 상자 목표 도달은2명6/24(25.0%)·3명1/7(14.3%)·4명1/9(11.1%)다. 혼합 장면의 독립 역할은 SIT1/1·CLIMB2/2·AT5/6·ON_TOP5/6·PUSH1/8이며 점/영역 결과는 같다. 사람 번호별 공동 참여 성공은 별도 JSON에 저장했다. [상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/breakdown.json).
- CPU 재채점과 GPU·JSON 건수가 일치하고 float32/64 성공 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. 낙상1/128명이다. 소표본 검사이며 본학습 프로세스는 유지했다. [검증](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/verification.json).
- 같은 궤적의1200step final은 공동2명4/24·3명0/7·4명0/9, 독립 AT4/6이며 나머지 독립 행동은 ever와 같다. 이전 낙상자를 제외하고 final이 ever에 포함됨을 확인했다. [final 상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/final_breakdown.json).

## 2026-10-30

### 공동 PUSH goal5 컨피그 추가

- `stage2_coupled_push_goal5` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 공동 PUSH는 사람→실제 상자 중심2–10m·상자→목표2–5m, 독립 PUSH는 두 거리2–10m다. 기존 coupled YAML은 원래 설정으로 복구했으며 사람 초기 배치·그 외 환경/학습 값은 유지했다. 새 실험의 기본 전이는 제공26k이고 출력·저장본 계약은 분리했다.
- CPU28개 통과(공통 모델26·새 설정 로딩2, 기본21k 파일이 필요한 기존 전이1개 제외). GPU5의1024환경 CHECK가 epoch2/frame262144 저장 후 정상 종료했다. 모델/정규화/optimizer finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. [학습 검사](output_etc/check/coupled_goal5/training_validation.json).
- 전용 test wrapper로 저장본을4환경·32step·4명 공동 PUSH로 headless 재로드했다. 셸/Python 구문·새 YAML/문서 경로를 확인하고 config/structure/USAGE/README·공동 PUSH 상세를 갱신했다. 장기 본학습·수렴 성능·실제 GUI/VNC 화면은 검증하지 않았다.

### 모션 데이터 연결 복구

- loco·sit·climb·carry 모션은 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`의 각 원본으로, PUSH 모션은 `/home/hwanhee/ksh/stage1_2_jointpush/tokenhsi/data/jointpush_motions`로 심링크를 연결했다. 외부 원본은 수정하지 않았다.
- 현재 PUSH 모션 YAML의 파일 참조161개가 모두 존재함을 확인했다. 학습은 실행하지 않았다.

### 공동 PUSH 거리2–5m 분석

- 공동 PUSH의 두 XY 거리를2–5m로 줄이는 안을 검사했다.32환경의41개 공동 그룹에 각각104만 후보를 검사하고, 실패4환경은 위치·회전 구간 검사를 통해 고정된 사람·상자 크기에서 거리/뒤쪽/여유 조건을 동시에 만족할 수 없음을 확인했다. [분석](output_etc/check/coupled_distance_analysis/summary.json).
- 공통 Stage2 CPU23개 통과(기본21k 파일이 필요한 전이1개 제외). 제공26k의165 tensor 전이와 입력 RMS 로드를 확인했다. 이후 사람→상자2–10m·상자→목표2–5m 조합은32환경의 혼합/4명 공동/독립 PUSH reset·짧은 시뮬레이션을 통과했다. 사람 초기 배치는 변경하지 않았고 기존 coupled YAML은 두 거리2–10m로 복구했다.
- 최초 검증은 GPU1에서 실행 후 종료했다. 이후 사용자 지시에 따라 GPU5·6만 사용하며 GPU5에서 같은 배치 오류를 재현했다. [로그](output_etc/check/coupled_distance_2to5/scene_gpu5.log), [초기 배치](output_etc/check/coupled_distance_2to5/scene/reset_failure.json).

## 2026-10-29

### PUSH viewer 목표구 확대

- PUSH 포함 설정의 viewer/VNC/영상 목표구를 반지름3cm에서20cm로 확대했다(지름 약6.7배). 공통 marker 생성부의 표시 scale만 변경했으며 목표 좌표·성공 판정·학습 설정과 비PUSH 설정의 marker 크기는 유지했다.
- coupled1환경의 동일 카메라에서 기존/확대 목표구를 offscreen 렌더링해 가시성을 확인했다. marker scale 변경 전후 정책 관측이 bitwise 동일하고 짧은 시뮬레이션 관측/보상 finite를 확인했다. [확대 화면](output_etc/check/push_goal_marker/render/after.png), [검증](output_etc/check/push_goal_marker/render/verification.json). USAGE에 표시 크기를 기록했다.


### PUSH26k 평가와 Stage2 제로샷 검증

- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 검사했다.5과제/4family·165 tensor·모델/RMS finite를 확인했고 두 입력 PTH의 실행 전후 SHA256은 동일하다. 제공78k는 기존 기준78k와 해시가 같고 새 pth 경로의4환경32step headless 재로드를 완료했다.
- 26k Stage1 PUSH는512환경×3seed·900step 일반3회/1명 행동별 단독15회를 완료했다. 점/영역5과제 macro는 일반 **84.55/88.32%**, 단독 **82.20/86.95%**다. PUSH 점 성공률은 일반87.12%·단독88.61%로21k보다 높지만 다른 과제의 하락으로 전체 점 macro는 낮았다.21k와18개 초기 장면의9개 필드가 bitwise 일치한다.
- near50·coupled는26k tensor165개를 그대로 복사하고 공통 coordination identity/Linear bias0인 제로샷 저장본을 각각 만들었다. 추가 학습0회·두 모델 tensor 동일을 확인했다. 각16환경·4명·1200step 실행과4환경32step headless viewer 재로드를 완료했다. BEFORE ON_TOP 쌍/4명 공동 PUSH viewer 명령을 USAGE에 추가했다. 관찰용 파일에는 optimizer가 없고 기본 YAML 전이21k는 변경하지 않았다.
- 총20개 trace의 CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite를 확인했다. [리포트](output_etc/check/push_26k/report.md), [검증](output_etc/check/push_26k/verification.json). Stage2 결과는 작은 제로샷 실행 검사이며 학습된 Stage2 성능·실제 GUI 화면 검사는 아니다. 기존 원본/프로세스는 보존했다.


### BEFORE PUSH near50 거리 변형 추가

- `stage2_before_push_near50` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. `stage2_before_push`의 mix226 대신 참고 설정의 `relative_sum` near50 거리만 적용했다. 기존 모든 YAML은 보존하고 공통 모델·PUSH/독립 행동 비율·4명/1024환경/8토큰/1200step·loco·21k 전이는 유지했다. RSI433과 attention mask는 이식하지 않았다.
- near 모드는 BEFORE 쌍의50%이며 SIT/CLIMB는 후행→받침 거리가 선행→받침보다 작고, ON_TOP은 후행→받침+후행→payload 거리 합이 선행→받침보다 작다. 나머지50%는 original 샘플링이다. validator·viewer/eval의 새 이름/모션/task 경로·checkpoint 혼용 거부를 연결했다.
- CPU313개 통과.1024환경 CHECK epoch2/frame262144 저장·모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다.32환경5preset에서 near 부등식·PUSH 경로/물리·alias 관측·부분 reset·가중치 불변을 확인했고4환경32step headless viewer와16환경×seed0·1200step 일반 평가를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever 차이0·finite다. [검증](output_etc/check/before_push_near50/verification.json).
- config/structure/USAGE/README·BEFORE PUSH 상세에 설정과 명령을 갱신했다. 장기 본학습·512환경×3seed 성능·GUI/VNC 실화면은 실행하지 않았다.


### 공통 Stage2 alias 전처리와 coordination 초기화 수정

- 공통 `Stage2PushBuilder`의 alias 처리를 config 플래그 대신 task packet의 BEFORE 의존 관계로 결정하도록 통일했다. encoder용 복사본만 변환하고 복제 물체 token feature/pose·원본 packet·coordination relation은 보존한다. coupled·독립 packet은 그대로 통과한다. 기존4과제 BEFORE 구현은 유지했다.
- 새 coordination의 모든 Linear bias를 원본처럼0으로 초기화했다. 초기 출력 identity와 Stage1 PUSH165 tensor 전이는 유지하며 기존 학습 PTH 로드 시 저장된 bias를 덮어쓰지 않는다. coupled 제로샷 PTH를 같은 경로에 새 초기화로 재생성하고 이전 관찰용 저장본은 `output_etc/check/stage2_common_fix/previous_zeroshot.pth`에 보존했다.
- CPU305개 통과. BEFORE SIT/CLIMB/ON_TOP·두 쌍·coupled·solo의 같은 가중치/입력에서 config에 무관하게 actor/critic 출력 bitwise 일치, 기존 encoder 동작·복제 feature/pose·원본 packet 보존을 확인했다. 실제 GPU BEFORE 관측과 기존 학습 가중치의 교차 실행도 action/critic 최대 차이0이다.
- 두 설정 모두4명·1024환경 CHECK epoch2/frame262144 저장, 모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다. BEFORE32환경5preset의 alias 관측/부분 reset/finite 검사와 재생성 제로샷4환경·32step headless 재로드를 완료했다. 이전/새 제로샷의 coupled·solo 출력은 동일하다. [검증](output_etc/check/stage2_common_fix/verification.json). 구조/설정 상세 문서를 갱신했고 장기 성능·실제 GUI 화면은 검증하지 않았다.


### Coupled 제로샷 viewer 저장본

- 제공 Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 초기화한 `output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth`를 만들었다. 추가 optimizer step0·tensor 일치·identity 초기화를 확인했다. 원본 PTH는 보존했다.
- `TASK_GRAPH=all_coupled`로4환경·32step headless test wrapper 재로드를 완료했다. 실제 GUI 화면은 실행하지 않았다. 제로샷 viewer 명령을 USAGE에 추가했고 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.


### 공통 Stage2 PUSH 모델과 coupled 이식

- `stage2_coupled_push` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 원격 `stage1_2_jointpush_st`의 coupled straight·밀도3·인원별 공동 상자/구역·그룹 gate·goal saturation을 이식했다. 독립 행동과 AMP 혼합은 현재 Stage1 PUSH 설정(AT25/ON_TOP20/SIT10/CLIMB15/PUSH30)을 읽도록 바꾸고 미사용 Stage2 RSI/adapter 스택은 가져오지 않았다.
- `stage2_before_push`와 coupled는 동일 `Stage2PushBuilder`·관측1272/AMP1330·5과제·NONE/SELF/BEFORE/COUPLED4종 사람 relation·6N task/group suffix를 사용한다. tensor 이름·크기가 같고 Stage1 PUSH21k의165 tensor를 그대로 전이한다. 기존 BEFORE PUSH의5N 관측 검사 저장본은 새 모델에 resume하지 않는다.
- 사용자 기준대로 Stage1은900step을 유지하고 모든 Stage2 학습/일반 평가/viewer 기본 시간을1200step으로 변경했다. 기존4과제 BEFORE 모델은 보존했다. coupled 평가의 PUSH 성공은 가상 구역이 아닌 실제 전체 상자 중심에서 계산한다.
- CPU292개 통과. 두 Stage2 모두4명·1024환경 CHECK가 epoch2/frame262144 저장 후 종료했고 coupled resume은 epoch3/frame393216까지 완료했다. actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신·모델/RMS/optimizer finite를 확인했다.
- coupled32환경8preset·BEFORE32환경5preset의 배치/구역/AMP/부분 reset/가중치 불변을 확인했다. 두 설정 모두 일반32환경×seed0·1200step 및 headless viewer4환경·32step 재로드를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite다. [검증 보고](output_etc/check/coupled_integration/report.md), [전체 결과](output_etc/check/coupled_integration/verification.json).
- config/structure/USAGE/README·BEFORE 상세와 [공통 Stage2/coupled 상세](markdowns/reference/stage2_coupled_push.md)를 갱신했다. Stage1 YAML·원본 checkout·외부 PTH·기존 프로세스는 보존했다. 장기 본학습·정규512환경×3seed 성능·GUI/VNC 실화면은 검증하지 않았다.


### Stage2 BEFORE solo PUSH 추가

- `stage2_before_push` 환경/학습 YAML과 전용 train/test/VNC/일반 eval을 추가했다. 독립 역할만 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%를 사용한다. 기존 0.8 쌍 구성·선행 AT·크기 호환 후행 SIT/CLIMB/ON_TOP 균등 선택·mix226·공유 alias·AT gate/progress는 유지한다. 기본4명·1024환경·8토큰·900step·모두 loco다.
- PUSH 전용 큰 상자 bank를 추가해 기본20개 물리 후보에서 배정하며 일반 distractor/BEFORE 받침으로 섞이지 않게 했다. Stage1 PUSH의 밀도3·167종·cone90·거리·접촉 가중치를 적용하고 다중 사람 경로/활성 장애물 검사를 추가했다. 정책5과제·AMP4family를 기존 BEFORE coordination에 연결해 기본21k PUSH에서 tensor165개를 그대로 전이한다. 기존5개 환경/학습 YAML은 변경하지 않았다.
- CPU280개 통과. 최종 CHECK1024환경은 epoch2/frame262144, resume은 epoch3/frame393216 저장을 완료했다. actor encoder73 tensor·입력 RMS 동결, action head6 tensor 갱신, 모델/정규화/optimizer finite를 확인했다. [학습·재개](output_etc/check/before_push/training_validation.json).
- 32환경7개 preset에서 물체 alias/숨김·거리·밀도/마찰·PUSH 경로·AT progress·AMP·부분 reset·가중치 불변을 확인했다. 최종 저장본 일반32환경×seed0·900step과 headless viewer4환경·32step 재로드를 완료했다. 평가 CPU/GPU/JSON 건수 일치·float32/64 ever 차이0·finite를 확인했다. [검증 보고](output_etc/check/before_push/report.md), [전체 검증](output_etc/check/before_push/verification.json). 이는 실행 검사이며 장기 학습 성능·512환경×3seed 평가·GUI/VNC 실화면은 검증하지 않았다.
- config/structure/USAGE/README와 [설정 상세](markdowns/reference/stage2_before_push.md)를 갱신했다. 원본 checkout·외부 checkpoint·기존 실행 프로세스는 보존했다.


### Stage1 PUSH 행동 비율 변경

- 사용자 지정대로 `stage1_push`의 새 학습 크기/행동 prior를 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%로 변경했다. 검증기의 허용 계약·샘플링 테스트·현재 PUSH/USAGE/평가 문서를 갱신했다. 과거 저장본의 분포와 평가 결과는 보존하며21k PTH의 기본 viewer/일반 평가는 저장된PUSH50%를 복원한다.
- CPU35개 통과:8만 사람 슬롯의 크기 조건부 샘플링 빈도·합계1·설정 검증·이전 저장본 복원/호환. [검증](output_etc/check/push_sampling_30/tests.log), [설정](output_etc/check/push_sampling_30/config_check.json). 비율 변경 검증에 GPU 학습/평가는 실행하지 않았다.


### Stage1 PUSH 통합과900step 통일

- `stage1_push` 환경/학습 YAML·train/test/VNC·일반/1명 단독 eval을 추가했다. scratch prior는 SIT7.5/CLIMB18.75/AT24.375/ON_TOP24.375/PUSH25%로 기존 네 과제 상대 비율을 유지한다. 정책은5과제·AMP4family로 확장하고 기존4과제/3family 기본값은 보존했다. PUSH167종·밀도3·cone90·접촉 RSI bank/모션을 이식했으며 모션21개와21k PTH는 외부 원본에 심링크로 연결했다.
- 학습·일반/단독 평가·viewer 기본 시간을900step으로 통일하고 시간 종료의 `-1` 경계를 수정했다. 새5과제 macro와 기존4과제 macro를 구분한다. PTH 옆 sidecar가 없으면 내부 저장 env/sim을 읽는다. 제공21k 일반 평가는 학습 당시 PUSH50% 분포를 복원하고 새 scratch는PUSH25%다. 서로 다른 학습 계약의 resume는 거부한다.1명 PUSH와1명 ON_TOP의 물체/받침 인덱스, 여섯 번째 RSI 통계 축을 일반화했다.
- 제공 `Stage1JointPushDensity3Cone90Finetune_00021000.pth`로512환경×3seed 일반3회·단독15회, 모두900step을 완료했다.5과제 macro 점/영역은 일반 **85.84±1.07 / 88.80±0.81%**, 단독 **83.32±0.51 / 88.78±0.32%**다. PUSH는 일반 **81.78±1.93%**, 단독 **81.51±1.47%**다.18개 trace CPU/GPU·JSON 건수 일치, float32/64 ever 차이0, 상태 finite를 확인했다. [리포트](output_etc/check/push_900/report.md), [검증](output_etc/check/push_900/verification.json).
- 이전78k1200step trace의 앞900frame을 별도로 재채점해 같은 제한 시간의 비교를 추가했다. 기존 원본 trace/보고서는 보존했다. 기존 네 과제의 단독 평균은21k에서 낮아졌으며 수치와 비교 조건은 리포트에 기록했다.
- CPU259개 통과. 실제78k→PUSH 전이 후 기존 네 과제 actor/critic 출력 보존, 원본 브랜치 정책·task encoder·clean scene 관측 코드 일치를 확인했다. Stage1/BEFORE 기존 PTH4환경·32step 재로드도 통과했다. PUSH scratch CHECK는2048환경으로 RSI5517 profile 준비 후 epoch2/frame262144를 저장했고 resume은 epoch3/frame393216까지 진행했다. 모델/RMS finite, scratch 시작·PUSH25%·900step 계약을 확인했다. [학습/재개](output_etc/check/push_900/training_validation.json). 원본 PTH/코드/기존 프로세스는 보존했으며 장기 본학습·GUI/VNC 실화면은 실행하지 않았다.


### 일반 평가와 Stage1 전용 단독 평가 분리

- 일반 `*_eval.sh`는 checkpoint 저장본의 학습 인원·물체·크기·과제/BEFORE 배치를 유지한다. Stage1만 `stage1_single_eval.sh`로1명·행동별 독립 평가를 제공한다. 단독 AT만 기존9종 크기, 나머지는 학습 크기/동적 물리를 따른다. viewer native 성공 진단은 `viewer_diagnostics.py`로 이름과 결과 태그를 분리했다.
- 정식 평가는 loco·정확히1200step·독립 seed0/1/2×512환경 첫 episode, 같은 궤적의 점/영역 ever·seed 평균/표본 std·macro다. 개인 낙상 frame까지 채점하고 해당 관절 drive를 해제하며 다른 사람은 계속 수행한다. BEFORE의 개인/follower 개인·둘 다 ever·동시 성공을 별도 집계한다. 학습 reward/종료 조건은 유지했다. viewer 기본 horizon도1200으로 맞췄다.
- Stage1 원본78000으로 일반3회(2명·4물체), 단독12회(행동별1명)를 완료했다. macro 점/영역은 일반 **95.17±0.84 / 97.17±0.18%**, 단독 **96.58±0.26 / 98.83±0.40%**다. [리포트](output_etc/check/eval_unified/stage1_78k_report.md). 15개 trace CPU float64/GPU·JSON 건수 및 float32/64 ever가 전부 일치했고 상태 finite·원본 checkpoint 해시 보존을 확인했다.
- CPU242개 통과. BEFORE 세 설정은 검사 저장본으로16환경·4명·1200step 실행/재채점(낙상8/10/15명, 각각24pair), viewer는4환경·32step의 새 진단 경로를 확인했다. Stage2 본학습 성능과 GUI/VNC 화면은 평가하지 않았다. [검증](output_etc/check/eval_unified/verification.json), [BEFORE](output_etc/check/eval_unified/before_validation.json). 새 규약·명령·코드 위치를 evaluation_single/config/structure/USAGE에 갱신하고 이전 단독 규약은 legacy에 보존했다.


### 네 실험 이름 통일

- 환경/학습 YAML·전용 train/test/VNC12개·실험/출력 이름·테스트·현재 문서를 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`로 통일했다. Stage1 보조 구현은 `stage1_config.py`, `stage1_geometry.py`로 바꿨으며 기본 전이 링크는 `checkpoints/Stage1_00078000.pth`다. 외부 원본 파일과 과거 결과 경로는 보존했다.
- 기존78000 및 이전 BEFORE checkpoint의 reward 태그·RSI 경로를 읽기 경계에서 변환한다. 보상·관측·모델 tensor·실험 간 혼용 거부는 유지한다. 기존 네 저장본을 새 test wrapper로4환경·32step headless 재로드해 정상 종료했고 네 모델의 전체 tensor·actor/critic 출력은 변경 전과 bitwise 동일했다. [재로드](output_etc/check/config_rename/smoke.json), [모델](output_etc/check/config_rename/model_equivalence.log). CPU224개 통과. 셸/Python/YAML·문서 링크·이전 실행 이름 잔여 검사 결과는 `output_etc/check/config_rename/`에 있다. GUI/VNC 실화면과 본학습은 실행하지 않았다.

### 네 컨피그와 학습·추론 경로 정리

- `stage1_v2_fixed`, `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200`만 유지했다. 나머지21쌍 YAML·63개 wrapper와 공동 운반·공유 받침·adapter·full RSI·carry50·단일인 태스크·전처리·미사용 에셋/모듈을 제거했다. 기존 파일322개를 삭제했으며 과거 문서는 legacy에 보존했다. [변경 목록](output_etc/check/four_config_cleanup/change_manifest.json).
- BEFORE의 공통 기하·배치·물리 검사·native 후행 RSI를 분리하고 모델을 공통 정책의 직접 확장으로 정리했다. 공통 환경의 관측·reset·모션/AMP를 별도 mixin으로 분리했다. 파라미터 이름·관측·보상·거리·초기화·동결/학습 범위를 유지했다.
- 전용12개 wrapper는 공통 `launch.sh`를 사용한다. conda 기본값은 모두 `tokenhsi`, seed는42, test 기본은 viewer다. GPU는 한 번 지정하며 기존 MPS만 연결한다. 설정 로더는 새 학습의 native YAML과 추론/resume의 저장 env/sim을 같은 경로에서 처리한다. `ENV_CONFIG`로 명시적 대체가 가능하며 fixed 임시 YAML 생성·실험 이름별 중복 분기를 제거했다.
- Stage1 78000은 `checkpoints/Stage1V2Fixed_00078000.pth` 심링크로 연결했다. 네 모션 데이터 디렉터리도 공유 원본에 연결했다. 원본 checkout·가중치·기존 학습 PID3432492는 보존했다. 확인 산출물·RSI 캐시·pytest 임시 파일은 `output_etc/check/`에 둔다.
- CPU214개 통과: 네 설정/계약 혼용 거부·실제78000 전이·설정 복원·wrapper/conda 선택·GTA·보상·거리·mask·RSI/fallback·재로드. [CPU](output_etc/check/four_config_cleanup/cpu.log). 네 모델의 전체 tensor·actor/critic 출력·계약이 정리 전과 bitwise 동일했다. [모델](output_etc/check/four_config_cleanup/model_equivalence.log). 네 설정의 전체 reset2회·부분 reset1회에서 관측·사람/상자 상태·크기·목표·물체 연결의 최대 차이0을 확인했다. [reset](output_etc/check/four_config_cleanup/reset_equivalence.json). 후행 RSI sampler의 출력·RNG도 동일했다.
- GPU0/MPS에서 fixed2명·2048환경과 BEFORE4명·1024환경의 CHECK가 각각 epoch2/frame262144 저장 후 종료했다. fixed native RSI6413 profile을 생성했다. BEFORE 세 설정은 source78000·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/scalar finite를 확인했다. mix226 resume은 epoch3 저장 후 종료했다. [학습/재개](output_etc/check/four_config_cleanup/training_validation.json).
- 실제78000 fixed headless 추론과 BEFORE 전용 test의4환경·32step 저장본 재로드를 확인했다. BEFORE32환경의 loco/mix·ON_TOP pickup/carry·단독 등14개 구성에서 alias 관측/pose·활성 물리 수·거리·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리 검사 진행 결과](output_etc/check/four_config_cleanup/physics.log), [wrapper](output_etc/check/four_config_cleanup/wrapper_validation.json). 셸/Python/YAML·현재 문서 링크·4쌍 YAML/12개 wrapper·미사용 import/실험 참조 제거를 검사했다. VNC 실행 의존성이 없어 실제 GUI/VNC 화면과 장기 성능은 검증하지 않았다. 본학습은 시작하지 않았다.

## 2026-10-28

- 코드 정리용 독립 복사본을 만들었다. 원본과 복사 범위·초기 검증은 `code_copy_manifest.json`과 [당시 changelog](markdowns/legacy/pre_four_configs/changelog.md)에 보존한다.
