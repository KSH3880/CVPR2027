# HOLDING+AT Table carry RSI

요청에 따라 **기존 HOLDING_AT 보상/계수·78k 모델/30D bbox·AMP expert 분포를 유지**한다. 원본 carry 보상으로 교체하는 이전 준비안은 실행하지 않고 정리했다. 기존 학습은 사용자 요청으로 중단된 상태이며 재개하지 않았다.

## 보정과 검증

- 원본 carry B20 `carryWith` 129frame/30Hz를 읽기 전용으로 사용한다. 현재는 carryWith RSI만 지원하고 pickUp/putDown은 추가하지 않았다.
- 실제 Table_Circle URDF collision mesh에 ray를 쏘아 상판 아랫면 접촉점을 얻고, native32 관절 한계 안에서 양팔 IK를 수행한다. 4cm 손 sphere가 아래에서 받치도록 배치한다. 사람 전체 스케일링·고정 grasp constraint·강제물체부착은 없다.
- 현재 프레임의 IK 관절 offset을 이전9개모션frame에도 동일하게 적용한다. 관절제한 clamp·관절속도 재계산·nativeFK로 body 및 AMP history를 함께 보정한다. endpoint를 복제해 history를 만들지 않는다.
- 현재 손 목표 오차≤2cm, 이력 손–mesh 거리≤6.5cm, 샘플링한 신체 관통≤2cm/지면≤2.5cm를 정적 검사한다. 열린 mesh의 occupancy 부호는 신뢰하지 않고 양면거리 검사 후 실제 VHACD 물리로 확인한다.
- 2048환경에서 원래 속도를 유지하고 PD 자세를 고정한0.3초 물리 검증을 수행한다. 모든 복제본에서 finite/사람속도<3m/s/root높이>.5m/root이동<.3m/물체낙하<.15m, 손–앵커 거리 첫0.1초<9cm·0.3초<10cm를 통과한 후보만 공개뱅크에 넣었다. 이는 짧은 reset 안정성 검증이며 장기 운반 성공 보증이 아니다.
- 옆면접촉 후보는0.3초 검증 전부실패하여 사용하지 않는다. 원시 실패후보/로그는 별도 보존한다.

## 현재 뱅크 범위

`runs/table_rsi_20261026/output_etc/check/bank` (프로젝트 루트 기준), `TABLE_RSI_BANK`로 경로 변경 가능. train16/21모델37자세, test4/9모델6자세가 통과했다. **나머지 모델은 아직 안전한 RSI가 없으므로 원래 locomotion reset을 유지**한다. 물체의 train21 균등추출 분포는 바꾸지 않는다.

RSI 요청확률=.5, 보행=.5지만 모델누락 fallback을 포함한 실제RSI율은 약38%다. 2048환경reset/부분reset 검증 후768환경(37.5%)가RSI였다. 요청/적용/누락누계와active비율은 `relation/table_rsi/*` TensorBoard에 기록한다. 사용 가능한 모델 안에서는 검증 후보를 균등추출한다.

HOLDING의 기존 성공식은 여전히 양손평균–bbox중심 거리 기반이다. 실제 상판아랫면을 받치는 접촉과 이 성공식은 동일하지 않다. 보상을 이번 작업에서 바꾸지 않았으며, RSI를 추가했다고 해당 학습 보상 문제가 해결됐다고 주장하지 않는다.

## 실행

`stage1_combined`에서:

```bash
# 원본78k로 보정 RSI 자세 확인. viewer는 안전 후보가 있는 test4모델만 사용한다.
TABLE_RSI_VIEW=1 TABLE_TEST_ENVS=1 TABLE_REPEATS=10 HEADLESS=0 TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/table_carry_rsi_full_ft_test.sh checkpoints/Stage1V2Fixed_00078000.pth

# 학습(현재 본학습은 시작하지 않았음)
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/table_carry_rsi_full_ft_train.sh
```

`TABLE_RSI_VIEW`를 켜지 않은 평가는 기존처럼 loco부터 시작하므로 정식 성능비교 조건을 유지한다. viewer subset을 정식 test9평가와 혼동하지 않는다. checkpoint는 원본78k 또는 본 RSI FT체크포인트를 사용한다. GUI/VNC는 동일 이름 wrapper와 config.md 참조.

검증: CPU2개,2048환경1회PPO/epoch1저장·모델165tensor finite/양토크나이저gradient비영,2048환경부분reset 비선택상태exact/보정history exact/AMPfinite 통과. 원시기록·그림 `runs/table_rsi_20261026/output_etc/check/`, 보고서 `runs/table_rsi_20261026/REPORT.md`. 확인용epoch1모델은 본학습 출발점으로 사용하지 않는다.

추가 실행 경로 검증: 원본78k+TABLE_RSI_VIEW=1의test wrapper를1환경headless/1repeat로끝까지실행해rc0 확인. 로그 output_etc/check/viewer_route.log.
