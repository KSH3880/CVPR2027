# Chair SIT-only 78k full fine-tuning

초기weight: checkpoints/Stage1V2Fixed_00078000.pth(epoch78000 확인). 전체모델및정규화통계학습, 동결parameter0. 35D chair tokenizer 유지. lr2e-5, 20,000iteration,100회마다checkpoint.

GPU5,tokenhsi_jhh,2명×2048환경. 각환경사용의자2개+주변의자2개, 모든과제SIT100%. CLIMB/Box/Table목표없음. 기존2명/4물체 scene/attention계약 유지.

RSI bank: output/rsi_cache_native_front_foot_20261023. 기존bank보존,chair49종sit.npz만사용. RSI probability50%,late draw70%/full30% 유지(phase없는clip은safe full poolfallback). 정면접근·지지·발교차검사유지,제외상태재추출. RSI AMP9프레임dt1/30 및129D이력→기존family부호추가.

1iteration2048환경검증rc0: SIT4096/CLIMB0,물리물체전부chair, RSI missing0, active2063, AMP history최대오차0/전체finite,source78000,frozen0,actor/critic chair gradient비영. 발장면교차12건재추출. 검증: output_etc/check/chair_sit_only_78k_20261023_v2/.

본학습현재경로/PID: 저장소루트 chair_sit_only_active_run.json 및 project runs/chair_sit_only_78k_20261023_203037/active_run.json. 현재실행상태는실제프로세스/로그기준.
