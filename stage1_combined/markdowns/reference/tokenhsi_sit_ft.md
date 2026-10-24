# TokenHSI SIT 환경으로 78k full fine-tuning

최종 사용자 지시: 기존2명학습종료,1명1의자,SIT만,AMP CLIMB제외.

GPU5,tokenhsi_jhh,2048환경,source checkpoint78000에서전체모델FT,lr2e-5,20,000iteration. 원본의4096환경 대신프로젝트기본2048환경을사용한다. 우리35Dchair/GTA입력과AMP3label차원은78k호환을위해유지하며,원본TokenHSI정책자체를교체학습하는것은아니다.

원본SIT환경재현: train49개 의자uniformreplacement,모델은환경생성때고정. 걷기참조초기화50%+sit참조50%, sample_time_rsi원본허용frame. 걷기초기화의자는사람XY주변U(1,5)m·360도,의자yawU(0,2π). sit초기화의자는참조motion의objectXY/yaw를쓰고native bbox반높이로바닥배치한다. 정면제한·nativeRSIbank·late/fullpool·추가support/foot필터미사용. 단일사람이므로사람간충돌조건없음. 기존공유데이터는보존.

목표: metadata tarSitPos와일치하는좌판+10cm. 보상함수는原본compute_sit_reward를AST동일복사: near.7+접근속도1.5m/s.3,속도/각속도벌점, power coefficient.0005, task/AMP.5/.5. episodeLength300(약10초),성공근처frame누적59회기준학습IET,평가IET비활성. relation진단은기존시스템에남지만실제taskreward는원본함수출력이다.

AMP파일 dataset_tokenhsi_sit_only.yaml에는원본loco_sit12개를loco별칭으로,원본sit20개만수록. CLIMB/CLIMBNoRSI뿐아니라불필요한Carry라이브러리도미로딩. 전문가선택걷기1/3·SIT2/3,모든AMP family SIT. 세onehot채널은체크포인트shape호환용으로유지하지만climb채널라벨/데이터를학습에사용하지않는다.

검증: CPU13개(기존모드포함)통과. 2048환경1iteration학습rc0. 초기loco1040/sit1008,loadedlib=['loco','sit'],climb_loaded=false, native_cache_used=false,front_filter=false,AMP finite/SITlabel만확인. source_epoch78000,frozen0,actor/criticchair gradient비영. 검증 output_etc/check/tokenhsi_sit_78k_20261024_v2/. 첫잘못된root관측옵션의실패로그보존,기존GTA필수관측규약으로수정.

활성run정보: 저장소tokenhsi_sit_active_run.json 및project runs/tokenhsi_sit_78k_20261024_130732/active_run.json.
