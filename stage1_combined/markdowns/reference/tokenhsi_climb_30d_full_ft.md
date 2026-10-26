# 78k 원본30D CLIMB 전체 FT

사용자요청으로 기존35D 의자 tokenizer 재사용을폐기하고78k 원본30D 모델을그대로사용한다. 모델key/shape와모든165state tensor를strict load후exact 대조하고scene/AMP/value 정규화를strict 복원한다. 새branch·입력열·파라미터복제는없다. Adam은새FT optimizer이며원본78k에서새롭게시작한다. 기존35D FT결과는보존하되이번학습의시작점으로사용하지않는다.

기존 CLIMB환경을상속하고 관측의35D 확장만제거한다. 1명/1고정물체/2048환경,train Box·Cabinet·Table 각1/3,원본모션loco/CLIMB RSI50:50,기존보상/AMP/IET/낙상/300step유지. lr2e-5,100000epoch,500회마다저장,전체층·정규화학습유지.

`train_climb30.py`는프로세스내30D task/agent를등록한다. 기존`amp_multi_agent` builder와일반`MAPlayerContinuous`를사용하며ChairTokenizer/ChairFTPlayer는사용하지않는다. task의크기/scene entity/관측은원래HumanoidMACarry의223/30/1 형식을사용한다. 환경의chairFT키는기존geometry/reset설정보관용일뿐신경망branch를추가하지않는다.

2048환경1update 검사: output_etc/check/tokenhsi_climb_30d_78k_start_20261026/. sourceepoch78000·object30D·state165개exact·새파라미터없음·actor/critic object tokenizer gradient확인. 이후 본학습실행원장: ../runs/tokenhsi_climb_30d_78k_20261026_120852/active_run.json.

전용30D test wrapper는원본TokenHSI 점평가를사용하며POINT_REGION=1이면기존영역도같은trace에서집계한다. 78k원본도이30D 평가경로에서직접load가능하다. 기존35D checkpoint에는기존35D wrapper를사용한다.

사용자 추가 요청: 저장주기500·최대100000으로config 수정. 기존 실행은설정을다시읽지않으므로최신checkpoint에서resume해야실행에적용된다. 사용자승인후기존학습을종료하고FT100에서resume완료. 모델·정규화·Adam exact,실제max100000/save500확인. 현재원장 ../runs/tokenhsi_climb_30d_100k_20261026_121912/active_run.json.
