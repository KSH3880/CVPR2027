# 78k의 2명·4물체 CLIMB 대상 교체 평가

78k checkpoint의 relation_experiment_config env/sim/train을 그대로 추출한다. 2명/4물체, 원본30D object/GTA모델과정규화, 원본reset/충돌/낙상/600step을유지한다. task_graph=climb로두사람이각각자기대상에CLIMB한다. eval초기화는checkpoint의loco100%이며등반RSI는사용하지않는다.

물리물체0,1은두사람의목표,2,3은원래크기분포의박스distractor로유지한다. Box대조에서는모두원래박스다. Table_Square/Cabinet에서는0,1만원본TokenHSI test native모델로종류내균등복원추출한다. 원래Stage1와같이동적물체(fix_base_link=False),native형상/URDF질량·관성을사용한다. 이전1명공식평가의고정가구와구분한다. 목표는각물체원점XY,Z+tarSitPos.z+.94m이며Box대조는원래bbox상판+.94m다. 정책관측은실제bbox/원본입력그대로,토크나이저나weight변환없음.

평가:512환경×2명×3repeat,seed0,deterministic정책. 첫scene종료까지사람별골반목표20cm ever와두사람모두성공을집계한다. 장면종료는원래multi-agent환경규칙을유지한다. 599실제step≈20초와처음299step≈10초성공을모두저장한다. 0회차는보존하고주요요약은1·2평균이다. 반복은독립학습seed가아니다.

준비된output디렉터리의env.yaml/train.yaml을사용해 `bash evaluation/two_agent_climb/run.sh Box|Table_Square|Cabinet [output_directory]` 실행. 기본output은output_etc/check/two_agent_climb_78k_20261025다. 각조건은고유output에서실행하며기존결과를덮어쓰지않는다.

각repeat의initial/trace NPZ와results.json을보존한다. 실행중GPU값과NumPy64재채점성공배열동일성,finite,환경별첫종료1회를검사한다. native model ID/실제bbox/전체사람·물체초기상태를저장한다. baseline/학습코드/프로세스는수정하지않으며등록은평가프로세스에한정한다.

고정대상대조는 `CLIMB_FIXED_TARGETS=1 bash evaluation/two_agent_climb/run.sh <category> <fresh_output>`로실행한다. 원래dynamic분기는기본값0으로보존한다. 고정분기에서는대상0/1의asset만fix_base_link=True로생성하며방해박스2/3은동적으로유지한다. Box는원래절차생성크기그대로고정asset을만든다. 모든step에서각scene첫episode가끝나기전target XYZ/quaternion의reset대비변화가2e-4미만인지assert하며최대값을results.json에저장한다. 고정실행결과는 `output_etc/check/two_agent_climb_78k_fixed_20261025/`에분리하고summarize.py로동적결과와함께비교한다.

30초평가는기존고정대조의env.yaml에서episodeLength만900으로바꾼별도output을사용한다. `CLIMB_FIXED_TARGETS=1 bash evaluation/two_agent_climb/run.sh <category> output_etc/check/two_agent_climb_78k_fixed_30s_20261025`로세종류를실행한다. 결과의agent_success_10s/20s/agent_success_rate는이번동일궤적의299/599/899step ever다. `python evaluation/two_agent_climb/summarize_horizons.py <output>`가전trace재채점·시간별성공비감소·첫종료·고정pose·환경파일시간외동일성검증후REPORT.md와summary_horizons.json을저장한다. 기존20초보고서와섞어시간효과를계산하지않는다.
