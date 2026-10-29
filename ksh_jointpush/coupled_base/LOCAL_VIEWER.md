# stage1_2_base coupled 준비 상태

고정 commit: `2979913dbfb3bd823da324e4bc0b493444782437`.
현재 상태: **사용자 제공 Stage2 coupled 체크포인트 대기. 뷰어 미실행.**
사용자 지시에 따라 추가 학습·제로샷 생성은 하지 않는다. 이전 확인용2epoch 모델은 사용자 모델로 대체하거나 viewer 기본값으로 사용하지 않는다.

- CPU 전체313개 통과. 별도PreLN forward와NONE/SELF/BEFORE/COUPLED bias gradient 확인.
- Stage2 PreLN은 사람 coordination에 적용된다. 전이된GTA encoder는PostLN이다. 유지된legacy `stage2_before_nonemask_rsi433_near50`의actor는NONE mask/SELF0 예외이며, 요청한coupled/BEFORE PUSH near50 기본값은all bias다.
- 자료235행 경로/PUSH21clip·캐릭터/mesh bank해시 확인. 로컬78k·21k는 검사 입력으로만 준비했다.
- 범위정정 전 확인학습2epoch 완료: encoder73tensor·입력RMS 불변/action head6tensor 갱신/optimizer195state finite.32환경8preset 물리·AMP·부분reset검사 완료. 후속16환경 평가는 사용자요청에 따라 중단했다. 본학습·정식512환경3seed·Stage1/BEFORE GPU전체평가·GUI실화면은 수행하지 않았다.
- 기존GPU5본학습/MPS 유지. 추가 검증프로세스 종료 확인.

뷰어 명령(체크포인트를 전달받은 후):
```bash
TOKENHSI_GPU=6 PORT=6240 TASK_GRAPH=all_coupled bash ksh_jointpush/scripts/coupled_base_vnc.sh /absolute/user_checkpoint.pth 4 1 8 10
```
`TASK_GRAPH=random_scenario`는저장분포, `all_coupled`는4명공동PUSH 관찰이다. Stage1PTH를Stage2저장본으로대신사용하지않는다. checkpoint와저장env/sim·model/RMS·coupled계약호환을확인한후실행한다. 기존19500기반JointPush의직접resume은하지않는다. 새로운뷰어는MPS에연결하지않는다.

근거: `output_etc/check/branch_audit/{preparation,bias_validation,checkpoint_validation,status}.json`, `pytest.log`, `physics/physics_validation.json`. 출력은이프로젝트 `output_etc/check/` 아래다.
