# 전용 브랜치 업로드

현재 루트는 TokenHSI PUSH + KD 전용 snapshot이다. 브랜치는 `baseline/tokenhsi-push-kd`이며 기존 CVPR2027 브랜치와 독립 이력이다.

```bash
git push -u origin baseline/tokenhsi-push-kd
```

같은 이름의 원격 브랜치가 있으면 먼저 내용을 확인한다. 강제 push하지 않는다.

다른 머신에서 코드만 받기:

```bash
git clone --single-branch --branch baseline/tokenhsi-push-kd https://github.com/KSH3880/CVPR2027.git tokenhsi_push
```

가중치·데이터는 별도로 연결한다. 원래 실행 폴더와 실행 중인 프로세스에는 영향이 없다.
