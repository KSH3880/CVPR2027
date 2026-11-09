# GitHub 업로드 가이드

추천 브랜치: `baseline/tokenhsi-push-kd`.

이름은 TokenHSI 기반 PUSH baseline과 KD 보존 학습임을 나타낸다. 현재 프로젝트 폴더는 Git 저장소로 초기화되지 않은 상태다. 아래 명령은 직접 실행하는 가이드이며, 이번 문서 작업에서 commit/push는 실행하지 않았다.

## 기존 CVPR2027 GitHub에 전용 브랜치로 올리기

확인한 기존 원격 주소는 `https://github.com/KSH3880/CVPR2027`이다. 아래 방식은 현재 폴더를 독립 Git 이력의 전용 브랜치로 올린다. 기존 main 기반 PR/merge를 만들 목적이라면 이 절차 대신 기존 저장소의 worktree에서 프로젝트를 추가한다. 기존 main 기반 이력과 독립 프로젝트 이력은 용도가 다르다.

```bash
cd /home/inkyu/projects/tokenhsi_push
git init -b baseline/tokenhsi-push-kd
git remote add origin https://github.com/KSH3880/CVPR2027.git
```

다음 조회 결과가 비어 있으면 같은 이름의 원격 브랜치가 없는 것이다. 결과가 나오면 기존 브랜치를 덮어쓰지 말고 다른 이름을 선택한다.

```bash
git ls-remote --heads origin refs/heads/baseline/tokenhsi-push-kd
```

코드·YAML·MD·모션 목록·RSI bank metadata를 올리고, 학습 가중치·실제 모션·실행 결과·로컬 데이터 심링크는 제외한다. `.gitignore`에 관련 규칙을 추가해 두었다.

```bash
git add .
git diff --cached --stat
git status --short
```

목록에 `weights/*.pth`, `output/`, `output_etc/`, 실제 NPY/NPZ 모션, 데이터 심링크가 없는지 확인한다. 확인한 뒤:

```bash
git commit -m "Add TokenHSI PUSH baseline with teacher distillation"
git push -u origin baseline/tokenhsi-push-kd
```

`--force`는 사용하지 않는다. 인증 오류가 나면 GitHub 인증을 먼저 준비한다. 로컬 Git 이름/이메일이 미설정이면 사용자 본인의 값을 이 저장소에 설정한 뒤 commit한다.

## 이후 변경 업로드

```bash
git add .
git diff --cached --stat
git commit -m "Update TokenHSI PUSH baseline"
git push
```

## 다른 머신에서 받을 때

```bash
git clone --single-branch --branch baseline/tokenhsi-push-kd \
  https://github.com/KSH3880/CVPR2027.git tokenhsi_push
```

모션 manifest는 현재 머신의 절대 경로를 포함한다. 다른 머신에서는 데이터 경로/심링크를 연결하고 teacher 가중치를 별도 준비해야 한다. `README.md`와 `METHOD.md`를 확인하고 `python scripts/preflight.py`로 검증한다. GitHub에 코드가 올라갔다고 다른 머신에서 즉시 학습이 가능한 것은 아니다.

본학습·GPU PPO는 아직 미실행이다. 업로드 commit 설명에는 CPU 검증 범위만 적고 장기 학습 성공률을 주장하지 않는다.
