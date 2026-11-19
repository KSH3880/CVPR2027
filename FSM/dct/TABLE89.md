# Table 8·9 로컬 재현

Ubuntu/Linux에서 Python 3.8, 네이티브 Isaac Gym과 호환되는 PyTorch/CUDA, NVIDIA 드라이버가 필요하다. 서버는 conda `tokenhsi` 환경을 사용한다. Python 의존성은 저장소의 `requirements.txt`와 `psutil`이다. Isaac Gym과 PyTorch/CUDA는 별도로 설치한다. 정식 실행은 물리 GPU 6 또는 7을 사용한다. 해당 슬롯이 없는 컴퓨터에서는 CPU 준비 검증까지 할 수 있다.

## 코드와 입력

새 디렉터리에 GitHub 브랜치를 받는다. 체크포인트와 모션 데이터, Coop. FT 및 TokenHSI* 런타임은 접근 권한이 있는 실험 서버에서 별도로 복사한다. 아래 호스트명은 실험 서버의 이름이며 로컬 SSH 설정에 등록된 주소나 별칭으로 바꿀 수 있다.

```bash
git clone --single-branch --branch koo_1119 https://github.com/KSH3880/CVPR2027.git cvpr-table89
cd cvpr-table89
scp hwanhee@blackwell00:/home/hwanhee/koo_cvpr/runs/koo_1119_table89_inputs.tar.gz .
sha256sum koo_1119_table89_inputs.tar.gz
tar -xzf koo_1119_table89_inputs.tar.gz
```

아카이브 SHA-256은 `8b9fa7a22fa8453dc8fc7df9273faee68b03266a8ede86d38071e8cbdf3bc270`이다. 실행 준비 단계에서 논문 평가에 사용한 4개 체크포인트의 해시와 모션 파일을 확인한다.

## 평가

```bash
conda activate tokenhsi
bash FSM/dct/table89.sh --prepare-only
bash FSM/dct/table89.sh --run
bash FSM/dct/table89.sh --post-at
```

다른 이름의 conda 환경을 쓸 경우 활성화하고 `TOKENHSI_CONDA_ENV`에 환경 이름을 지정한다. 실행 스크립트의 Python은 `TABLE89_PYTHON`으로 지정할 수 있다.

- Enabling SIT·CLIMB·ON_TOP 및 Coupled PUSH 2·3·4인, 5개 방법을 평가한다. 시드 1·6·7, 900 step(30초), 0초와 Uniform[1, 5)초 조건을 사용한다. 평가 계열·조건·방법·시드마다 512개 장면을 평가하며 180개 작업으로 나뉜다.
- 진행과 로그는 `runs/table89_delay/cohort/`에서 확인한다. 완료된 결과는 같은 디렉터리의 `report.md`와 `summary.json`에 저장한다. Post-AT 결과는 `runs/table89_delay/post_at/`에 저장한다.
- GPU 메모리 상한은 감지된 총 메모리에서 1,024 MiB를 뺀 값과 원본 88,000 MiB 중 작은 값이다. 실제 필요한 메모리는 작업 크기와 GPU에 따라 달라진다.
- `FSM/dct/README.md`의 앞선 파일럿과 달리 Table 8·9의 실행 진입점은 `FSM/dct/run_cohort.py`다.
