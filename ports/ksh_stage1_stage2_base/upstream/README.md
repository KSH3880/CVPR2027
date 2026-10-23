# Stage1 Multi-Agent TokenHSI

`stage1`, `stage1_v2`, `stage1_v2_fixed`, `stage2` 네 multi-agent 실험을 유지한다. [TokenHSI](https://github.com/liangpan99/TokenHSI)의 데이터·AMP 기반 위에 다중 사람의 독립 SIT/CLIMB/AT/ON_TOP 과제를 학습하며, stage2는 fixed 위에 공동 운반을 추가한다.

공유 task bias·사람 토큰 residual MLP·GTA, 별도 actor/critic, CPA와 크기별 RSI를 사용한다. 기본 학습은 **2명·2048환경·4물체**다.

## 시작하기

- [실행과 config](markdowns/config.md): 현재 컨피그와 학습·로컬 추론·VNC 명령·추론 옵션
- [코드 구조](markdowns/structure.md): 구현과 테스트 위치
- [설치·추가 실행 옵션](markdowns/USAGE.md), [머신 포팅](markdowns/reference/PORTING_GUIDE.md)
- [최근 변경](changelog.md), [작업 규칙](AGENTS.md)
- [정책·환경 상세](markdowns/reference/ma_clean_scene_gta.md), [과거 기록](markdowns/legacy/README.md)

```bash
nvidia-smi
# output_etc/check/에 저장하는 짧은 확인 학습
TOKENHSI_GPU=0 CHECK=1 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4
# CKPT에 fixed 또는 v2 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_test.sh "$CKPT" 2 32 4 3
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_vnc.sh "$CKPT" 2 1 4 10
```

실행 스크립트는 `runtime_env.sh`로 `tokenhsi` conda 환경과 CUDA를 준비한다. 원본 레포 `/home/cvlab/Desktop/CVPR2027/stage1_message`와 공유 데이터는 보존하고, 본학습은 `output/`, 모든 확인·평가 결과는 `output_etc/check/`에 저장한다.

원본 단일인 TokenHSI 태스크·데이터 전처리는 별도로 유지한다. 큰 모션·물체 데이터와 SMPL 모델은 Git에 포함되지 않으며 설치 안내를 따른다.

## Acknowledgements

This work builds directly on [TokenHSI](https://github.com/liangpan99/TokenHSI) by Liang Pan,
Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura and Jingbo Wang, which in
turn builds on [ASE](https://github.com/nv-tlabs/ASE), [PADL](https://github.com/nv-tlabs/PADL) and
[InterScene](https://github.com/liangpan99/InterScene). All credit for the base method belongs to
the original authors.

```bibtex
@inproceedings{pan2025tokenhsi,
  title={TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization},
  author={Pan, Liang and Yang, Zeshi and Dou, Zhiyang and Wang, Wenjia and Huang, Buzhen and Dai, Bo and Komura, Taku and Wang, Jingbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

Released under the [MIT License](LICENSE), matching the upstream TokenHSI license. External
libraries and datasets (AMASS, SAMP, OMOMO, SMPL, IsaacGym) remain subject to their own terms.
