# BONES-SEED Uniform 후보 검증

공식 viewer: https://seed-viewer.bones.studio/

## 검증 범위

- 전체 archive의 다운로드 크기 및 공식 LFS SHA256 검증 통과.
- 우선 후보 11개는 공식 viewer API에서 받았으며 BVH frame 수·FPS·root·FK skeleton을 분석했다.
- 문 모델·문 pose·접촉 flags·다른 사람 정보가 없어 완전 통과/doorway clearance/hold-open은 확정하지 않는다.
- 변환·retarget·AMP/YAML 변경·학습·평가는 실행하지 않았다.

## 실제 representation

- Root wrapper는 6채널이 모두 0인 정적 노드다. Hips translation을 human root로 사용한다.
- Wrapper 포함 78 joints, 제외 77 joints, 총 240 channels. Hips에는 translation과 rotation 6채널이 있다.
- Y-up, cm 단위. Frame Time=0.008333초(약120FPS). Metadata frame 수는 우선 11개 모두 일치한다.
- Hips OFFSET를 translation 채널에 다시 더하면 root 높이가 잘못된다. 공식 parser와 같이 Hips translation을 root로 사용한다.

## 우선 후보 측정값

| 분류 | filename | frames | Root 총 순이동 m | 기준 event 이후 전후 이동 m | 기준 event 이후 좌우 이동 m |
|---|---|---:|---:|---:|---:|
| TYPE 1 | `inside_door_handle_left_side_open_walk_R_001__A515` | 675 | 2.032 | 0.167 | 0.118 |
| TYPE 1 | `inside_door_handle_right_side_open_walk_R_001__A515` | 621 | 2.273 | 2.110 | -0.003 |
| TYPE 1 | `inside_door_handle_left_side_open_walk_close_behind_R_001__A514` | 1033 | 1.570 | 1.384 | 0.029 |
| TYPE 1 | `inside_door_handle_left_side_open_walk_R_001__A512` | 408 | 0.983 | 0.121 | 0.097 |
| TYPE 1 | `inside_door_knob_left_side_open_walk_R_001__A514` | 552 | 1.792 | 1.284 | -0.014 |
| TYPE 2 검토 | `inside_door_handle_right_side_open_full_R_001__A515` | 688 | 0.054 | -0.439 | -0.076 |
| TYPE 2 검토 | `inside_door_handle_left_side_open_full_R_001__A515` | 654 | 0.014 | -0.097 | -0.075 |
| TYPE 2 검토 | `inside_door_knob_right_side_open_full_R_001__A515` | 664 | 0.012 | -0.385 | -0.024 |
| TYPE 2 검토 | `inside_door_handle_right_side_open_R_001__A512` | 439 | 0.011 | -0.195 | 0.030 |
| TYPE 2 검토 | `inside_door_handle_right_side_open_full_R_001__A512` | 467 | 0.052 | -0.239 | -0.064 |
| TYPE 1 Pull | `outside_door_knob_right_side_open_walk_R_001__A515` | 854 | 1.525 | 1.459 | 0.442 |

전후/좌우는 초기 몸 방향 기준이다. 문 평면 기준이 아니다. TYPE 1은 개방과 이동이 겹칠 수 있으므로 event 이후 이동량이 작아도 전체 root 전진이 작다는 뜻은 아니다. 세 번째 clip은 전체값에 이후 닫기·회전이 포함된다.

## 직접 보기

- [로컬 원본 skeleton 재생](candidate_preview.html): pause, 0.5x, scrub, viewpoint, pelvis follow. 기본 follow OFF로 실제 root 이동을 보인다.
- [원본 skeleton 시점별 그림](candidate_contact_sheet.png)
- [Root 전진 그래프](root_forward_trajectories.png)
- [수치표](candidate_numeric_validation.csv)
- [상세 JSON](candidate_numeric_validation.json)

TYPE 1 push: A515 left/right handle open_walk를 우선 확인. A514 left handle open_walk_close_behind는 통과 직후 3.5초 근처 crop 검토. TYPE 2는 후퇴가 있지만 측면 이동은 작고 손을 해제하는 패턴이며 현재 UNKNOWN을 유지한다.

## Archive 및 선별 파일 검증 완료

- Archive: 45,188,088,297 bytes, SHA256 `0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c`, 공식값 일치.
- 선정 184개만 extraction: 373,134,682 bytes. 누락 0개. 전체 archive extraction은 하지 않았다.
- Viewer 원본과 archive의 동일성: 11개 SHA256 일치, 불일치 0개.
- 선정 184개 모두 frame 수·공통 skeleton·finite 검증 통과. 77 joints, 240 channels, header FPS 약120.
- [184개 수치 검증](archive_selected_numeric_validation.csv), [검증 요약](archive_selected_validation_summary.json), [Extraction 목록](archive_extraction_report.json).
