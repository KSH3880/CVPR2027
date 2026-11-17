# SparseOcc E6 Fixed-ID IoU / GT-relative-yaw oracle 평가 (2026-11-10)

이 문서는 다른 repo(`/home/user/jhh/Projects/SparseOcc`)의 평가 결과를 정리한 것이다. TokenHSI 코드와는 무관하다.
평가 규약은 「E6 Fixed-ID IoU 및 GT-relative-yaw rotation oracle 구현 명세」를 따르고, 아래 **명세와 다른 점**에 적은 항목만 바꿨다.
평가 코드와 규칙은 [EfficientOCF 평가](efficientocf_e6_yaw_oracle.md)와 같다. 이번에는 **GT 바닥 crop 같은 GT 기반 예측 수정을 쓰지 않았다.**

## 1. 입력

| 항목 | 값 |
|---|---|
| 모델 / checkpoint | SparseOcc r50 nuImg 704×256 8f 60e v1.1, `checkpoints/sparseocc_r50_nuimg_704x256_8f_60e_v1.1.pth` |
| config | `configs/r50_nuimg_704x256_8f_60e_gmo_val.py` (RGB 8 frame, zero pad, query mask > 0.5, GMO-argmax query) |
| 예측 (재사용, 주 결과) | `outputs/original_sparse_prediction_full` (2026-08-28, 5119 NPZ). 이 repo의 original sparse GT 평가와 sparse FP tolerance 평가가 쓴 export다. |
| 예측 (비교) | `outputs/gmo_prediction_export_2026-08-16_23-19-19` (pedestrian-argmax query 포함) |
| GT, yaw, visibility drop | EfficientOCF 평가와 같다. hybrid-solid v1 val, `bbox_3d_reference_lidar.yaw_rad`, 원본 LighTROcc 규칙으로 재구축한 drop ID를 그대로 복사해 사용했다. |
| 평가 코드 | `tools_custom/eval_e6_yaw_oracle.py`, 테스트 `tools_custom/test_e6_yaw_oracle.py` (SparseOcc repo, 커밋하지 않은 새 파일). EfficientOCF 버전에서 모델 설명과 기본 경로만 바꿨고 bottom-crop 옵션은 뺐다. code sha256 `5d89152fa2b93e60…` |
| 산출물 | `outputs/e6_yaw_oracle/{full5119, full5119_zero, full5119_export0816}/` |

추론은 다시 돌리지 않았고 CPU만으로 계산했다(설정당 약 30초, 16 worker).

## 2. 명세와 다른 점 (SparseOcc adapter)

1. **미래 예측은 static copy다.**
   - SparseOcc는 현재 frame만 예측하고, t+1..t+4에는 현재 예측을 그대로 복사한다. export의 5개 frame은 비트 단위로 같다.
   - 따라서 Native 미래 형상은 "현재 형상을 예측 미래 center로 평행 이동한 것"이며, 이동량이 0인 경우에 해당한다. 명세 §6의 구조와 가깝다.
2. **Rotation oracle은 voxel 변형(§8.2)이다.**
   - SparseOcc export에는 query center가 없어서 mask의 BEV centroid를 회전 중심으로 쓴다.
   - 회전 방식은 inverse-mapping nearest이고 Z는 그대로다: `mask_oracle(q,k) = NN-resample(R(Δθ_GT(g,k)) about BEV centroid of mask(q,0))`.
   - 0°에서는 Native와 같다. Gaussian 연속 회전이 아니므로 LighTROcc oracle과 정의가 다르다.
3. **GT 기반 crop은 없다.** SparseOcc는 3D를 직접 예측하고, 예측 z 분포(z≈10~35)가 GT(z≈8~37)와 겹치므로 crop이 필요 없다.
4. **예측 선택 규칙:**
   - export의 query는 GMO-argmax이면서 pedestrian-argmax가 아닌 query다(최대 100개 중 sample당 평균 47.3개).
   - AP candidate는 이 선택 query 전체이고, `selected_query_score` 내림차순으로 정렬한다. frame에서 mask가 빈 query는 ranking에서 뺀다(static copy라 실제로는 빈 mask가 없다).
5. **Ped 정책은 vehicle-only yaw oracle이다.** 매칭된 pedestrian 76쌍은 Native를 유지했다. scene oracle은 scope마다 a=0.1 매칭으로 따로 보정했고, 두 scope의 매칭이 다른 sample은 19개였다.
6. GT에 미래에만 나타나는 객체가 없어서 `new_gt`는 0이다.

## 3. Fixed-ID 결과 (미래 pooled; 분모는 sample–instance 관측)

`cond` = 조건부(현재 매칭 성공 + 미래 GT 존재), `zero` = 현재 미매칭을 0점으로 처리.

| scope | a | 현재 GT | 현재 매칭 | coverage | 현재 matched IoU | Native cond | Oracle cond | Native zero | Oracle zero |
|---|---|---|---|---|---|---|---|---|---|
| ped_included | 0.1 | 62499 | 7949 | 12.72% | 0.2494 | 0.1599 | 0.1587 | 0.0215 | 0.0213 |
| ped_included | 0.2 | 62499 | 4482 | 7.17% | 0.3280 | 0.2096 | 0.2086 | 0.0159 | 0.0158 |
| ped_included | 0.3 | 62499 | 2309 | 3.69% | 0.4054 | 0.2554 | 0.2552 | 0.0099 | 0.0099 |
| ped_excluded | 0.1 | 48208 | 7930 | 16.45% | 0.2497 | 0.1601 | 0.1589 | 0.0275 | 0.0273 |
| ped_excluded | 0.2 | 48208 | 4482 | 9.30% | 0.3280 | 0.2096 | 0.2086 | 0.0203 | 0.0202 |
| ped_excluded | 0.3 | 48208 | 2309 | 4.79% | 0.4054 | 0.2554 | 0.2552 | 0.0127 | 0.0127 |

horizon별, a=0.1:

| 지표 | scope | 0.5s | 1.0s | 1.5s | 2.0s |
|---|---|---|---|---|---|
| eligible GT | incl / excl | 59918 / 46792 | 57620 / 45012 | 55076 / 43104 | 52640 / 41132 |
| matched-valid | incl / excl | 7868 / 7849 | 7719 / 7700 | 7508 / 7489 | 7202 / 7183 |
| Native cond | incl | 0.1836 | 0.1584 | 0.1496 | 0.1465 |
| Oracle cond | incl | 0.1815 | 0.1572 | 0.1489 | 0.1456 |
| Native cond | excl | 0.1838 | 0.1586 | 0.1498 | 0.1467 |
| Oracle cond | excl | 0.1817 | 0.1574 | 0.1491 | 0.1458 |
| success_rate_matched (Native / Oracle) | incl | 0.7317 / 0.7194 | 0.6431 / 0.6386 | 0.6087 / 0.6056 | 0.5971 / 0.5933 |
| eligible_gt_coverage | incl | 0.1313 | 0.1340 | 0.1363 | 0.1368 |
| current_gt_survival | incl | 0.9587 | 0.9219 | 0.8812 | 0.8423 |
| matched_gt_survival | incl | 0.9898 | 0.9711 | 0.9445 | 0.9060 |
| 회전한 pair (nonzero Δ) | incl | 3177 | 3366 | 3461 | 3413 |
| Δ=0 / yaw 없음 | incl | 4672 / 0 | 4333 / 1 | 4026 / 2 | 3764 / 6 |

a=0.2와 a=0.3의 horizon별 조건부 값:

| scope | a | variant | 0.5s | 1.0s | 1.5s | 2.0s |
|---|---|---|---|---|---|---|
| incl = excl | 0.2 | Native | 0.2416 | 0.2080 | 0.1954 | 0.1911 |
| incl = excl | 0.2 | Oracle | 0.2400 | 0.2071 | 0.1948 | 0.1903 |
| incl = excl | 0.3 | Native | 0.2963 | 0.2533 | 0.2363 | 0.2326 |
| incl = excl | 0.3 | Oracle | 0.2960 | 0.2533 | 0.2361 | 0.2321 |

### 3.1 Oracle gain과 bootstrap CI (a=0.1)

scene token 단위 bootstrap을 2000회, seed 20261109, 150개 scene으로 돌렸다. draw마다 충분통계를 다시 합산했다.

| scope | cond gain [95% CI] | zero gain [95% CI] |
|---|---|---|
| incl | −12.3e-4 [−18.9e-4, −7.5e-4] | −1.7e-4 [−2.6e-4, −1.0e-4] |
| excl | −12.4e-4 [−18.9e-4, −7.5e-4] | −2.1e-4 [−3.3e-4, −1.3e-4] |

### 3.2 회전 구간 (ped_included, a=0.1)

고정 2초 구간, 미래 pooled:

| 그룹 | eligible | matched | Native cond | Oracle cond | Native zero | Oracle zero |
|---|---|---|---|---|---|---|
| <5° | 153591 | 25521 | 0.1750 | 0.1745 | 0.0291 | 0.0290 |
| 5–15° | 5107 | 1612 | 0.0767 | 0.0708 | 0.0242 | 0.0224 |
| ≥15° | 5400 | 1564 | 0.0625 | 0.0528 | 0.0181 | 0.0153 |
| end_missing | 11692 | 1503 | 0.0996 | 0.0996 | 0.0128 | 0.0128 |
| yaw_missing | 250 | 21 | 0.1254 | 0.1254 | 0.0105 | 0.0105 |
| 전체 차량 | 176040 | 30221 | 0.1601 | 0.1589 | 0.0275 | 0.0273 |
| pedestrian (Native 유지) | 49214 | 76 | 0.0762 | 0.0762 | 0.0001 | 0.0001 |

- ≥15° 그룹의 horizon별 조건부 값(Native → Oracle): 0.5s 0.1438 → 0.1166, 1.0s 0.0595 → 0.0491, 1.5s 0.0290 → 0.0279, 2.0s 0.0176 → 0.0177.
- horizon별 구간(각 k의 |Δθ|로 다시 분류)의 미래 pooled 조건부 값: 5–15° 0.0702 → 0.0583, ≥15° 0.0264 → 0.0257.

### 3.3 미매칭 0점 Fixed-ID IoU 모음

`zero_unmatched_fixed_id_iou[k] = Σ fixed_iou_sum / Σ eligible_gt`. 분모(eligible GT)는 §3 horizon 표와 같고, a에 관계없이 같다. 합계는 ped_included 225254, ped_excluded 176040이다.

| scope | a | variant | 0.5s | 1.0s | 1.5s | 2.0s | 미래 pooled |
|---|---|---|---|---|---|---|---|
| included | 0.1 | Native | 0.0241 | 0.0212 | 0.0204 | 0.0200 | 0.0215 |
| included | 0.1 | Oracle | 0.0238 | 0.0211 | 0.0203 | 0.0199 | 0.0213 |
| included | 0.2 | Native | 0.0179 | 0.0157 | 0.0150 | 0.0147 | 0.0159 |
| included | 0.2 | Oracle | 0.0178 | 0.0156 | 0.0150 | 0.0146 | 0.0158 |
| included | 0.3 | Native | 0.0113 | 0.0098 | 0.0093 | 0.0091 | 0.0099 |
| included | 0.3 | Oracle | 0.0113 | 0.0098 | 0.0093 | 0.0091 | 0.0099 |
| excluded | 0.1 | Native | 0.0308 | 0.0271 | 0.0260 | 0.0256 | 0.0275 |
| excluded | 0.1 | Oracle | 0.0305 | 0.0269 | 0.0259 | 0.0255 | 0.0273 |
| excluded | 0.2 | Native | 0.0229 | 0.0201 | 0.0192 | 0.0188 | 0.0203 |
| excluded | 0.2 | Oracle | 0.0227 | 0.0200 | 0.0191 | 0.0187 | 0.0202 |
| excluded | 0.3 | Native | 0.0145 | 0.0126 | 0.0119 | 0.0117 | 0.0127 |
| excluded | 0.3 | Oracle | 0.0144 | 0.0126 | 0.0119 | 0.0116 | 0.0127 |

## 4. 전체 scene IoU / ordinary AP (rotation oracle)

- AP는 binary occupancy instance AP다. non-interpolated이고, IoU 0.1/0.2/0.3의 평균이며, GT가 있는 frame들의 macro 평균이다.
- Scene IoU는 horizon별 dataset-pooled 값이다.

| scope | variant | IoU 0s | 0.5s | 1.0s | 1.5s | 2.0s | IoU future avg | AP 0s | 0.5s | 1.0s | 1.5s | 2.0s | AP future avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incl | Native | 0.2157 | 0.1729 | 0.1512 | 0.1443 | 0.1409 | 0.1523 | 0.1107 | 0.0768 | 0.0619 | 0.0577 | 0.0558 | 0.0631 |
| incl | Oracle | 0.2157 | 0.1691 | 0.1470 | 0.1398 | 0.1361 | 0.1480 | 0.1107 | 0.0752 | 0.0603 | 0.0557 | 0.0535 | 0.0612 |
| excl | Native | 0.2176 | 0.1742 | 0.1523 | 0.1452 | 0.1419 | 0.1534 | 0.1508 | 0.1030 | 0.0850 | 0.0798 | 0.0769 | 0.0862 |
| excl | Oracle | 0.2176 | 0.1703 | 0.1481 | 0.1408 | 0.1370 | 0.1491 | 0.1508 | 0.1008 | 0.0829 | 0.0772 | 0.0738 | 0.0837 |

- Scene IoU future avg gain: incl −0.0043 [−0.0056, −0.0031], excl −0.0043 [−0.0057, −0.0032].
- AP future avg gain: incl −0.0019, excl −0.0025.
- 현재(0s)를 포함한 5시점 균등 평균(Avg_5h): IoU 0.1650 / 0.1663, AP 0.0726 / 0.0991 (incl / excl, Native).
- AP 유효 frame 수는 incl 5041/5031/5017/4998/4969, excl 4967/4956/4938/4912/4873이다. Native와 Oracle이 같다.
- 이 AP는 repo 자체 평가(`gmo_eval_instance` 로그)의 AP와 GT 전처리(visibility drop), candidate 집합, 집계 방식이 달라 직접 비교하지 않는다.

## 5. export 선택에 따른 차이 (08-16 export)

두 full export는 sample마다 선택된 query와 voxel이 대부분 다르다(앞 200개 sample 중 voxel이 같은 sample은 2개). 원인은 두 가지다.

- 08-16 export는 pedestrian-argmax query를 포함한다.
- run 간 추론이 비결정적이다. score가 0.0005 근처인 query가 많아 argmax가 쉽게 흔들린다. 같은 기간에 추론 코드(`gmo_model.py`, `gmo_query_instances.py`)는 바뀌지 않았다.

| export | scope | 현재 매칭 | coverage | matched IoU | Native / Oracle cond | Native / Oracle zero | scene IoU fut (N / O) | AP fut (N / O) |
|---|---|---|---|---|---|---|---|---|
| 08-28 (주) | incl | 7949 | 12.72% | 0.2494 | 0.1599 / 0.1587 | 0.0215 / 0.0213 | 0.1523 / 0.1480 | 0.0631 / 0.0612 |
| 08-16 | incl | 8955 | 14.33% | 0.2424 | 0.1504 / 0.1493 | 0.0227 / 0.0226 | 0.1509 / 0.1466 | 0.0633 / 0.0615 |
| 08-28 (주) | excl | 7930 | 16.45% | 0.2497 | 0.1601 / 0.1589 | 0.0275 / 0.0273 | 0.1534 / 0.1491 | 0.0862 / 0.0837 |
| 08-16 | excl | 7975 | 16.54% | 0.2492 | 0.1600 / 0.1587 | 0.0276 / 0.0274 | 0.1505 / 0.1463 | 0.0792 / 0.0769 |

08-16 export는 matched pedestrian이 3634쌍이고, 두 scope의 a=0.1 매칭이 다른 sample이 979개다. 그래도 oracle gain의 방향과 크기는 두 export에서 같다.

## 6. 검증

| 항목 | 결과 |
|---|---|
| 단위 테스트 10개 (EfficientOCF와 같은 테스트) | 통과 |
| 0° renderer (첫 sample과 `sha256 % 67 == 0` 표본 433쌍) | IoU 최대 오차 0 |
| 강제 0° 전체 실행(`full5119_zero`): Fixed-ID 전 지표, scene I/P/G, AP | Native와 정확히 일치 |
| Paired 분모: 상태별 합계 = matched-valid; scene GT 수와 AP frame 수 | 일치 |
| GT class 일관성 | 불일치 0 |
| 예측·GT 정렬 (첫 sample) | BEV IoU 0.27 (x/y를 바꾸면 0.02), 3D IoU 0.21 |
| yaw 부호 | EfficientOCF 평가에서 같은 GT로 확인(GT 회전 후 IoU 0.43 → +Δ 0.62 / −Δ 0.30) |

## 7. 해석과 한계

- **SparseOcc에서는 이 oracle 변형이 미래 IoU를 낮춘다.** Fixed-ID 조건부 −0.0012, scene IoU −0.0043이고, CI가 모두 0 아래에 있다. 하락은 회전량이 큰 그룹에 몰려 있다(≥15°, 0.5s에서 0.144 → 0.117).
- **진단** (2초 시점 |Δ|≥15°인 matched 차량 391쌍):
  - GT 현재 형상을 static copy한 뒤 제자리에서 회전하면 GT 미래와의 IoU가 0.301 → 0.315(0.5s), 0.024 → 0.026(2.0s)로 **오른다**. 따라서 이동 없이 제자리에서 돌리는 것 자체가 원인은 아니다.
  - 같은 쌍에서 예측 mask의 voxel 수는 GT의 2.8배(중앙값)이고, mask 중심과 GT 중심의 거리는 **중앙값 7.7 m(75 percentile 13.3 m)**다. 여러 객체나 배경을 덮는 큰 query mask가 IoU 0.1 이상으로 매칭되므로, GT 객체에서 먼 mask 중심을 기준으로 돌리면 겹치던 부분이 밀려난다.
  - 2.0s에는 예측 mask의 79%가 GT 미래와 이미 겹치지 않는다(static copy). 그래서 회전의 영향이 거의 없다.
- 이 해석은 진단에 근거한 추정이다. 회전 중심을 예측 query center로 할 수 없는 adapter 한계(§8.2)가 결과에 직접 영향을 준다. 이 oracle은 upper bound가 아니며, 음수 gain을 회전 모델링이 불필요하다는 근거로 쓰지 않는다.
- 현재 coverage(12.7%)가 낮으므로 조건부 값만 따로 인용하지 않는다. Fixed-ID IoU는 ID switch 횟수가 아니고, 일반 scene AP는 Fixed-ID AP가 아니다.

## 8. 재현

```bash
cd /home/user/jhh/Projects/SparseOcc
PY=/home/user/anaconda3/envs/eo_jhh/bin/python
E=outputs/e6_yaw_oracle
$PY -m pytest -q tools_custom/test_e6_yaw_oracle.py
$PY tools_custom/eval_e6_yaw_oracle.py run --pred-dir outputs/original_sparse_prediction_full \
    --drop-ids $E/gt_visibility_drop_ids_val.json --out $E/full5119 --workers 16
$PY tools_custom/eval_e6_yaw_oracle.py summarize --out $E/full5119
# 세 설정 일괄 실행(주 / 강제 0° / 08-16 export): setsid nohup $E/run_full.sh > $E/run_full.log 2>&1 &
```

- 각 output에는 `manifest.json`, `summary.json`, `assignments.csv`, `oracle_cache.csv`, `cache/*.pkl`(sample 단위, 원자적 저장, 재개 가능)이 있다.
- 설정 fingerprint가 다르면 실행을 거부한다.
