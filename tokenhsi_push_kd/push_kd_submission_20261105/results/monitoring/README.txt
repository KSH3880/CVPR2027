학습 중 감시 평가(validation seed 0) 집계표. 서버별로 나뉘어 있습니다.
- summary_vast.tsv  : RTX 4090. A(kd0p001), B(seed42) 500~5800, C 일부, B(seed43)
- summary_vast2.tsv : RTX 4080 SUPER. D(fixed0p0128), C(lagrange_feat), E 초반
- summary_vast3.tsv : RTX 6000 Ada. B(seed42) 5800~12100, E
열: run, epoch, SIT, CARRY, CLIMB 성공률, PUSH final, ever, native final, 낙상률, 종료 3D 거리 중앙값(m), mode
mode: full = 기존 과제 512 x 3 반복, quick = 512 x 1 반복(추이 확인용). PUSH 평가는 두 모드가 같습니다(512 장면 = 1,024명).
