# Проверка сравнения моделей

Ориентиры:
     split             model    rows  goa       AP  ROC_AUC    Brier
validation     airport_prior 1619847 4984 0.003672 0.565240 0.003067
validation          geometry 1619847 4984 0.003732 0.569868 0.003067
validation geometry_aircraft 1619847 4984 0.004144 0.585495 0.003067
      test     airport_prior  546577 1739 0.003808 0.563901 0.003171
      test          geometry  546577 1739 0.003759 0.562344 0.003171
      test geometry_aircraft  546577 1739 0.003980 0.572297 0.003171

Одинаковый бюджет test:
            model  budget_test  positions  expected_tp  expected_fp   recall  precision  boundary_tie_rows  boundary_tie_weight
    airport_prior        0.001        546     2.309036   543.690964 0.001328   0.004229              72121             0.007571
    airport_prior        0.005       2732    11.553639  2720.446361 0.006644   0.004229              72121             0.037881
    airport_prior        0.010       5465    23.111507  5441.888493 0.013290   0.004229              72121             0.075775
    airport_prior        0.050      27328   115.570222 27212.429778 0.066458   0.004229              72121             0.378919
         geometry        0.001        546     1.000000   545.000000 0.000575   0.001832                  1             1.000000
         geometry        0.005       2732     9.000000  2723.000000 0.005175   0.003294                  1             1.000000
         geometry        0.010       5465    18.000000  5447.000000 0.010351   0.003294                  1             1.000000
         geometry        0.050      27328   108.000000 27220.000000 0.062105   0.003952                  1             1.000000
geometry_aircraft        0.001        546     3.000000   543.000000 0.001725   0.005495                  1             1.000000
geometry_aircraft        0.005       2732    10.000000  2722.000000 0.005750   0.003660                  1             1.000000
geometry_aircraft        0.010       5465    14.000000  5451.000000 0.008051   0.002562                  1             1.000000
geometry_aircraft        0.050      27328   124.000000 27204.000000 0.071305   0.004537                  1             1.000000

Чувствительность парных интервалов:
 block_weeks                        comparison  metric  estimate     lower    upper  valid_replicates  observed_weeks
           1      geometry_aircraft - geometry      AP  0.000221  0.000027 0.000461              1000              23
           1          geometry - airport_prior      AP -0.000049 -0.000215 0.000130              1000              23
           1 geometry_aircraft - airport_prior      AP  0.000172  0.000014 0.000431              1000              23
           1      geometry_aircraft - geometry ROC_AUC  0.009953  0.002152 0.018186              1000              23
           1          geometry - airport_prior ROC_AUC -0.001557 -0.009444 0.006079              1000              23
           1 geometry_aircraft - airport_prior ROC_AUC  0.008396  0.000866 0.016205              1000              23
           2      geometry_aircraft - geometry      AP  0.000221  0.000031 0.000474              1000              23
           2          geometry - airport_prior      AP -0.000049 -0.000210 0.000156              1000              23
           2 geometry_aircraft - airport_prior      AP  0.000172  0.000026 0.000421              1000              23
           2      geometry_aircraft - geometry ROC_AUC  0.009953  0.001677 0.018409              1000              23
           2          geometry - airport_prior ROC_AUC -0.001557 -0.008822 0.007986              1000              23
           2 geometry_aircraft - airport_prior ROC_AUC  0.008396  0.002180 0.015363              1000              23
           4      geometry_aircraft - geometry      AP  0.000221  0.000021 0.000487              1000              23
           4          geometry - airport_prior      AP -0.000049 -0.000208 0.000142              1000              23
           4 geometry_aircraft - airport_prior      AP  0.000172  0.000053 0.000408              1000              23
           4      geometry_aircraft - geometry ROC_AUC  0.009953  0.002706 0.017350              1000              23
           4          geometry - airport_prior ROC_AUC -0.001557 -0.008973 0.007283              1000              23
           4 geometry_aircraft - airport_prior ROC_AUC  0.008396  0.003441 0.013885              1000              23

Старые модели, прогнозы и F1-пороги не менялись. Обучения нет.
Дополнительные числовые пороги определены по scores validation без меток.
Ранговый бюджет test — описательная оценка, не заранее доступный числовой порог.
Ориентир аэропорта и новые сравнения введены после просмотра test.
