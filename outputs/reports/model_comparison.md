# Algorithm comparison (mean across crimes)

| algorithm     |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:--------------|-----------:|------------:|---------:|-------:|----------:|
| random_forest |     0.8824 |      0.8458 |   0.569  | 0.6431 |    0.9318 |
| xgboost       |     0.8825 |      0.82   |   0.5876 | 0.6573 |    0.924  |
| logreg        |     0.8505 |      0.6808 |   0.6826 | 0.6663 |    0.8735 |

## Per-crime detail

| crime                  | algorithm     |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:-----------------------|:--------------|-----------:|------------:|---------:|-------:|----------:|
| aggr_robbery           | logreg        |     0.94   |      0.7917 |   0.95   | 0.8636 |    0.9762 |
| aggr_robbery           | random_forest |     0.9667 |      0.9808 |   0.85   | 0.9107 |    0.9897 |
| aggr_robbery           | xgboost       |     0.9667 |      0.9808 |   0.85   | 0.9107 |    0.9903 |
| arson                  | logreg        |     0.88   |      0.7164 |   0.7385 | 0.7273 |    0.8996 |
| arson                  | random_forest |     0.9133 |      0.9333 |   0.6462 | 0.7636 |    0.961  |
| arson                  | xgboost       |     0.91   |      0.8654 |   0.6923 | 0.7692 |    0.9616 |
| assault_gbh            | logreg        |     0.9267 |      0.7746 |   0.9016 | 0.8333 |    0.9698 |
| assault_gbh            | random_forest |     0.96   |      0.9804 |   0.8197 | 0.8929 |    0.9824 |
| assault_gbh            | xgboost       |     0.97   |      0.9643 |   0.8852 | 0.9231 |    0.9762 |
| attempted_murder       | logreg        |     0.8733 |      0.7895 |   0.5    | 0.6122 |    0.8335 |
| attempted_murder       | random_forest |     0.94   |      0.85   |   0.85   | 0.85   |    0.9776 |
| attempted_murder       | xgboost       |     0.92   |      0.8462 |   0.7333 | 0.7857 |    0.9756 |
| attempted_sexoff       | logreg        |     0.82   |      0.6349 |   0.5634 | 0.597  |    0.8005 |
| attempted_sexoff       | random_forest |     0.8733 |      0.8511 |   0.5634 | 0.678  |    0.9118 |
| attempted_sexoff       | xgboost       |     0.8733 |      0.7619 |   0.6761 | 0.7164 |    0.8846 |
| burglary_nonres        | logreg        |     0.8433 |      0.7419 |   0.371  | 0.4946 |    0.8691 |
| burglary_nonres        | random_forest |     0.85   |      0.9474 |   0.2903 | 0.4444 |    0.9346 |
| burglary_nonres        | xgboost       |     0.87   |      0.9259 |   0.4032 | 0.5618 |    0.9247 |
| burglary_res           | logreg        |     0.8167 |      0.5333 |   0.6667 | 0.5926 |    0.8902 |
| burglary_res           | random_forest |     0.82   |      0.875  |   0.1167 | 0.2059 |    0.9399 |
| burglary_res           | xgboost       |     0.8233 |      0.8889 |   0.1333 | 0.2319 |    0.9545 |
| carjacking             | logreg        |     0.8967 |      0.6747 |   0.9333 | 0.7832 |    0.9647 |
| carjacking             | random_forest |     0.97   |      0.8923 |   0.9667 | 0.928  |    0.9859 |
| carjacking             | xgboost       |     0.95   |      0.8689 |   0.8833 | 0.876  |    0.9885 |
| cash_transit_robbery   | logreg        |     0.8    |      1      |   0.8    | 0.8889 |  nan      |
| cash_transit_robbery   | random_forest |     0.9833 |      1      |   0.9833 | 0.9916 |  nan      |
| cash_transit_robbery   | xgboost       |     0.99   |      1      |   0.99   | 0.995  |  nan      |
| commercial_crime       | logreg        |     0.9667 |      1      |   0.8333 | 0.9091 |    0.9642 |
| commercial_crime       | random_forest |     0.93   |      0.9149 |   0.7167 | 0.8037 |    0.9806 |
| commercial_crime       | xgboost       |     0.9667 |      0.9464 |   0.8833 | 0.9138 |    0.9762 |
| common_assault         | logreg        |     0.9033 |      0.9189 |   0.5667 | 0.701  |    0.9431 |
| common_assault         | random_forest |     0.9367 |      0.9556 |   0.7167 | 0.819  |    0.9815 |
| common_assault         | xgboost       |     0.9467 |      0.9783 |   0.75   | 0.8491 |    0.9844 |
| common_robbery         | logreg        |     0.8967 |      0.8372 |   0.6    | 0.699  |    0.9017 |
| common_robbery         | random_forest |     0.8867 |      0.9333 |   0.4667 | 0.6222 |    0.9636 |
| common_robbery         | xgboost       |     0.91   |      0.9459 |   0.5833 | 0.7216 |    0.959  |
| contact_sexoff         | logreg        |     0.6233 |      0.4218 |   0.6889 | 0.5232 |    0.7249 |
| contact_sexoff         | random_forest |     0.7633 |      0.6939 |   0.3778 | 0.4892 |    0.7767 |
| contact_sexoff         | xgboost       |     0.7133 |      0.5312 |   0.3778 | 0.4416 |    0.7301 |
| drug_crime             | logreg        |     0.88   |      0.7308 |   0.6333 | 0.6786 |    0.9078 |
| drug_crime             | random_forest |     0.8633 |      0.913  |   0.35   | 0.506  |    0.9565 |
| drug_crime             | xgboost       |     0.8833 |      0.9032 |   0.4667 | 0.6154 |    0.9445 |
| dui                    | logreg        |     0.7767 |      0.4474 |   0.2698 | 0.3366 |    0.6417 |
| dui                    | random_forest |     0.8067 |      0.7778 |   0.1111 | 0.1944 |    0.8424 |
| dui                    | xgboost       |     0.7967 |      0.5833 |   0.1111 | 0.1867 |    0.8243 |
| illegal_firearms       | logreg        |     0.88   |      0.6857 |   0.7742 | 0.7273 |    0.921  |
| illegal_firearms       | random_forest |     0.8967 |      0.7385 |   0.7742 | 0.7559 |    0.9648 |
| illegal_firearms       | xgboost       |     0.9    |      0.7759 |   0.7258 | 0.75   |    0.9645 |
| kidnapping             | logreg        |     0.66   |      0.3417 |   0.6406 | 0.4457 |    0.6892 |
| kidnapping             | random_forest |     0.8433 |      0.7179 |   0.4375 | 0.5437 |    0.8932 |
| kidnapping             | xgboost       |     0.7667 |      0.4    |   0.1875 | 0.2553 |    0.8112 |
| malicious_damage       | logreg        |     0.9367 |      0.7808 |   0.95   | 0.8571 |    0.9852 |
| malicious_damage       | random_forest |     0.9367 |      1      |   0.6833 | 0.8119 |    0.9921 |
| malicious_damage       | xgboost       |     0.95   |      0.9787 |   0.7667 | 0.8598 |    0.9933 |
| murder                 | logreg        |     0.87   |      0.618  |   0.9167 | 0.7383 |    0.969  |
| murder                 | random_forest |     0.93   |      0.7532 |   0.9667 | 0.8467 |    0.9879 |
| murder                 | xgboost       |     0.9367 |      0.7662 |   0.9833 | 0.8613 |    0.9902 |
| other_theft            | logreg        |     0.86   |      0.75   |   0.45   | 0.5625 |    0.8669 |
| other_theft            | random_forest |     0.8433 |      0.9333 |   0.2333 | 0.3733 |    0.9206 |
| other_theft            | xgboost       |     0.8533 |      0.9444 |   0.2833 | 0.4359 |    0.9047 |
| police_detected_sexoff | logreg        |     0.6267 |      0.3636 |   0.4878 | 0.4167 |    0.5436 |
| police_detected_sexoff | random_forest |     0.72   |      0.4615 |   0.1463 | 0.2222 |    0.6765 |
| police_detected_sexoff | xgboost       |     0.73   |      0.5135 |   0.2317 | 0.3193 |    0.6988 |
| rape                   | logreg        |     0.93   |      0.8545 |   0.7833 | 0.8174 |    0.958  |
| rape                   | random_forest |     0.9633 |      0.9153 |   0.9    | 0.9076 |    0.9906 |
| rape                   | xgboost       |     0.9667 |      0.9167 |   0.9167 | 0.9167 |    0.9878 |
| robbery_nonres         | logreg        |     0.8    |      0.5111 |   0.7419 | 0.6053 |    0.85   |
| robbery_nonres         | random_forest |     0.84   |      0.6129 |   0.6129 | 0.6129 |    0.8717 |
| robbery_nonres         | xgboost       |     0.83   |      0.5902 |   0.5806 | 0.5854 |    0.8765 |
| robbery_res            | logreg        |     0.9433 |      0.8209 |   0.9167 | 0.8661 |    0.9711 |
| robbery_res            | random_forest |     0.9533 |      0.8833 |   0.8833 | 0.8833 |    0.9878 |
| robbery_res            | xgboost       |     0.9533 |      0.8833 |   0.8833 | 0.8833 |    0.9874 |
| sexual_assault         | logreg        |     0.84   |      0.6522 |   0.4839 | 0.5556 |    0.9205 |
| sexual_assault         | random_forest |     0.8767 |      0.7778 |   0.5645 | 0.6542 |    0.9568 |
| sexual_assault         | xgboost       |     0.89   |      0.8085 |   0.6129 | 0.6972 |    0.9578 |
| shoplifting            | logreg        |     0.86   |      0.625  |   0.75   | 0.6818 |    0.8848 |
| shoplifting            | random_forest |     0.9067 |      0.9444 |   0.5667 | 0.7083 |    0.9681 |
| shoplifting            | xgboost       |     0.8967 |      0.8919 |   0.55   | 0.6804 |    0.973  |
| stock_theft            | logreg        |     0.86   |      0.6849 |   0.7246 | 0.7042 |    0.8438 |
| stock_theft            | random_forest |     0.9    |      0.7826 |   0.7826 | 0.7826 |    0.9245 |
| stock_theft            | xgboost       |     0.8967 |      0.7794 |   0.7681 | 0.7737 |    0.8994 |
| theft_from_vehicle     | logreg        |     0.9233 |      0.8491 |   0.75   | 0.7965 |    0.9672 |
| theft_from_vehicle     | random_forest |     0.8833 |      1      |   0.4167 | 0.5882 |    0.9864 |
| theft_from_vehicle     | xgboost       |     0.8933 |      1      |   0.4667 | 0.6364 |    0.9582 |
| truck_hijacking        | logreg        |     0.7667 |      0.5507 |   0.4935 | 0.5205 |    0.7711 |
| truck_hijacking        | random_forest |     0.7833 |      0.5909 |   0.5065 | 0.5455 |    0.7923 |
| truck_hijacking        | xgboost       |     0.77   |      0.5909 |   0.3377 | 0.4298 |    0.786  |
| vehicle_theft          | logreg        |     0.8633 |      0.6418 |   0.7167 | 0.6772 |    0.9032 |
| vehicle_theft          | random_forest |     0.8333 |      0.9167 |   0.1833 | 0.3056 |    0.9251 |
| vehicle_theft          | xgboost       |     0.86   |      0.95   |   0.3167 | 0.475  |    0.9337 |