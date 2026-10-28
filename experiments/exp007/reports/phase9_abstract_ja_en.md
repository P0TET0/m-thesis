# 酸化物の細分類が電気伝導率予測に与える影響

English title: Effect of Finer Oxide Classification on Electrical Conductivity Prediction Using Starrydata2

## 日本語要旨

Starrydata2のゼーベック係数と電気伝導率から推定したフェルミ準位情報を用いる予測法では、酸化物群の低電気伝導率領域に大きな誤差が残っていた。本研究では、従来一括していた酸化物を組成表記に基づくoxide_v1へ細分類し、材料分類以外の条件を固定して電気伝導率予測を再計算した。試料単位で分離したホールドアウトのうち、両分類で予測可能な2,984行、552試料、395論文を主解析とした。行均等MAEは1.023から0.875へ14.5%低下し、差の試料クラスターブートストラップ95%信頼区間は-0.243〜-0.066であった。RMSEは1.587から1.359へ低下し、Factor-5精度は54.7%から62.4%、Factor-10精度は67.3%から71.7%へ向上した。温度ビン幅、参照支持数、集約法、重み付け、外れ値除外、一群除外を変えた内部感度分析でも全体の改善方向は維持された。一方、予測可能率は100.0%から98.4%へ低下し、Mixed-anion O + As/Pでは改善しなかった。以上より、組成に基づく材料分類の粒度は電気伝導率予測誤差の一要因であり、酸化物の細分類は大きな誤差の抑制に有効である。ただし、未知論文への一般化、結晶構造、ドーピング、焼結条件の影響は今後の検証課題である。

キーワード: Starrydata2、電気伝導率、フェルミ準位、材料分類、酸化物、機械学習

## English abstract

Electrical conductivity predictions using Fermi-level information inferred from Seebeck coefficient and conductivity data in Starrydata2 retained large errors for oxides, particularly in the low-conductivity region. We tested whether a finer composition-based oxide taxonomy could reduce these errors while keeping the sample holdout, reference-data subset, temperature protocol, aggregation method, and support thresholds fixed. The primary paired analysis included 2,984 rows from 552 samples and 395 papers that were predictable under both the broad-oxide baseline and oxide_v1. Row-weighted MAE decreased from 1.023 to 0.875, a 14.5% reduction, with a sample-cluster bootstrap 95% confidence interval of -0.243 to -0.066 for the paired difference. RMSE decreased from 1.587 to 1.359. Factor-5 accuracy increased from 54.7% to 62.4%, and factor-10 accuracy increased from 67.3% to 71.7%. The overall improvement persisted across predefined changes in temperature-bin width, reference support, aggregation, weighting, extreme-row inclusion, and leave-one-group-out analyses. Prediction coverage decreased from 100.0% to 98.4%, and Mixed-anion O + As/P did not improve. These results indicate that material-classification granularity contributes to conductivity-prediction error and that finer oxide grouping reduces large errors, although external DOI-held-out validation and explicit structural and processing descriptors remain necessary.

Keywords: Starrydata2, electrical conductivity, Fermi level, material classification, oxide, machine learning
