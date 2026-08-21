# exp005 / exp006 ZT比較 引き継ぎ資料

作成日: 2026-08-19  
元プロジェクト: `C:\Users\miots\m-thesis\m-thesis`

## 1. この資料の目的

exp006で予測済みの電気伝導率 `sigma_pred` からPF・ZTを計算し、exp005のDF法・ML法と公平に比較するまでの会話、実装、判断、主要結果を別チャットへ引き継ぐための要約である。

このZIPは生のチャットログではなく、別チャットが作業を継続できるように整理した構造化ハンドオフである。主要スクリプト、軽量な集計CSV、説明文、比較図を同梱している。50 MBを超える行単位CSVや元データはZIPから除外し、元プロジェクト内の場所を記載した。

## 2. 会話で行ったこと

ユーザーから順に、以下の確認・依頼があった。

1. exp006の予測済み `sigma_pred` から `PF_pred` と `ZT_pred` を計算し、図として出力。
2. 出力フォルダと、ZTデータ数が少ない理由を確認。
3. ZT数を増やす方法を検討。
4. exp005の比較方法、25 K温度ビン結合の有無、同一試料・±2 K最近傍温度結合時のデータ数を調査。
5. exp005 Step10相当として元の全物性曲線から温度対応を再構築する方法を採用し、再実装。
6. 材料系分類ごとのZT比較を作成し、データ数を増加。
7. exp005とexp006で共通する試料・DOI、同一元物性曲線、±2 K、同一評価単位による再集計を作成。
8. exp005 ML法・DF法・exp006を比較し、exp006がexp005 MLより低性能と推測されることを可視化。
9. `ZT < 1`に限定しないネイティブ全範囲比較を作成。
10. 推奨案として、共通試料集合と共通DOI foldを用いる5-fold DOI-group交差検証を新規作成。

## 3. PF・ZTの計算条件

各行のexp006予測電気伝導率から、次式で計算した。

```text
PF_pred = S_obs^2 * sigma_pred
ZT_pred = S_obs^2 * sigma_pred * T / kappa_obs
```

- `S_obs`: exp006行のSeebeck係数。`microV/K`から`V/K`へ変換。
- `T`: exp006の評価温度。
- `kappa_obs`: 元のStep5全物性曲線から、同一試料・評価温度±2 K以内の最近傍値を使用。
- `ZT_obs`: 同じStep5全物性曲線から、同一試料・評価温度±2 K以内の最近傍reported ZTを使用。
- kappaとreported ZTは独立に最近傍結合し、補間は行わない。

## 4. exp005の温度対応について得られた結論

- exp005 Step12の電気伝導率基準曲線では25 Kビンを使用している。
- ただしreported ZTとの最終比較を単純な25 Kビン結合だけで行っていた、という理解は正確ではない。
- 今回の公平比較では、元のStep5全物性曲線をStep10相当の規則で再展開し、同一試料・±2 K最近傍としてkappaとreported ZTを対応させた。
- exp005のStep12互換prefactorは、元のStep5電気系曲線、25 Kビン、leave-one-sample-out階層を使って再構築した。

## 5. これまで作成した比較

### 5.1 共通試料・DOIの厳密ペア比較

元プロジェクト出力:

```text
data/output/exp005_exp006_common_cohort_zt_comparison
```

exp005の既存DOI-test集合とexp006の共通試料を使った厳密比較。reported ZTを持ち、全手法で予測可能な行に限定すると、383行・109試料・83 DOIとなり、`ZT >= 1`が0件だった。このためMAEが小さく見えやすく、高ZT性能の比較には不十分だった。

### 5.2 exp005 DF・exp005 ML・exp006の3手法共通比較

元プロジェクト出力:

```text
data/output/exp005_ml_df_exp006_common_cohort_zt_comparison
```

同じ厳密共通集合で3手法を比較。ただし高ZT試料が含まれず、全範囲性能の主張には使いにくい。

### 5.3 ネイティブ全範囲比較

元プロジェクト出力:

```text
data/output/exp005_exp006_native_full_range_zt_comparison
```

各手法の既存ネイティブ評価集合をそのまま使った非ペア比較。`ZT < 1`制限を設けない。

主なsample-max結果:

| 方法 | 行数 | 試料数 | 高ZT試料数 | log10 RMSE | Factor-2率 | Spearman | 高ZT F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp005 DF | 6,381 | 1,115 | 268 | 0.812 | 0.934 | 0.926 | 0.808 |
| exp005 ML | 6,381 | 1,115 | 268 | 1.285 | 0.546 | 0.684 | 0.570 |
| exp006 | 2,279 | 619 | 10 | 1.559 | 0.371 | 0.440 | 0.138 |

注意: 評価集合が異なるため、性能傾向の把握には使えるが、厳密な手法間比較ではない。

## 6. 推奨版: 共通DOI-group 5-fold交差検証

元プロジェクト出力:

```text
data/output/exp005_exp006_common_doi_group_cv
```

再実行:

```powershell
python run_common_doi_group_cv_exp005_exp006.py --overwrite
```

### 6.1 共通条件

- exp005 Step18で学習可能、かつexp006 Step5Aに存在する試料の交差集合を使用。
- 正規化DOIがexp005とexp006で一致することを要求。
- 共通学習母集団: 9,764試料・2,855 DOI。
- DOI単位の5-fold交差検証。同じDOIは必ず同じfold。
- DOIの学習・評価間重複は0件で検証済み。
- reported max ZTが1以上の試料を含むDOIをfold間で分散。
- fold作成時の評価可能高ZT試料は53件で、各foldに10～11件。
- 全3手法の予測がそろう厳密共通評価は10,661行・2,881試料・954 DOI。
- 厳密共通評価の高ZT試料は39件。fold別に6、8、9、8、8件。
- `ZT < 1`のカットは行っていない。厳密共通評価のreported sample-max ZT範囲は約-0.001～2.276。
- 同じ温度、S、kappa、reported ZTで3手法を評価。

### 6.2 各手法

- exp005 ML: 過去のGradientBoostingRegressor設定を使用。fold内学習データだけで欠損補完とゼロ分散特徴除外を実施し、未知DOIにOOF予測。
- exp006: fold内学習行だけから `material_group + carrier + 25 K bin` のsample-median `sigma0`参照曲線を再構築し、未知DOIにOOF予測。現在のStep5Aではmaterial groupが実質`unknown_material_family`一種類なので、実態はcarrier + 25 K bin中心の参照になる。
- exp005 DF: 評価試料自身のStep12 fitted tau_effを使うper-sample fit参照。foldで評価行を分割しているが、未知試料予測モデルではない。ML・exp006と同列のデプロイ可能OOF予測として解釈しない。

### 6.3 主要結果

| 方法 | sample-max log10 RMSE↓ | Factor-2率↑ | Spearman↑ | 高ZT F1↑ |
|---|---:|---:|---:|---:|
| exp005 DF | 0.427 | 0.926 | 0.934 | 0.318 |
| exp005 ML | 0.539 | 0.542 | 0.620 | 0.083 |
| exp006 | 0.607 | 0.507 | 0.482 | 0.155 |

### 6.4 解釈

- exp006はexp005 MLより、log10 RMSE、Factor-2率、Spearmanで悪い。
- 一方、高ZT F1はexp006の0.155がexp005 MLの0.083を上回る。
- したがって「exp006はすべての指標でexp005 MLより悪い」とは言えない。
- 論文・発表では「共通DOI-group OOF評価において、exp006はexp005 MLより総合的な回帰精度と順位性能が低かった。ただし高ZT二値判定F1では逆転した」と表現するのが安全。
- exp005 DFは最も高い順位相関とFactor-2率を示すが、per-sample fitting referenceであり、未知DOI予測のML・exp006より有利な条件。

## 7. 図の読み方

同梱した主な図:

- `results/common_doi_cv/figure_common_doi_cv_performance_summary.png`: 主要4指標の比較。
- `results/common_doi_cv/figure_common_doi_cv_sample_max_parity.png`: 見やすさを優先したsample-max parity。
- `results/common_doi_cv/figure_common_doi_cv_sample_max_parity_full_range.png`: 点を切らず、`ZT < 1`制限なしで全範囲表示。DFの最大予測ZTが80.7のため縦軸はsymlog。
- `results/common_doi_cv/figure_common_doi_cv_fold_stability_and_balance.png`: foldバランスとfold別性能。

## 8. データ数に関する重要点

- exp006の元のZT件数が少なかった主因は、予測sigma行と同じ試料・温度でkappaおよびreported ZTがそろう必要があるため。
- 25 Kビン結合だけで増やすと温度差の大きな誤対応を招く可能性がある。
- 推奨した増加方法は、元の全物性曲線を展開し、同一試料・±2 K最近傍でkappaとZTを独立結合する方法。
- 共通DOI CVでは学習母集団を既存のexp005 DOI-testだけに限定せず、exp005 Step18とexp006 Step5Aの全共通試料へ広げたため、高ZT試料を含む比較が可能になった。
- 53件の高ZT評価可能試料から厳密共通評価で39件になるのは、reported ZT・kappa・exp005 prefactor・exp005 ML・exp006 OOF予測がすべて同一行で利用可能という共通条件を課したため。

## 9. 次のチャットへの推奨依頼文

以下を新しいチャットへ貼り付けると継続しやすい。

```text
添付ZIPのHANDOFF.mdを読み、exp005 DF・exp005 ML・exp006のZT比較作業を引き継いでください。主解析は共通DOI-group 5-fold CVです。まず同梱のREADME、metrics CSV、図、run_common_doi_group_cv_exp005_exp006.pyを確認し、exp006がexp005 MLより総合回帰性能で低いという結論が妥当かレビューしてください。DFはper-sample fitting referenceで、純粋なOOFモデルではない点を維持してください。
```

## 10. ZIPに含めていない大容量ファイル

以下は元プロジェクトに残っている。

```text
data/output/exp005_exp006_common_doi_group_cv/common_doi_cv_row_predictions.csv
data/output/exp005_exp006_common_doi_group_cv/common_doi_cv_fold_assignments.csv
data/output/exp005_exp006_common_doi_group_cv/common_doi_cv_sample_max_predictions.csv
data/output/exp005_exp006_common_doi_group_cv/exp005_ml_oof_tau_predictions.csv
data/output/exp006_pf_zt_from_sigma_prediction_full_curves/exp006_pf_zt_row_predictions.csv
data/output/starrydata2_step5_core_properties/candidate_core_curves_step5.csv
experiments/exp006/data/processed/step5a_validation_rows_with_splits.parquet
```

## 11. 再現性上の注意

- スクリプトは元プロジェクトの相対パスを前提としている。
- ZIPを別環境へ移しただけでは元データがないため完全再実行できない。別チャットを同じCodex workspaceで開く場合はそのまま再実行可能。
- 出力は現在git未追跡のものを含むため、別branch/worktreeへ移ると見えない可能性がある。同じworkspaceでの引き継ぎを推奨する。

