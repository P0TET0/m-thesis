# Step18元素抽出の掲載結果への影響範囲

確認日：2026-10-26。既存処理を変更せず、保存データの読み取りと集計だけを実施した。モデルの読み込み・予測・再学習は行っていない。新規の監査用スクリプト・集計表・本報告書だけを追加した。

**確認できたのは、掲載系の学習・評価入力に元素特徴の相違が含まれる範囲である。性能低下の量、修正後の性能、手法順位の変化は確認していない。** 取りこぼしは旧ML候補で4,421試料、図2・6共通で31試料、図4で859試料、図7掲載で127試料に確認された。曖昧な表記等は要確認として別集計し、相違なしには含めていない。

## 1. 比較方法と対象外にした表記

### 独立した化学式の読み取り規則

Step9の抽出結果は正解として使っていない。元のStep18入力であるmetadataのcompositionを、独立した118元素記号辞書と次の完全一致文法で読む。式の一部だけを正規表現で拾って成功とする判定はしない。

```text
式      := 項の1回以上の連続
項      := 元素記号 [数値係数]
元素記号 := 大文字1字 + 必要なら小文字1字（118元素の辞書に含まれるもの）
数値係数 := 非負の整数または十進小数
係数省略 := 1
含まれる元素 := 係数を同元素ごとに合計した値が正の元素
```

外側の空白のみ除去する。整数・小数はDecimalで読み、丸め閾値を導入しない。例：`CoSb3`→{Co,Sb}、`SrTiO3`→{Sr,Ti,O}、`Bi0.5Sb1.5Te3`→{Bi,Sb,Te}。化学量論的な電荷中性、相の実在性、元論文の組成の正しさまでは判定していない。「明確」は**保存表記をこの文法で最後まで読み、元素集合を一意に得られた**という意味である。

係数0の元素は含まれないものとする。これは隣接記号の取りこぼしとは別の比較規則であり、後述の表では分けて数える。例えば`Ba7.96Ga15.9In0Sn30.1`のInを旧Step18が拾う相違は「0係数の余分検出」に分類する。

実装：[audit_arob_element_impact_20261026.py:41](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:41>)（元素辞書）、[audit_arob_element_impact_20261026.py:49](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:49>)（トークン）、[audit_arob_element_impact_20261026.py:95](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:95>)（全文消費・判定）、[audit_arob_element_impact_20261026.py:142](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:142>)（元素集合から同じ29特徴へ変換）。

### 「要確認」の扱い

以下は主集計の化学式比較から外した。

- 括弧・角括弧・波括弧を含む式。単なるサイト表記もあるが、成分比や複合体の表記もあるため、今回は一律に保留する。
- /、+、@、,、:、%、中点等を含む複合・注記表記。
- 内部空白を含む式や材料の説明文。空白を消して別の元素記号につなげる補正はしない。
- x、δ等の変数、電荷・ハイフン・未対応文字、未知元素記号、文字列を最後まで解析できないもの。
- 空欄、unknown、undefined等の未指定表記。

例えば`(Bi0.25Sb0.75)2Te3`のような解析できる可能性の高い括弧式も保留している。**これは化学的に解析不能だと断定したものではない。** 誤って複合表記を単一式扱いしないための監査範囲の制限であり、今回の確定件数は全体の下限である。要確認をすべて確認して初めて最終的な総数が決まる。

要確認一覧：[review_cases.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/review_cases.csv:1>)。試料ごとの原表記と理由を残している。対象集合ごとの内訳：[review_reason_counts.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/review_reason_counts.csv:1>)。

### 旧抽出値・保存特徴値との比較

旧Step18は元素記号に一致しても、直前が英字、または直後が小文字ならその記号を捨てる。例えば`CoSb3`のSbは直前がoのため捨てられる。[build_tau_eff_ml_dataset_step18.py:430](<C:/Users/miots/m-thesis/m-thesis/experiments/exp005/build_tau_eff_ml_dataset_step18.py:430>)（430–443行）。

まず現行の旧抽出規則を保存18,467試料に適用し、元素数1列・元素存在21列・元素群7列の計29列を再構成した。**保存feature matrixとの相違は0セル**だった。そのうえで、明確に解析できた式の元素集合から、元コードと同じ群定義で29列を計算し、保存特徴値と比較した。群の構成や閾値は変えていない。記号辞書も旧Step18の118元素と一致することを検査した。

この確認により、単に「コードなら取りこぼすはず」と推定したのではなく、**掲載系で使われた保存特徴値に相違がある**ことを確認している。入力表：[tau_eff_ml_metadata_step18.csv](<C:/Users/miots/m-thesis/m-thesis/data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_metadata_step18.csv:1>)、[tau_eff_ml_feature_matrix_step18.csv](<C:/Users/miots/m-thesis/m-thesis/data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_feature_matrix_step18.csv:1>)。集計実装：[audit_arob_element_impact_20261026.py:156](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:156>)、[audit_arob_element_impact_20261026.py:227](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:227>)。

## 2. 集合ごとの試料数

数える単位は重複しないsample_key。温度点の数ではない。各行は重なりのある集合なので、縦方向に合計しない。

| 集合 | 全試料 | 式を確認 | 要確認 | 取りこぼし | 0係数の余分検出 | 相違あり計 | 影響列数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧ML候補 | 15,623 | 12,930 | 2,693 | 4,421 | 36 | 4,456 | 28 |
| 旧ML DOI学習 | 12,422 | 10,234 | 2,188 | 3,512 | 30 | 3,541 | 28 |
| 旧ML DOI評価 | 3,201 | 2,696 | 505 | 909 | 6 | 915 | 25 |
| 図2・6共通 | 109 | 90 | 19 | 31 | 0 | 31 | 16 |
| 図4掲載 | 2,943 | 2,489 | 454 | 859 | 6 | 865 | 25 |
| 図7 ML学習（全材料） | 3,577 | 3,044 | 533 | 1,084 | 3 | 1,087 | 27 |
| 図7 ML掲載 | 280 | 259 | 21 | 127 | 0 | 127 | 18 |

「相違あり計」は両種類の相違の和集合であり、実際に保存入力の少なくとも1列が異なる試料数とも一致した。旧候補およびDOI学習側で各1試料が両方に該当するので、4,421+36−1=4,456、3,512+30−1=3,541となる。その他の集合で重複は0。

取りこぼしだけを見る場合、図7掲載では127/280試料（45.4％）、解析対象259試料に限ると127/259（49.0％）。残る132試料は今回の29列に相違なし、21試料は要確認である。**この比率は入力相違の比率であり、誤予測の比率ではない。**

集計表：[cohort_summary.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/cohort_summary.csv:1>)。全試料の集合所属・抽出元素・不足元素・余分元素・差分列・保存値/式由来値は[sample_assessment.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/sample_assessment.csv:1>)。要確認行の相違フラグは未判定のためfalseとなっているので、必ずparse_statusと併用すること。報告の「相違なし」はclear行に限定している。

### 各集合の同定方法

| 集合 | 使用した保存記録と選択 |
|---|---|
| 旧ML候補 | recommended CSVのsample_key集合15,623件 |
| 旧ML DOI学習・評価 | 上記集合と、保存split_doi_group_80_20_step18のtrain/testを結合。12,422/3,201件 |
| 図2・6共通 | three_method_sample_summary.csvの109 sample_key。旧DOI testの部分集合であることを確認 |
| 図4 | plot_source_ML.csvの2,943 sample_key。旧DOI testの部分集合であることを確認 |
| 図7 ML学習 | fine版ml_sample_audit_fine.csvのused_to_train_ML=True、3,577試料。酸化物だけに限定せず実際の全材料の学習集合 |
| 図7掲載 | fine版plot_sourceでmethod=ML、broad_group=broad::oxide、観測σ・予測σが有限かつ正。1,547点・280 sample_key |

図7の「train split全体」は12,822試料だが、その全員をExtra Treesに入力しているわけではない。旧特徴表と対応でき、5点以上から教師を作れた試料等を残したused_to_train_MLの3,577試料を今回の学習対象とした。元コードの選択式は[build_oxide_comparison.py:228](<C:/Users/miots/m-thesis/m-thesis/experiments/exp011_oxide_v1_df_ml_fml/build_oxide_comparison.py:228>)（228–247行）。

図7学習には旧推奨候補15,623件に含まれない408試料、図7掲載には同候補外の72試料がある。したがって図7を旧recommendedだけで監査すると対象を欠落させる。今回は元特徴表の18,467試料を基礎にし、全対象のsample_keyが揃うことを検査した。

保存根拠：[tau_eff_ml_dataset_recommended_step18.csv](<C:/Users/miots/m-thesis/m-thesis/data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_dataset_recommended_step18.csv:1>)、[tau_eff_ml_splits_step18.csv](<C:/Users/miots/m-thesis/m-thesis/data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_splits_step18.csv:1>)、[three_method_sample_summary.csv](<C:/Users/miots/m-thesis/m-thesis/data/output/exp005_ml_df_exp006_common_cohort_zt_comparison/three_method_sample_summary.csv:1>)、[plot_source_ML.csv](<C:/Users/miots/m-thesis/m-thesis/data/output/df_ml_fml_material_parity_20261025/plot_source_ML.csv:1>)、[ml_sample_audit_fine.csv](<C:/Users/miots/m-thesis/m-thesis/experiments/exp011_oxide_v1_df_ml_fml/data/ml_sample_audit_fine.csv:1>)、[plot_source_fine.csv](<C:/Users/miots/m-thesis/m-thesis/experiments/exp011_oxide_v1_df_ml_fml/data/plot_source_fine.csv:1>)。監査での集合選択は[audit_arob_element_impact_20261026.py:185](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:185>)。

### 要確認の内訳

| 理由 | 旧候補 | DOI学習 | DOI評価 | 図2・6 | 図4 | 図7学習 | 図7掲載 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 括弧式・成分表記を未裁定 | 2,541 | 2,064 | 477 | 18 | 432 | 495 | 21 |
| 内部空白・説明文 | 58 | 49 | 9 | 1 | 6 | 13 | 0 |
| 複合区切り・注記 | 56 | 41 | 15 | 0 | 13 | 10 | 0 |
| 未解釈部分・未知記号 | 20 | 16 | 4 | 0 | 3 | 10 | 0 |
| 変数・電荷・未対応文字 | 14 | 14 | 0 | 0 | 0 | 5 | 0 |
| 欠損・未指定 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |

理由はスクリプトの先に一致した分類を1件だけ付けている。複合材がすべて「複合区切り」に入るという分類ではなく、括弧を含む複合材は先に括弧式へ分類される。

## 3. 相違のある特徴量列

下表の値は**その列で相違を確認した試料数**。0係数の余分検出も含む。1試料で複数列が変わるので列間で加算して試料数にしない。28/28/25/16/25/27/18という集合ごとの影響列数は、取りこぼしだけに限定しても同じであり、影響する列の集合も同一だった。

| 特徴量 | 旧候補 | DOI学習 | DOI評価 | 図2・6 | 図4 | 図7学習 | 図7掲載 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `element_count_step18` | 4,455 | 3,540 | 915 | 31 | 865 | 1,087 | 127 |
| `contains_C_step18` | 39 | 22 | 17 | 0 | 17 | 14 | 1 |
| `contains_Bi_step18` | 128 | 107 | 21 | 0 | 19 | 40 | 2 |
| `contains_Te_step18` | 586 | 453 | 133 | 12 | 130 | 127 | 0 |
| `contains_Se_step18` | 493 | 411 | 82 | 3 | 69 | 107 | 17 |
| `contains_Sb_step18` | 873 | 692 | 181 | 5 | 165 | 186 | 3 |
| `contains_Pb_step18` | 69 | 60 | 9 | 3 | 9 | 11 | 0 |
| `contains_Sn_step18` | 624 | 492 | 132 | 2 | 119 | 140 | 0 |
| `contains_Ge_step18` | 38 | 28 | 10 | 0 | 10 | 8 | 0 |
| `contains_Si_step18` | 197 | 163 | 34 | 2 | 32 | 40 | 0 |
| `contains_Mg_step18` | 12 | 12 | 0 | 0 | 0 | 4 | 0 |
| `contains_Co_step18` | 263 | 187 | 76 | 2 | 73 | 58 | 19 |
| `contains_Fe_step18` | 200 | 152 | 48 | 8 | 46 | 58 | 4 |
| `contains_Cu_step18` | 143 | 114 | 29 | 0 | 28 | 56 | 6 |
| `contains_Ag_step18` | 59 | 43 | 16 | 0 | 16 | 17 | 1 |
| `contains_In_step18` | 68 | 56 | 12 | 0 | 12 | 13 | 0 |
| `contains_Ga_step18` | 58 | 47 | 11 | 0 | 11 | 26 | 0 |
| `contains_Ca_step18` | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| `contains_Na_step18` | 10 | 8 | 2 | 0 | 2 | 1 | 0 |
| `contains_K_step18` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `contains_O_step18` | 666 | 572 | 94 | 2 | 93 | 135 | 97 |
| `contains_N_step18` | 4 | 4 | 0 | 0 | 0 | 5 | 0 |
| `has_chalcogen_step18` | 1,577 | 1,300 | 277 | 12 | 260 | 346 | 95 |
| `has_pnictogen_step18` | 982 | 787 | 195 | 5 | 177 | 238 | 10 |
| `has_transition_metal_step18` | 732 | 545 | 187 | 8 | 182 | 202 | 23 |
| `has_alkali_or_alkaline_step18` | 37 | 31 | 6 | 0 | 6 | 12 | 4 |
| `has_heavy_element_step18` | 1,043 | 829 | 214 | 9 | 194 | 228 | 5 |
| `has_toxic_attention_element_step18` | 1,302 | 1,048 | 254 | 9 | 234 | 310 | 17 |
| `has_rare_metal_attention_element_step18` | 847 | 656 | 191 | 7 | 164 | 212 | 3 |

`contains_K_step18`は今回の明確な式では相違0件だった。他の対象外列や要確認試料まで正しいと認定したものではない。

機械可読の列別集計は[feature_difference_counts.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/feature_difference_counts.csv:1>)。different_samplesは上表の全相違、omission_different_samplesは取りこぼしだけを戻した場合の相違件数である。後者は、0係数の検出をそのままにして不足元素だけを集合へ追加し、二つの原因を分けて数えている。どちらもモデルへ投入した結果ではない。

直接の相違対象はStep18の元素由来29列である。Step9のrare_metal_flag_auto_step9/toxicity_flag_auto_step9、ナノカーボン、n/p、測定source、添加物、構造の列をこの比較で変更したものではない。Step9側に別の誤判定がないことを保証する監査でもない。

### 掲載集合の具体例

| 所属 | sample_key | composition | 旧Step18の元素 | 式から読む元素 | 取りこぼし |
|---|---|---|---|---|---|
| 図7 | `10.1002_adem.201400183__sample_19732` | `TiO1.75` | Ti | O, Ti | O |
| 図7 | `10.1002_adma.201301675__sample_20786` | `Pb0.08BiCuSeO` | Bi, Pb | Bi, Cu, O, Pb, Se | Cu, O, Se |
| 図7 | `10.1002_pssa.200723480__sample_26666` | `YBa2Cu2.99Ag0.01O7` | Ag, Cu, O, Y | Ag, Ba, Cu, O, Y | Ba |
| 図2・6 | `10.1002_pssa.202000096__sample_83951` | `TiHfFeNiSb1.7Sn0.3` | Sn, Ti | Fe, Hf, Ni, Sb, Sn, Ti | Fe, Hf, Ni, Sb |
| 図2・6 | `10.1007_s13391-018-0088-0__sample_40954` | `FeV0.9Ti0.1Sb` | Fe, Sb, Ti | Fe, Sb, Ti, V | V |
| 図2・6 / 図7 | `10.1016_j.matlet.2021.130180__sample_45720` | `Na0.96CoO2` | Co, Na | Co, Na, O | O |

例えば図7の`TiO1.75`は、保存element_count=1、contains_O=0、has_chalcogen=0だが、式からはそれぞれ2、1、1となる。一方、図2・6の`FeV0.9Ti0.1Sb`ではVが落ちるものの、Vは21の個別元素列にも既存のtransition_metal集合にも含まれないため、相違するのは元素数だけである。元素集合の差から、すべての群フラグが必ず変わると推定せず、各列を実計算した。

## 4. 図7の最終入力列

**図7のfine版では最終74列が入力に残り、問題の対象となる元素由来29列はすべて残っている。** 列を生成した後の定数除去によって今回の問題が解消されていた、という状況ではない。

処理は以下のとおり。

1. 保存Step18 feature matrixの55列を読み、material_system__から始まる列を外して54列を候補にする。
2. 図7の実学習試料に出現する材料分類28種類をone-hotで追加し、計82候補列とする。
3. 元のprepare_featuresで学習側中央値補完・全欠損/分散0列の除去を行い、46の旧特徴列+28材料分類列=74列を残す。
4. 元素数1+個別元素21+元素群7の29列はすべてused。今回取りこぼしによる相違がある列は、学習側27列、掲載側18列である。

根拠コード：[build_oxide_comparison.py:241](<C:/Users/miots/m-thesis/m-thesis/experiments/exp011_oxide_v1_df_ml_fml/build_oxide_comparison.py:241>)（241–269行）、[train_tau_eff_ml_step19.py:127](<C:/Users/miots/m-thesis/m-thesis/experiments/exp005/train_tau_eff_ml_step19.py:127>)（127–173行）。保存記録：[feature_audit_fine.csv](<C:/Users/miots/m-thesis/m-thesis/experiments/exp011_oxide_v1_df_ml_fml/data/feature_audit_fine.csv:1>)、[feature_importance_fine.csv](<C:/Users/miots/m-thesis/m-thesis/experiments/exp011_oxide_v1_df_ml_fml/data/feature_importance_fine.csv:1>)。

今回、保存データから**特徴量の前処理だけ**を再計算した。列順を含む74列がfeature_auditのused列と完全一致し、feature_importanceの列順とも一致した。補完中央値も全使用列で一致した。既存モデルを読み込んでpredict/fitした検証ではない。監査コード：[audit_arob_element_impact_20261026.py:203](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:203>)（203–224行）。

最終列の完全な一覧・採否・除外理由・学習/掲載側の差分数は[fig7_final_input_columns.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/fig7_final_input_columns.csv:1>)に保存した。used_in_model_step19=Trueの74行が最終列であり、旧関数由来の列名にstep19と付いていても、このファイルの対象は図7のfineモデルである。

除外された8候補列はいずれも分散0で、以下のとおり。元素由来列は含まれない。

- `nanocarbon_type_auto_step9__fullerene`
- `nanocarbon_type_auto_step9__graphene`
- `additive_final_step17__composite_nanocomposite`
- `structure_final_step17__1d`
- `structure_final_step17__nanocomposite`
- `structure_final_step17__nanostructure`
- `structure_final_step17__nanotube_thin_film_film`
- `sintering_method_final_step17__unknown`

図7の新しい材料分類one-hotは別の分類経路で作られる。例えば酸化物の材料分類が正しく付いていても、旧特徴表から来るcontains_Oが0のまま併存し得る。この材料分類列の存在だけで、元素特徴の相違がモデルに影響しないとは判断できない。

## 5. 確認できた範囲と判断できない範囲

| 確認した事項 | 判断できない事項 |
|---|---|
| 保存された元素特徴が、明確な化学式から読める集合と異なる試料・列・値 | 修正した場合に個々の予測がどれだけ変わるか |
| 相違が旧MLの学習側にも評価側にも存在する | 保存モデルへ別特徴を入れたときに同じ葉へ到達するか、予測が変化するか（推論も未実施） |
| 図2・6、図4、図7の掲載試料に相違が存在する | 修正して再学習した場合のMAE・RMSE・R²や手法順位 |
| 図7の最終入力に対象列が残り、保存記録と前処理結果が一致する | 相違がある試料で必ず誤差が増えていた、修正すれば必ず改善するという因果 |
| 入力相違なしの試料と要確認を区別した | 入力相違なしの試料なら再学習後も予測が同じだという保証 |

学習側にも相違があるため、仮に今後修正して再学習すれば、**自身の入力に相違がなかった試料も含めてモデル全体の予測が変わり得る**。したがって「掲載試料のうち取りこぼしがある31/859/127試料だけが性能面の影響対象」とは限定できない。今回の件数は、直接の特徴量相違を確認した範囲である。

旧DF/FMLの数値を今回再計算したり、今回の相違数からその性能を補正したりはしていない。ここで示した相違はML入力特徴に関するもので、図2・6の109試料を3手法分に数えてはいない。

**既存数値の保存**：元コード、入力CSV、保存予測・指標、図2/4/6/7のPNG、図7モデルを含む24ファイルのSHA-256を読み取り前後で比較し、全件同一だった。保存予測や評価値を変更していない。記録は[audit_results.json](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/audit_results.json:1>)のinput_manifestとinput_hashes_before_and_after_equalにある。これは今回の非変更を確認するもので、過去の実行コードの完全な履歴を証明するものではない。

## 6. 新規成果物と再集計方法

| 新規ファイル | 内容 |
|---|---|
| [cohort_summary.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/cohort_summary.csv:1>) | 7集合の試料数・要確認数・相違数 |
| [feature_difference_counts.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/feature_difference_counts.csv:1>) | 29列×7集合。全相違と取りこぼし由来を分離 |
| [sample_assessment.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/sample_assessment.csv:1>) | 18,467試料の原組成、解析状態、旧/式由来元素、差分列、保存値/比較値、集合所属 |
| [review_cases.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/review_cases.csv:1>) | 要確認の原組成・理由。対象集合所属列で絞り込める |
| [review_reason_counts.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/review_reason_counts.csv:1>) | 集合別の要確認理由件数 |
| [fig7_final_input_columns.csv](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/fig7_final_input_columns.csv:1>) | 図7の82候補列と最終74列の採否 |
| [audit_results.json](<C:/Users/miots/m-thesis/m-thesis/reports/AROB_JPv4_元素抽出影響調査_20261026/audit_results.json:1>) | 検査結果と既存入力24ファイルのSHA-256 |
| [audit_arob_element_impact_20261026.py](<C:/Users/miots/m-thesis/m-thesis/reports/audit_arob_element_impact_20261026.py:1>) | 読み取り・集計専用スクリプト。元パイプラインを起動しない |

スクリプトを出力先引数なしで実行すると集計JSONを標準出力に出すだけである。--output-dirを指定する場合は存在しない新規ディレクトリだけを許し、既存ディレクトリへの上書きを拒否する。本監査用スクリプト内の化学式規則は監査専用であり、元のStep18を置き換えるものではない。

