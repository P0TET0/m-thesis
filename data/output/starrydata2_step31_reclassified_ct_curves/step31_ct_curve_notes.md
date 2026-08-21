# Step31 C(T) の定義

Step31 では、組成から新たに再分類した材料系ごとに、25 K 幅の温度 bin 内にある Starrydata2 由来の実測電気伝導率 $\sigma_{\mathrm{exp}}(T)$ の中央値を求め、その温度依存曲線を経験的基準曲線 $C(T)$ と定義した。primary の $C(T)$ は n/p 型で分割せず、各 bin に3点以上の実測値がある区間だけを図示した。

この $C(T)$ は Step31 で新たに導入した論文・説明用の基準曲線であり、Step12 の tau_eff fitting で用いた経験的 prefactor $C(T)$ とは定義が異なる。Step12 の出力は変更していない。今後の tau_eff は、必要に応じてこの Step31 の $C(T)$ に対する相対倍率として解釈する想定である。

組成式から元素比を推定し、既定では 5 at% 以上を主成分、5 at% 未満をドーパント候補とした。母相は `host_family_key`、主成分元素系は `major_element_system_key` として保存し、後者が行数・試料数・論文数の基準を満たす場合だけ `final_material_group_key` に採用した。基準未満では母相へ戻すため、微量置換による過分割を抑えている。Ge–Te は `host::GeTe` として独立分類した。元素抽出可能だが既知母相に該当しないものは `host::other_formula_system`、抽出不能なものは `unknown_material_group` とした。Step31 の C(T) は安定した母相比較のため `host_family_key` ごとに集計し、細分類キーは解析用CSVに保持した。

すべての図の横軸は少なくとも 0–1300 K を含み、1300 K を超える実測 bin がある場合はその最大温度を切り捨てない範囲まで拡張した。
