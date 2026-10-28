# exp011_composition_review：組成比を考慮したDF・ML・FML法の材料分類の見直し

Starrydata2の組成式から母材と少量添加元素を区別し、DF・ML・FML法で共通に使う材料分類を検討するためのフォルダです。微量元素の有無によって母材と異なる材料系に分類される問題を調べます。

現在は、母材の合金成分とドーパントを区別する方法を検討しています。Si0.8Ge0.2はSi–Ge合金として扱い、Geまで消してSi主体にまとめる案と区別します。係数0.5以下の機械的除外は影響調査のための初期案で、採用していません。現段階では既存分類の変更、再学習、再フィッティングは行いません。

`Si0.5Ge0.5` はSiGe混合母材として両元素を残す方針に合意しました。閾値0.5の採用、他の材料への同率規則の拡張、同率とみなす許容幅は未確定です。

- `atomic_fraction_review_ja.md`：原子分率で少量元素を除く案の試算。全試料の計算可能件数、閾値別の影響、文献上の元素役割との不一致、残る懸念。
- `audit_atomic_fraction.py` / `atomic_fraction_audit.json`：上記試算の再現スクリプト、集計値、試料ID、根拠箇所。
- `host_alloy_dopant_review_ja.md`：母材・合金成分・ドーパントの違い、SiGeの実例、文献に基づく判定方法。
- `doi_host_review/`：全DOIの公開情報照合、試料別の母材・ドーパント判定CSV、取得件数と根拠箇所の記録。確認済み・自動候補・未判定を区別します。
- `composition_review_ja.md`：初期調査、実データの例、検討した分類案。
- `audit_composition_notation.py`：組成表記と閾値の影響を調べる診断用スクリプト。新しい材料分類は割り当てません。
- `composition_audit.json`：試料数、表記の集計、実例、入力ファイルのSHA-256。
- `threshold_review_ja.md`：係数をそろえた後も残る問題、0.5という境界、除外後に残った比率の合計を1にする案の検討。
- `audit_threshold_tradeoffs.py` / `threshold_tradeoffs.json`：限定した組成群での閾値別の診断と試料ID。分類変更は含みません。
- `material_threshold_counts_ja.md`：17種類の組成群について、係数0.05〜0.5で除外対象が生じる試料数と抽出条件。
- `audit_material_thresholds.py` / `material_threshold_counts.json`：材料別の試算スクリプトと試料ごとの根拠。DF・ML・FMLの既存図に含まれる試料の件数も記録します。

再集計はリポジトリのルートで `python experiments/exp011_composition_review/audit_composition_notation.py` を実行します。
