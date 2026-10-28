# DOIを根拠とする母材判定

Starrydataの全対象DOIを公開書誌サービスで照合し、母材・ドーパントの判定と根拠を試料別に保存するフォルダです。係数0.5以下の一律除去は行いません。集計結果は `coverage_report_ja.md` を参照してください。情報の取得件数と、母材を判定できた件数は別に集計します。

## 出力CSV

- `outputs/sample_host_review.csv`：全104,832試料の原組成、確認済み母材、ドーパント、候補、判定状態、根拠ID、注意点。
- `outputs/paper_coverage.csv`：全17,337 DOIの書誌・要旨・公開本文の取得状況と、試料の判定件数。
- `outputs/host_evidence.csv`：根拠URL、頁・節・図または取得項目、短い抜粋、日本語説明。試料表の `evidence_ids` と結合します。

CSVはUTF-8 BOM付き。試料キーは **doi + sample_id** です。`sample_id` 単独では別論文の試料を混同します。曲線233,297行を試料に集約し、`curve_count` に対応曲線数を残しています。統一前のDOI表記は `doi_original` に保存しています。

## 判定状態

| judgment_status | 意味 |
|---|---|
| literature_host_and_roles | 一次資料と試料を照合し、母材と成分の役割を確認 |
| literature_host_only | 母材は確認。ドーパント・添加物・相構成の一部は保留 |
| automatic_candidate | タイトル・要旨・取得本文から自動抽出した候補。個別試料では未確定 |
| unresolved_after_review | 個別確認を試みたが根拠不足・不整合等で未確定 |
| unresolved | 自動照合では特定できず、個別確認が必要 |

`host_material` は個別確認した母材、`host_candidate` は自動候補です。`review_level` で本文の該当箇所・抄録・検索索引の公開抜粋等を区別します。母材の「確認」は、論文全文の精読や相分率・活性キャリア濃度の確定を意味しません。比較試料も含まれるため、論文タイトルの母材を全試料へ一律適用しません。

## 比率と根拠

役割を確認でき、組成式を厳密に解析でき、第二相等が未解決でない場合に限り、ドーパントを除いて係数の総和を1にした値を `retained_atomic_fractions` に保存します。`excluded_elements` が除外した元素です。原組成は変更せず、空いたサイトの補完や非化学量論の修復も行いません。この比率は分類用で、理想化学量論や相分率ではありません。

個別確認の根拠は `source_location` に頁・節・図等を記載します。自動抽出は `/title/0`、`/abstract`、`/abstractText` 等、または本文XMLのbody内の段落番号と、HTMLタグ除去・空白統一後の文字位置（0始まり、終端を含まない）です。`local_cache` は本フォルダからの相対パスで、取得応答を再確認できます。

`source_sample_record` は元samples CSVのヘッダーを1としたレコード番号です。引用内改行があるため物理行番号とは異なります。`source_material_family` は既存分類の参考情報であり、新しい母材の確定根拠には使いません。索引で検索結果が0件でも、論文が存在しないとは判断しません。

## 入力・再現

入力は `C:\Users\miots\OneDrive - 山陽小野田市立山口東京理科大学 (1)\USBにいたヤツ\大学\1_研究\starrydata2` 内の `starrydata_curves.csv`、`starrydata_samples.csv`、`starrydata_papers.csv`。パス・サイズ・SHA-256は `cache/manifest.json` に保存します。

`retrieve_literature.py` の `inventory`、`crossref`、`europepmc`、`registries`、`fulltext` で読取・取得します。API応答をバッチごとに保存し、再開時は成功済み応答を再利用します。Crossrefの公開APIは応答ヘッダーの同時実行数1・毎秒1リクエストに従います。未収録DOIはDataCiteと解決先で補完します。本文XMLはEurope PMCで公開アクセス可能と記載されたものが対象です。

`agents/*evidence*.json` が個別照合、`agents/auto_host_candidates.py` が候補抽出、`build_review.py` が統合、`export_review.mjs` がCSV出力です。入力更新時は個別判定の試料ID・組成条件も再照合してください。既存DF・ML・FMLへの適用・再学習は行っていません。
