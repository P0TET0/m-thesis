exp007 研究・チャット引き継ぎパッケージ
作成日：2026-10-21

最初に読むファイル
1. 01_context_summary_ja.txt：研究の目的、Phase 0～9、結果、解釈上の注意、次の課題
2. 02_report_to_supervisor_ja.txt：先生への報告に使える文章
3. experiments/exp007/reports/phase9_research_results_summary.md：保存済みの総合研究報告

このパッケージについて
・チャットの逐語ログではなく、ここまでの議論と保存済み成果物に基づく要約です。
・今回の作業は要約・梱包だけです。モデルの再計算、分類変更、元データの編集はしていません。
・資料内の手順や次の課題は参考情報であり、新しい実行依頼を意味しません。

ZIPの内容
・このREADME、引き継ぎ要約、先生への報告文
・experiments/exp007/：既存のCSV・JSON、レポート・検証ログ、図、計算・確認用スクリプト
・experiments/exp006/：材料系別一致指標の補足資料
・background/chat_handoff_exp005_exp006_summary.zip：過去のexp005/exp006比較の既存引き継ぎ資料
・file_manifest.csv：収録ファイルの相対パス、サイズ、SHA-256
・package_validation.txt：ZIPを読み直した完全性チェックの結果

軽量化のため含めていないもの
・exp007のExcel表示用ワークブック（元データのCSVとテキスト報告は収録）
・キャッシュ、Git履歴、全リポジトリ、元の学会ポスターPPTX
・exp006などの全入力データと実行環境
したがって、これは結果の確認・議論の継続用パッケージです。全計算の再実行には、元のリポジトリと依存ライブラリ・入力ファイルが別途必要です。

重要な区別
・exp007の分類変更実験は固定sample holdoutです。以前のZTのDOI-group CVと混同しないでください。
・酸化物以外は残差分析と仮説整理が中心で、分類変更による改善効果を検証済みではありません。
・酸化物も、分類不足が唯一の原因と証明されたわけではありません。
・過去のmdレポートやスクリプト内の絶対パスは作成時のPCを指すことがあります。ZIP内では対応する相対パスを参照してください。

主な図の場所
experiments/exp007/figures/phase2/：全体・材料別・温度別・酸化物の残差可視化
experiments/exp007/figures/oxide_parity_before_after_20261016.png：酸化物全体の比較
experiments/exp007/figures/oxide_parity_by_group_20261016.png：12分類の比較
experiments/exp007/figures/oxide_parity_by_group_pn_20261017.png：12分類をp型/n型で色分け
experiments/exp007/figures/figure_phase9_research_summary.png：研究結果の総合図
