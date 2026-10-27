Batch 4 — GB10 最小功能驗收交付包
受測版本：7fc417c00602023a4bdce24afe5292f5c8ba5625
驗收日期：2026-10-27

結論
Codex 已複核本次約定範圍的最小功能驗收，判定通過，無須追加測試。
正常 launch 載入既有模型及 SP；API 新結果契約、WebUI、CSV 與 Markdown 內容一致；服務正常停止，checkout 已還原，11 份指定輸入摘要相符。
完整結論、證據層級與既有提示見 review/CODEX_ACCEPTANCE_REVIEW.txt。

目錄
- evidence/：原始驗收目錄的完整副本，含 API request/response/status、服務 log、退出碼、截圖、CSV、配置、受測版本、checkout 前後及輸入摘要核對。
- review/：Codex 複核報告、原始證據摘要清單、Markdown 補存來源說明及補存的 Markdown。
- SHA256SUMS：本交付包所有其他檔案的相對路徑 SHA-256；共 25 份檔案。

Markdown 來源 — 請在收尾登記中保留這項區分
原驗收 evidence/exports/ 中只有 CSV，沒有 Markdown。
補存檔：review/diagnosis_batch4_smoke_01_sess_6469db6034c4_report.md
它是從同一次 WebUI session（sess_6469db6034c4）的 Gradio 匯出快取逐位元複製的原檔，沒有重跑診斷、沒有重新產生報告、沒有修改內容。
來源位置及 SHA-256 見 review/RECOVERED_REPORT_SOURCE.txt。
服務 log 證明 CSV 下載 HTTP 200；未記錄 Markdown 下載 GET。Markdown 的下載／開啟成功依使用者回報，Codex 獨立核對的是快取中的實際報告內容，不將其標為已獨立證明的瀏覽器下載。

兩次請求
- API：sess_7c26f2a10f43，原始回應 evidence/diagnose.json。
- WebUI：sess_6469db6034c4，對應截圖、CSV 與補存 Markdown。
這是相同輸入和服務身分的兩次獨立請求，不是同一 session。10 筆候選的順序及格式化分數已逐筆核對。

收尾登記建議
沿用現有 Batch 4 文件，記錄實作／審查通過及本次最小 GB10 功能驗收通過，受測版本鎖定 7fc417c。
保留上述 session 與快取來源說明；不要將本輪擴張為模型品質、醫學效能、容量、全套測試或新論文架構的驗收。
BACKLOG 60 的兩個既有 Training 測試問題獨立保留，未在本次修正或複測。
後續 base/finetune、家族樹、加速插槽與 UI 重構另行安排。

未包含的大型輸入
checkpoint、真實 workspace、SP 張量原檔留在 homelab，沒有重複裝入本包；其位置與摘要保留於 evidence/inputs-before.sha256。
該檔和 review/ORIGINAL_EVIDENCE_SHA256SUMS.txt 保留原本的絕對路徑，供追溯，不是另一台機器上驗證此交付包的入口。

核對方法
在解壓後的本包根目錄執行：
    sha256sum -c SHA256SUMS
應為 25/25 OK。壓縮檔本身的摘要在旁邊的 .tar.gz.sha256 檔。

原檔保留
原始驗收目錄：/home/user/batch4-acceptance-7fc417c-01/
原始複核目錄：/home/user/Documents/Codex/2026-09-20/github-plugin-github-openai-curated-remote/outputs/batch4-acceptance-review-7fc417c-20261027/
本次只複製及打包，未修改、移動或刪除以上任何原檔，未切換 repo 或啟動服務。
