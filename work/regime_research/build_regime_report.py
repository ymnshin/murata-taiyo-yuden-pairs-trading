from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
CHARTS = WORK / "charts"
OUT = ROOT / "outputs" / "murata_taiyo_yuden_regime_pair_strategy_report.pdf"

with (WORK / "analysis_summary.json").open(encoding="utf-8") as f:
    S = json.load(f)
with (WORK / "sources.json").open(encoding="utf-8") as f:
    SOURCES = json.load(f)

DIAG = pd.read_csv(WORK / "regime_diagnostics_weekly.csv", parse_dates=["Date"]).set_index("Date")
STRAT = pd.read_csv(WORK / "strategy_cost_borrow_matrix.csv")
RANK = pd.read_csv(WORK / "murata_taiyo_monthly_rank.csv", parse_dates=["screen_date"])
TIMELINE = pd.read_csv(WORK / "event_timeline.csv")
TRADES = pd.read_csv(WORK / "2026_low_always_trades.csv")
UNIVERSE = pd.read_csv(WORK / "universe.csv")
LOW_DAILY = pd.read_csv(WORK / "2026_low_always_daily.csv", parse_dates=["Date"]).set_index("Date")

FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf")
if not FONT_PATH.exists():
    FONT_PATH = Path(r"C:\Windows\Fonts\YuGothM.ttc")
pdfmetrics.registerFont(TTFont("NotoJP", str(FONT_PATH)))

NAVY = colors.HexColor("#173B57")
BLUE = colors.HexColor("#1479B8")
TEAL = colors.HexColor("#138A8A")
GREEN = colors.HexColor("#2E8B57")
ORANGE = colors.HexColor("#E38B2C")
RED = colors.HexColor("#C44E52")
PURPLE = colors.HexColor("#6657A3")
MID = colors.HexColor("#5F6B76")
DARK = colors.HexColor("#253644")
LIGHT = colors.HexColor("#DFE6EC")
BG = colors.HexColor("#F6F8FA")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREEN = colors.HexColor("#EAF5EF")
PALE_ORANGE = colors.HexColor("#FFF3E5")
PALE_RED = colors.HexColor("#FBECEC")

styles = getSampleStyleSheet()
BASE = ParagraphStyle(
    "BaseJP", parent=styles["BodyText"], fontName="NotoJP", fontSize=8.8,
    leading=13.1, textColor=DARK, wordWrap="CJK", spaceAfter=3,
)
TITLE = ParagraphStyle(
    "TitleJP", parent=BASE, fontSize=23.5, leading=31, textColor=NAVY,
    alignment=TA_LEFT, spaceAfter=7,
)
SUBTITLE = ParagraphStyle(
    "SubtitleJP", parent=BASE, fontSize=11.8, leading=18, textColor=MID,
)
H1 = ParagraphStyle(
    "H1JP", parent=BASE, fontSize=16.8, leading=23, textColor=NAVY,
    spaceBefore=1, spaceAfter=7,
)
H2 = ParagraphStyle(
    "H2JP", parent=BASE, fontSize=11.2, leading=16, textColor=BLUE,
    spaceBefore=4, spaceAfter=3,
)
SMALL = ParagraphStyle(
    "SmallJP", parent=BASE, fontSize=7.2, leading=10.2, textColor=MID,
)
TINY = ParagraphStyle(
    "TinyJP", parent=BASE, fontSize=6.25, leading=8.5, textColor=MID,
)
WHITE = ParagraphStyle(
    "WhiteJP", parent=BASE, fontSize=9.4, leading=14.2, textColor=colors.white,
)
WHITE_BIG = ParagraphStyle(
    "WhiteBigJP", parent=WHITE, fontSize=18, leading=22, alignment=TA_CENTER,
)
TABLE_HEAD = ParagraphStyle(
    "TableHeadJP", parent=BASE, fontSize=7.2, leading=9.8, textColor=colors.white,
)
TABLE_BODY = ParagraphStyle(
    "TableBodyJP", parent=BASE, fontSize=7.25, leading=10.0,
)
CENTER = ParagraphStyle(
    "CenterJP", parent=BASE, alignment=TA_CENTER,
)
RIGHT = ParagraphStyle(
    "RightJP", parent=BASE, alignment=TA_RIGHT,
)


def p(text: str, style: ParagraphStyle = BASE) -> Paragraph:
    return Paragraph(str(text), style)


def bullet(text: str, color: str = "#253644") -> Paragraph:
    return p(f'<font color="{color}">●</font> {text}', BASE)


def pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def date_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def table(rows, widths, header=True, font_size=7.2, row_colors=(colors.white, BG), repeat=True):
    converted = []
    for i, row in enumerate(rows):
        style = TABLE_HEAD if header and i == 0 else TABLE_BODY
        converted.append([c if hasattr(c, "wrap") else p(str(c), style) for c in row])
    t = Table(converted, colWidths=widths, repeatRows=1 if header and repeat else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "NotoJP"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.8),
        ("GRID", (0, 0), (-1, -1), 0.35, LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), list(row_colors)),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    t.setStyle(TableStyle(commands))
    return t


def callout(title: str, body: str, fill=PALE_BLUE, edge=BLUE, width=174 * mm):
    t = Table([[[p(title, H2), p(body, BASE)]]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.8, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def tags_row(items):
    cells = []
    widths = []
    for title, body, color in items:
        cells.append([p(title, WHITE), p(body, TABLE_BODY)])
        widths.append(174 * mm / len(items))
    t = Table([cells], colWidths=widths)
    commands = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.35, LIGHT),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    for i, (_, _, color) in enumerate(items):
        commands.append(("BACKGROUND", (i, 0), (i, 0), colors.white))
        # Header paragraph itself is white, so give its first nested line a dark block via cell background is not possible.
    t.setStyle(TableStyle(commands))
    return t


def metric_cards(items):
    cells = []
    for value, label, fill in items:
        inner = [p(str(value), ParagraphStyle("Metric", parent=WHITE_BIG, textColor=colors.white)), p(label, WHITE)]
        cells.append(inner)
    t = Table([cells], colWidths=[174 * mm / len(items)] * len(items))
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
    ]
    for i, (_, _, fill) in enumerate(items):
        commands.append(("BACKGROUND", (i, 0), (i, 0), fill))
    t.setStyle(TableStyle(commands))
    return t


def chart(name: str, max_w=174 * mm, max_h=105 * mm):
    path = CHARTS / name
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def page_canvas(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(LIGHT)
        canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
        canvas.setFont("NotoJP", 7.1)
        canvas.setFillColor(MID)
        canvas.drawString(18 * mm, height - 11.5 * mm, "村田製作所 x 太陽誘電 | レジーム型ペア戦略")
        canvas.drawRightString(width - 18 * mm, height - 11.5 * mm, "データ: 2026-09-01まで")
    canvas.setStrokeColor(LIGHT)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("NotoJP", 6.9)
    canvas.setFillColor(MID)
    canvas.drawString(18 * mm, 8.5 * mm, "調査レポート。投資助言ではありません。")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, str(doc.page))
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
    topMargin=20 * mm, bottomMargin=18 * mm,
    title="村田製作所と太陽誘電 レジーム型ペアトレード戦略レポート",
    author="OpenAI Codex",
)
story = []

# 1 Cover
story += [Spacer(1, 17 * mm), p("PAIR REGIME RESEARCH / 2026", SMALL),
          p("村田製作所 × 太陽誘電", TITLE),
          p("“最近強い”を、再現可能な戦略へ", TITLE),
          p("社会・企業要因、レジーム開始、当時検知、銘柄探索、PDCAまで", SUBTITLE),
          Spacer(1, 7 * mm), HRFlowable(width="100%", thickness=2, color=BLUE), Spacer(1, 7 * mm)]
story += [metric_cards([
    ("+33.25%", "2026年 低閾値ルール<br/>後知恵候補・コスト後", PURPLE),
    ("2.09", "Sharpe<br/>11取引", BLUE),
    ("2/6–6/19", "後知恵の開始感応度<br/>一点ではない", ORANGE),
]), Spacer(1, 7 * mm)]
story += [callout("結論", "最近の強さは本物だった。しかし、低閾値ルールはOOSを見て選ばれたため、その+33%は配備可能な証拠ではない。実務の回答は、<b>4月末の経済ウォッチリスト入り、3月後半の構造ゲート、8月末の厳格FDR適格</b>を分け、複数ペアのポートフォリオで小さく使うこと。", PALE_RED, RED),
          Spacer(1, 6 * mm),
          bullet("共通因子: AIサーバー/データセンター向けMLCC、円安、業界出荷回復。"),
          bullet("差分因子: 発表タイミング、事業構成、村田の1,500億円自己株取得、太陽誘電の大幅上方修正。"),
          bullet("機構: 高相関 + 残差半減期短縮 + 交差回数増加。単なるボラ拡大ではない。"),
          Spacer(1, 12 * mm), p("作成日 2026-09-02 / 価格ソース Yahoo Finance / 調整後終値", SMALL),
          p("2026-09-02終値は未収録。2026-09-01を最終カットオフとして固定。", SMALL), PageBreak()]

# 2 Executive answers
story += [p("1. エグゼクティブ・アンサー", H1),
          table([
              ["問い", "答え", "証拠 / 留保"],
              ["なぜ最近強い?", "両社が同じAI/MLCC・円安サイクルに入りつつ、非同期の個社材料で相対価格が振れ、その後に収束した。", "業界・会社資料は整合。ただし相関から因果は証明できない。"],
              ["どこでレジーム入り?", "経済的な開始は2026-02-06〜06-19の範囲。後知恵の一点推定は弱い。", "週次収益の1回構造変化。BIC改善は2.18〜2.28に留まる。"],
              ["当時いつ分かった?", "構造ゲートは3/19、実績確認ゲートは5/15。", "週次、過去データのみ、2週連続確認。"],
              ["再現的に見つけられた?", "経済スコア順位は4/30に4位、5/29に2位。厳格FDR適格は8/31で遅い。", "14銘柄91ペア、BH補正。探索器は確認用として保守的。"],
              ["今どうする?", "研究シグナルはショート村田/ロング太陽誘電継続判定。しかし配備は見送り。", "9/1 z=+1.16。entry 1.0の後知恵ルール、entry 2.0の保守ルールはフラット。"],
          ], [32 * mm, 91 * mm, 51 * mm], font_size=7.0), Spacer(1, 5 * mm),
          callout("一番重要な区別", "<b>“2026年に儲かった”</b>と<b>“2026年初に選べた”</b>は別問題。低閾値ルールは前者には合格、後者には未合格。", PALE_ORANGE, ORANGE),
          p("読者への実務メッセージ", H2),
          bullet("高相関ペアを常時稼働させるのではなく、経済ウォッチリスト → 統計ゲート → 個別取引の順に絞る。"),
          bullet("ゲートOFFは新規停止だけ。保有中の通常決済を壊さない。強制決済は独立したハード・キルスイッチに限定する。"),
          bullet("最終的なエッジはペア1組ではなく、同じ手順を複数ペアに適用した横断ポートフォリオで検証する。"), PageBreak()]

# 3 Drivers
story += [p("2. 社会・産業・企業要因: 強い共通因子と大きな差分ショック", H1),
          p("一次資料で確認できる事実", H2),
          table([
              ["層", "事実", "ペアへの読み"],
              ["AI / DC", "村田26Q1のデータセンター関連売上は前年比+81.1%。太陽誘電はAIサーバー中心にコンデンサ受注+41%QoQ、BB1.72。[M4][T4]", "両社株を同方向に動かす最も強い共通因子。"],
              ["業界循環", "JEITA 26年6月: 世界電子部品出荷+25.6%、受動部品+30.5%、中国向け+26.0%。[J1]", "個社だけでなく広い受動部品サイクルが回復。"],
              ["為替", "村田26Q1平均159.49円/USD（前年144.60円）。太陽誘電の通期想定は150円から159.70円へ。[M4][T4]", "両社に追い風だが感応度・予約レートの差が相対価格を動かす。"],
              ["スマホ/車", "村田は26年度スマホ-4%、車-2%予想。太陽誘電は中国自動車減速で自動車構成比-2pt。[M4][T4]", "AIの強さと既存用途の弱さが製品構成差を拡大。"],
              ["在庫/需給", "両社とも前倒し受注・顧客在庫積み増しの可能性を注意喚起。[M3][M4][T3][T4]", "強い受注は実需100%とは限らず、反転しやすい。"],
          ], [27 * mm, 100 * mm, 47 * mm], font_size=6.9), Spacer(1, 5 * mm),
          p("会社固有の差分", H2),
          table([
              ["村田製作所", "太陽誘電", "ペアに生じること"],
              ["コンデンサに加え高周波、電源、機能デバイス。1,500億円自己株取得。[M2]", "売上の約7割がコンデンサ。8/5に営業利益予想を50%上方修正。[T4]", "同じMLCC需要でも株価反応の大きさ・日付がずれる。"],
              ["4/30、7/31に強材料。", "5/8、8/5に追随材料。", "数日〜数週間の相対ショックとキャッチアップ。"],
          ], [58 * mm, 58 * mm, 58 * mm], font_size=7.0), Spacer(1, 5 * mm),
          callout("推論（因果ではない）", "2026年は“同じテーマで同方向”と“非同期の個社材料で一時乖離”が両立した。これが高相関を維持しながら、1σの小さな乖離を何度も収益化できた最も整合的な説明。", PALE_ORANGE, ORANGE), PageBreak()]

# 4 timeline
story += [p("3. イベント・タイムラインと相対価格", H1),
          p("3営業日ロングスプレッド = 0.5×村田収益 - 0.5×太陽誘電収益。イベント公表と同時期の動きであり、因果推定ではありません。", SMALL)]
timeline_rows = [["日付", "出来事 / 一次資料の事実", "相対3日", "帰属"]]
for _, r in TIMELINE.iterrows():
    spread = "-" if pd.isna(r["long_spread_3session_return"]) else f'{float(r["long_spread_3session_return"]) * 100:+.1f}%'
    timeline_rows.append([
        r["date"],
        f'<b>{r["event"]}</b><br/>{r["sourced_fact"]} [{r["source_ids"]}]',
        spread,
        f'{r["inference"]}<br/><font color="#6B7785">信頼度: {r["confidence"]}</font>',
    ])
story += [table(timeline_rows, [22 * mm, 74 * mm, 18 * mm, 60 * mm], font_size=6.4), Spacer(1, 5 * mm),
          callout("読み方", "2/6の太陽誘電発表後は太陽誘電が相対優位、4/30の村田発表後は村田が相対優位。強い共通テーマの中で主役が交代した。ペア利益は“方向を当てた”のではなく、この交代と収束を取った。", PALE_BLUE, BLUE), PageBreak()]

# 5 onset/detection
story += [p("4. レジームはいつ始まり、いつ分かったか", H1),
          chart("01_recent_regime.png", max_h=96 * mm), Spacer(1, 3 * mm),
          table([
              ["概念", "日付 / 範囲", "判定", "確度"],
              ["経済的な開始", "2026-02-06〜06-19", "週次低閾値収益の1回平均変化点。最小区間8/13/21週で感応度。", "低〜中: BIC改善2.18〜2.28"],
              ["構造ゲート検知", "2026-03-19", "相関、正のβ、β安定、半減期、ADF、交差、機会数を2週連続で満たす。", "中: 過去データのみ"],
              ["実績確認ゲート", "2026-05-15", "上記に加え過去126日Sharpe>0、3取引以上、勝率50%以上。", "中: 遅いが頑健"],
              ["厳格FDR適格", "2026-08-31", "91ペアのEG p値をBH補正しq≤0.30。総合1位。", "中: 多重検定対応、非常に遅い"],
              ["最新", "2026-09-01", "構造・実績ゲートON。EG p=0.011、ADF p=0.002、半減期11.3日。", "研究状態のみ"],
          ], [32 * mm, 31 * mm, 79 * mm, 32 * mm], font_size=6.9), Spacer(1, 4 * mm),
          callout("開始日を一点で言わない理由", "収益変化点は最小区間の置き方で2月から6月へ動く。取引数11件で統計力も低い。したがって“2月初旬に芽、3月後半に構造検知、5月中旬に実績確認”という段階表現が最も誠実。", PALE_RED, RED), PageBreak()]

# 6 mechanism
story += [p("5. 何が改善したのか: 定常性・収束・機会密度", H1),
          chart("02_diagnostics_gate.png", max_h=112 * mm), Spacer(1, 3 * mm)]
dates = [pd.Timestamp("2025-12-26"), pd.Timestamp("2026-03-19"), pd.Timestamp("2026-05-15"), pd.Timestamp("2026-09-01")]
diag_rows = [["情報日", "相関126", "ADF p", "EG p", "半減期", "交差126", "機会63", "過去126d Sharpe"]]
for dt in dates:
    r = DIAG.loc[dt]
    diag_rows.append([date_text(dt), num(r.corr126), num(r.adf_p252, 3), num(r.eg_p252, 3), f"{r.half_life252:.1f}日", int(r.zero_cross126), int(r.opportunities63), num(r.trail126_sharpe)])
story += [table(diag_rows, [27 * mm, 18 * mm, 18 * mm, 18 * mm, 21 * mm, 20 * mm, 18 * mm, 34 * mm], font_size=6.8), Spacer(1, 4 * mm),
          p("診断", H2),
          bullet("<b>改善したもの:</b> 相関0.69→0.83、半減期26.6→11.3日、126日交差7→18、EG/ADF p値の低下。"),
          bullet("<b>主因ではないもの:</b> 残差σは0.092→0.078で低下。利益は単なるボラ急増ではなく、収束速度と往復回数の改善。"),
          bullet("<b>閾値適合:</b> 1σは小さな往復を11回捉え、2σの保守ルールは1回だけ。2026年は“薄く頻繁”が合った。"),
          callout("ただし", "252日窓の統計は過去を引きずる。最新のp値が良くても、次の決算・需給変化で壊れる。ゲートは確率を上げるフィルターであり、保証ではない。", PALE_ORANGE, ORANGE), PageBreak()]

# 7 exact strategy
story += [p("6. 再現した戦略と2026年の結果", H1),
          p("低閾値候補の仕様", H2),
          table([
              ["要素", "仕様"],
              ["スプレッド", "log(村田) - α - β×log(太陽誘電)。α, β, 平均, σは直近252営業日、当日を含めず推定。"],
              ["新規", "z≤-1.0: ロング村田/ショート太陽誘電。z≥+1.0: ショート村田/ロング太陽誘電。"],
              ["決済", "平均方向へ±0.5σ到達、20営業日、または逆行4.0σ。"],
              ["実行", "シグナル翌日の終値。ドルニュートラル、各レッグ50%、gross 1.0。"],
              ["コスト", "片道10bp×売買回転 + ショート名目に借株年1%。最終日に清算。"],
          ], [37 * mm, 137 * mm], font_size=7.3), Spacer(1, 5 * mm)]
m = S["holdout_2026"]["low_always"]
story += [metric_cards([
    (pct(m["total_return"], 2), "累積収益", PURPLE),
    (num(m["sharpe"]), "Sharpe", BLUE),
    (pct(m["max_drawdown"], 1), "最大DD", RED),
    (f'{m["trades"]}件', f'勝率 {pct(m["win_rate"], 1)}', GREEN),
]), Spacer(1, 5 * mm),
          p("取引一覧", H2)]
trade_rows = [["Entry", "Exit", "方向", "日数", "純損益"]]
for _, r in TRADES.iterrows():
    direction = "村田買 / 太陽売" if r.direction == "long_spread" else "村田売 / 太陽買"
    trade_rows.append([r.entry_date, r.exit_date, direction, int(r.holding_days), pct(r.trade_return, 2)])
story += [table(trade_rows, [29 * mm, 29 * mm, 56 * mm, 20 * mm, 40 * mm], font_size=6.7), Spacer(1, 3 * mm),
          p("注: 2026年だけで選ばれた後知恵候補。全取引を年度初めフラットで再計算。", SMALL), PageBreak()]

# 8 PDCA
story += [p("7. PDCA: 強いバックテストを、壊れにくい運用へ", H1),
          table([
              ["段階", "仮説 / 変更", "検証結果", "Act"],
              ["Plan", "1σルールの強さは平均回帰レジームで説明でき、ゲートで改善できる。", "設計: 2022-24、検証: 2025、最終: 2026。", "2026を見ずに透明閾値を固定。"],
              ["Do-1", "ゲートOFFで強制清算。", "2026 +2.00%、Sharpe 0.27。5/26取引を6/15に-15.74%で切り、通常exitなら6/22 -3.04%。", "通常ゲートと緊急停止を分離。"],
              ["Do-2", "ゲートは新規エントリーだけ制御。", "構造ゲート +17.31%、Sharpe 1.37、6取引。", "保有中はz/stop/時間で通常決済。"],
              ["Check", "遅いゲートでもリスク調整後の利益が残るか。", "確認ゲート +11.66%、Sharpe 0.99。常時 +33.25%には劣後。", "リターン最大化ではなく誤配備抑制として使う。"],
              ["Act", "単一ペアの“勝ち逃げ”を避ける。", "厳格探索は8/31で遅い。", "経済ウォッチリスト + 週次entry gate + 複数ペア。"],
          ], [20 * mm, 53 * mm, 64 * mm, 37 * mm], font_size=6.7), Spacer(1, 5 * mm),
          p("設計 / 検証 / 最終テスト（基準コスト）", H2)]
base = STRAT[(STRAT.tc_bp == 10) & (STRAT.borrow_rate == 0.01)]
period_labels = {"design_2022_2024": "設計 22-24", "validation_2025": "検証 2025", "holdout_2026": "最終 2026"}
strat_labels = {"low_always": "1σ常時", "low_structure_gate": "構造entry gate", "low_confirmed_gate": "確認entry gate", "conservative": "2σ保守"}
pdca_rows = [["期間", "戦略", "収益", "Sharpe", "MDD", "取引"]]
for period in period_labels:
    for strategy_name in strat_labels:
        r = base[(base.period == period) & (base.strategy == strategy_name)].iloc[0]
        pdca_rows.append([period_labels[period], strat_labels[strategy_name], pct(r.total_return), num(r.sharpe), pct(r.max_drawdown), int(r.trades)])
story += [table(pdca_rows, [30 * mm, 47 * mm, 27 * mm, 24 * mm, 26 * mm, 20 * mm], font_size=6.5), Spacer(1, 4 * mm),
          callout("負の結果を残す", "設計期間は全戦略が弱く、2025も1σ常時は-14.36%。したがって2026の高収益は構造変化か偶然かをまだ識別できない。", PALE_RED, RED), PageBreak()]

# 9 holdout comparison
story += [p("8. 2026ホールドアウト: 常時稼働は強いが配備証拠ではない", H1),
          chart("03_pdca_holdout.png", max_h=95 * mm), Spacer(1, 4 * mm)]
holdout_rows = [["戦略", "収益", "年率vol", "Sharpe", "MDD", "勝率", "取引", "平均保有"]]
for name in strat_labels:
    r = base[(base.period == "holdout_2026") & (base.strategy == name)].iloc[0]
    holdout_rows.append([strat_labels[name], pct(r.total_return), pct(r.ann_vol), num(r.sharpe), pct(r.max_drawdown), pct(r.win_rate), int(r.trades), f"{r.avg_holding_days:.1f}日"])
story += [table(holdout_rows, [42 * mm, 21 * mm, 22 * mm, 20 * mm, 22 * mm, 18 * mm, 14 * mm, 25 * mm], font_size=6.7), Spacer(1, 5 * mm),
          callout("選択バイアス", "1σ常時は2024-26のOOSを後から見た候補で、2022-24設計では-25.0%、2025検証では-14.4%。2026の+33.3%だけを見て採用すると、レジーム終了時に逆戻りする危険が高い。", PALE_ORANGE, ORANGE),
          p("現時点の研究信号", H2),
          bullet("9/1 z=+1.16。1σ研究ルールはショート村田 / ロング太陽誘電を継続判定。"),
          bullet("2σ保守ルールは新規なし。実運用の推奨はフラット。決算・借株・寄付きギャップを含む実発注検証が未了。"), PageBreak()]

# 10 stress/bootstrap
story += [p("9. コスト・借株・サンプル不確実性", H1),
          p("2026年 低閾値常時ルールの累積収益", H2)]
stress = STRAT[(STRAT.period == "holdout_2026") & (STRAT.strategy == "low_always")]
stress_rows = [["借株料", "片道5bp", "片道10bp", "片道20bp"]]
for br in [0.01, 0.03, 0.05]:
    row = [f"年{br:.0%}"]
    for tc in [5.0, 10.0, 20.0]:
        x = stress[(stress.borrow_rate == br) & (stress.tc_bp == tc)].iloc[0]
        row.append(f"{pct(x.total_return)} / S {x.sharpe:.2f}")
    stress_rows.append(row)
story += [table(stress_rows, [35 * mm, 46 * mm, 46 * mm, 47 * mm], font_size=7.2), Spacer(1, 7 * mm)]
boot = S["bootstrap_low_2026_trade_mean"]
story += [metric_cards([
    (pct(boot["mean"], 2), "1取引平均", PURPLE),
    (f'{pct(boot["ci_low"], 2)}〜{pct(boot["ci_high"], 2)}', "単純取引bootstrap 95%CI", BLUE),
    (f'{boot["trades"]}件', "観測取引数", ORANGE),
]), Spacer(1, 6 * mm),
          p("ここから言えること", H2),
          bullet("2026年内では20bp・借株5%でも+29.4%。この年の粗い強さは通常コストだけでは消えない。"),
          bullet("ただしbootstrapは11取引を独立同分布と仮定する単純再標本化。同じレジーム・同じ材料に依存し、真の不確実性を過小評価する。"),
          bullet("最大DDは全ゲートで約-15.7%。ゲートは取引数を減らしたが、5/26の悪い取引を避けられず、尾部リスクは残った。"),
          callout("必要な次の証拠", "複数ペア・複数年のウォークフォワードで、ペア選択を含めた全工程を再試験すること。単一ペアの取引bootstrapだけでは配備判断に足りない。", PALE_RED, RED), PageBreak()]

# 11 universe
story += [p("10. 再現可能なペア探索: 14銘柄・91ペア", H1),
          p("固定ユニバース（2026-09-02時点）", H2)]
urows = [["Ticker", "企業", "分類", "価格開始", "最終日"]]
for _, r in UNIVERSE.iterrows():
    urows.append([r.ticker, r["name"], r.subsector, r.first_date, r.last_date])
story += [table(urows, [25 * mm, 43 * mm, 53 * mm, 27 * mm, 26 * mm], font_size=6.4), Spacer(1, 5 * mm),
          callout("サバイバーシップ制約", "現在知られている上場企業で固定したため、過去に上場廃止・合併した企業を含まない。結果は“当時の完全な投資可能集合”ではない。", PALE_ORANGE, ORANGE),
          p("流動性フィルター", H2),
          bullet("各レッグの直近63日中央値売買代金が5億円以上。調整後終値×出来高で近似。"),
          bullet("株価はYahoo Financeのみ。ファンダメンタル指標を過去に遡って埋め戻していない。"), PageBreak()]

# 12 rank
story += [p("11. 発見順位: ウォッチは4月、厳格適格は8月", H1),
          chart("04_discovery_rank.png", max_h=90 * mm), Spacer(1, 4 * mm)]
rank_2026 = RANK[RANK.screen_date.dt.year == 2026]
rrows = [["情報日", "全体順位", "FDR q", "検証収益", "検証Sharpe", "適格"]]
for _, r in rank_2026.iterrows():
    rrows.append([date_text(r.screen_date), int(r.overall_rank), num(r.eg_q_bh, 3), pct(r.val_return), num(r.val_sharpe), "○" if r.eligible else "-"])
story += [table(rrows, [31 * mm, 29 * mm, 28 * mm, 31 * mm, 31 * mm, 24 * mm], font_size=6.9), Spacer(1, 5 * mm),
          callout("実務上の折衷", "4/30時点で総合4位、5/29に2位だったため、<b>経済ウォッチリスト</b>には入れられた。一方、91ペアの多重検定を通す厳格適格は8/31で、利益の大半の後。よって“ウォッチ入り”と“資本配賦”を分ける。", PALE_GREEN, GREEN),
          p("凍結ルールを変えなかった意味", H2),
          bullet("このペアを勝たせるためFDR閾値を緩めていない。7/31は1位でもq=0.315で不適格。"),
          bullet("厳格性は偽発見を減らす代わりに遅れる。実務は、少額パイロットとentry gateで遅延コストを管理する。"),
          p("凍結スコア", H2),
          p("経済リンク、63日売買代金、252日相関・正規化価格距離・EG/BH q・半減期、4×63日β安定、126日交差・1σ機会、先行252日回帰→後続126日コスト後検証を固定加重。月次評価日は実際の最終取引日、シグナルは1日遅延。", SMALL), PageBreak()]

# 13 playbook
story += [p("12. 推奨プレイブック: 研究信号から配備候補へ", H1),
          p("エントリー条件（すべてAND）", H2),
          table([
              ["チェック", "条件", "頻度"],
              ["Watchlist", "経済リンク≥0.85、流動性合格、総合上位5が2か月中1回以上。", "月次"],
              ["Entry gate", "相関≥0.45、β>0、βCV≤0.75、半減期2〜40日、ADF p≤0.20、交差≥4、機会≥2を2週連続。", "週次"],
              ["Signal", "252日ローリングzが±1.0を終値で突破。翌日終値執行。", "日次"],
              ["Sizing", "各レッグ25bpリスクから開始。ペアgross≤10%、同一テーマ合計gross≤30%。", "発注時"],
              ["Exit", "±0.5σ、20日、逆行4σ。ゲートOFFだけでは強制清算しない。", "日次"],
              ["Costs", "片道20bp + 借株5%ストレスでも期待値正を要求。", "事前"],
          ], [31 * mm, 116 * mm, 27 * mm], font_size=6.8), Spacer(1, 5 * mm),
          p("実行の優先順位", H2),
          table([
              ["1", "候補化", "経済ウォッチリスト。ここでは資本を置かない。"],
              ["2", "小額パイロット", "entry gate ONかつ借株・板・決算日を確認。通常サイズの25%。"],
              ["3", "昇格", "直近6か月で4取引以上、コスト後プラス、データ品質エラーなし。最大50%。"],
              ["4", "標準", "複数ペアポートフォリオで寄与が分散。単一ペアへフル配賦しない。"],
          ], [12 * mm, 34 * mm, 128 * mm], header=False, font_size=7.1), Spacer(1, 5 * mm),
          callout("このペアの現在地", "研究ウォッチ・ゲートは合格、低閾値シグナルは継続。しかしルール選択が後知恵で、ペア選択も厳格適格が8/31と遅い。したがって“配備候補”止まり。", PALE_RED, RED), PageBreak()]

# 14 dashboard and kill switches
story += [p("13. 日次 / 週次モニタリングとキルスイッチ", H1),
          p("ダッシュボード仕様", H2),
          table([
              ["頻度", "表示", "アラート"],
              ["日次", "価格、z、ポジション、保有日、gross/net PnL、turnover、借株料/在庫、次回決算日。", "|z|≥3、日次ペア損失>2%、データ欠損、借株急騰。"],
              ["週次", "相関126、β/βCV、ADF/EG p、半減期、交差126、機会63、過去126日Sharpe/勝率。", "構造gate OFF、β符号反転、半減期>40、相関<0.35。"],
              ["月次", "91ペア順位、BH q、流動性、コスト後検証、テーマ集中、ペア間PnL相関。", "上位外3か月、流動性閾値割れ、テーマgross>30%。"],
          ], [24 * mm, 103 * mm, 47 * mm], font_size=6.8), Spacer(1, 6 * mm),
          p("決定木", H2),
          table([
              ["データ正常?", "No → 新規停止。欠損補完で発注しない。", "Yes → 次へ"],
              ["ハード停止?", "Yes → 即時縮小/清算。", "No → 次へ"],
              ["保有中?", "Yes → 通常exit/stop/時間を優先。ordinary gate OFFだけで切らない。", "No → 次へ"],
              ["Watch + gate + signal?", "すべてYes → 翌日執行候補。", "いずれかNo → フラット。"],
          ], [39 * mm, 94 * mm, 41 * mm], header=False, font_size=7.1), Spacer(1, 5 * mm),
          p("ハード・キルスイッチ（通常ゲートと分離）", H2),
          bullet("借株回収、売買停止、企業行動、決算サプライズで片脚が取引不能。"),
          bullet("β符号反転が5日継続、|z|≥4、ペア累積損失が事前リスク予算の2倍、データ整合性エラー。"),
          bullet("同一テーマ全体が同時損失し、ポートフォリオ相関が急上昇。"), PageBreak()]

# 15 limitations
story += [p("14. 限界と、次のPDCA", H1),
          table([
              ["限界", "影響", "次の検証"],
              ["単一ペア・11取引", "信頼区間が不安定。イベント依存。", "20〜40ペアの横断walk-forward。"],
              ["後知恵の低閾値", "+33%を期待値と読めない。", "パラメータを固定し2026年9月以降を完全未使用テスト。"],
              ["現在ユニバース", "サバイバーシップバイアス。", "月次時点の上場銘柄・業種・借株可否を復元。"],
              ["終値約定", "寄付きギャップ、板、ショート制約を過小評価。", "VWAP/翌日寄付/出来高制約、実借株料で再試験。"],
              ["価格のみのゲート", "決算・商品構成の離散変化を遅れて検知。", "一次資料イベントフラグを事前定義し、新規停止窓を検証。"],
              ["因果推定なし", "AI/円安が利益を生んだと断定不可。", "業界指数・為替・市場βを除去した残差で機構を再検証。"],
              ["多重検定", "FDR対応でもモデル選択全体の楽観性は残る。", "White Reality Check / SPA、nested walk-forward。"],
          ], [40 * mm, 66 * mm, 68 * mm], font_size=6.9), Spacer(1, 6 * mm),
          p("次の90日PDCA", H2),
          bullet("Plan: 14銘柄を維持し、月次top5から最大5ペア、各ペアgross 5%で紙上運用。"),
          bullet("Do: 週次entry gate、決算前後1営業日は新規停止、翌日VWAP近似で約定。"),
          bullet("Check: 選択込みPnL、コスト後Sharpe、hit rate、テーマ集中、ゲートの機会損失を記録。"),
          bullet("Act: 20取引未満なら閾値を再最適化しない。ルール違反とデータ障害だけを修正。"),
          callout("採用条件", "少なくとも30〜50取引、複数ペア、未使用期間、ストレスコスト後で正の期待値を確認するまで、本番資本へ昇格しない。", PALE_GREEN, GREEN), PageBreak()]

# 16 references/reproducibility
story += [p("15. データ、一次資料、再現性", H1),
          p("一次資料（公表日 / 閲覧日 2026-09-02）", H2)]
ref_rows = [["ID", "発行元 / 資料", "公表日", "リンク"]]
for src in SOURCES:
    pub = src["publication_date"] or "-"
    url = src["url"]
    visible = url.replace("https://", "").replace("http://", "")
    if len(visible) > 52:
        visible = visible[:49] + "..."
    link = f'<link href="{url}" color="#1479B8">{visible}</link>'
    ref_rows.append([src["id"], f'{src["publisher"]}<br/>{src["title"]}', pub, link])
story += [table(ref_rows, [12 * mm, 73 * mm, 24 * mm, 65 * mm], font_size=5.9), Spacer(1, 4 * mm),
          p("価格・計算", H2),
          bullet("価格: Yahoo Finance調整後終値、2021-01-04〜2026-09-01。14銘柄。9/2終値は未収録。"),
          bullet("コード: work/regime_research/analyze_regime.py。主要出力: analysis_summary.json、validation_checks.json、regime_diagnostics_weekly.csv、monthly_pair_screen_all.csv。"),
          bullet("検査: PnL再構成誤差0、gross exposure最大1.0、11取引=11 entry、1日遅延一致、全評価日≤カットオフ。"),
          p("免責", H2),
          p("本資料は過去データの調査であり、特定銘柄の売買推奨ではありません。税、配当落ち、実際の貸株料、注文失敗、価格制限、企業行動、信用規制、容量制約を完全には反映していません。", SMALL)]

doc.build(story, onFirstPage=page_canvas, onLaterPages=page_canvas)
print(OUT)
