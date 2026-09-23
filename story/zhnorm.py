"""Text normalization for lexical matching of Chinese lyrics.

Matching happens on a NORMALIZED form: traditional->simplified (curated table,
or OpenCC when installed), full-width->half-width, lowercase, punctuation and
whitespace stripped. Display always uses the ORIGINAL text.

The built-in table only needs to cover common lyric/title vocabulary — a miss
just weakens one match, it never crashes anything.
"""

from __future__ import annotations

import re
import unicodedata

# Curated traditional -> simplified pairs (common lyric/title characters).
_T2S_PAIRS = (
    "愛爱 說说 給给 彎弯 樂乐 飛飞 陽阳 聽听 見见 覺觉 夢梦 淚泪 離离 讓让 從从 "
    "來来 時时 間间 裡里 裏里 後后 過过 還还 沒没 開开 關关 頭头 遠远 邊边 這这 "
    "為为 麼么 們们 記记 憶忆 點点 燈灯 華华 風风 雲云 電电 車车 馬马 鳥鸟 魚鱼 "
    "龍龙 鳳凤 學学 寫写 讀读 書书 話话 語语 詞词 詩诗 聲声 樣样 種种 願愿 應应 "
    "當当 會会 誰谁 幾几 歲岁 對对 錯错 難难 題题 問问 變变 壞坏 傷伤 憂忧 煩烦 "
    "惱恼 滿满 圓圆 單单 雙双 隻只 兩两 場场 戲戏 劇剧 終终 結结 續续 緣缘 紅红 "
    "綠绿 藍蓝 顏颜 髮发 臉脸 齒齿 腳脚 靈灵 體体 膚肤 溫温 熱热 涼凉 霧雾 閃闪 "
    "晝昼 曉晓 鏡镜 門门 牆墙 樓楼 橋桥 島岛 灘滩 樹树 葉叶 實实 擁拥 親亲 牽牵 "
    "鬆松 緊紧 鎖锁 鑰钥 謊谎 諾诺 約约 認认 識识 並并 現现 絕绝 決决 獲获 捨舍 "
    "棄弃 掛挂 繫系 聯联 絡络 訊讯 傳传 遞递 紙纸 筆笔 畫画 圖图 攝摄 錄录 憾憾 "
    "遺遗 戀恋 談谈 歡欢 慶庆 禮礼 節节 氣气 息(same) 陣阵 隨随 轉转 動动 靜静 "
    "運运 進进 連连 遲迟 速(same) 慢(same) 趕赶 醒(same) 睡(same) 夢梦 醉(same) "
    "煙烟 塵尘 灑洒 濕湿 亂乱 齊齐 整(same) 舊旧 新(same) 老(same) 幼(same) 孩(same) "
    "兒儿 童(same) 嬰婴 媽妈 爸(same) 爺爷 奶(same) 婆(same) 孫孙 歸归 鄉乡 國国 "
    "區区 縣县 鎮镇 村(same) 廣广 場场 廳厅 房(same) 廚厨 衛卫 濟济 經经 營营 買买 "
    "賣卖 價价 錢钱 費费 貴贵 賤贱 窮穷 富(same) 產产 業业 職职 務务 員员 師师 "
    "長长 級级 班(same) 隊队 團团 體体 眾众 群(same) 獨独 孤(same) 寂(same) 寞(same) "
    "靜静 默(same) 語语 詢询 訴诉 訴诉 講讲 談谈 論论 議议 評评 讚赞 譽誉 罵骂 "
    "責责 備备 讓让 護护 衛卫 守(same) 攻(same) 擊击 戰战 爭争 勝胜 敗败 輸输 贏赢 "
    "獎奖 罰罚 賽赛 競竞 強强 弱(same) 剛刚 柔(same) 硬(same) 軟软 輕轻 重(same) "
    "濃浓 淡(same) 深(same) 淺浅 高(same) 矮(same) 寬宽 窄(same) 厚(same) 薄(same) "
    "粗(same) 細细 尖(same) 鈍钝 銳锐 利(same) 鈍钝 圓圆 方(same) 直(same) 曲(same) "
    "斜(same) 正(same) 歪(same) 順顺 逆(same) 反(same) 復复 複复 雜杂 純纯 淨净 "
    "髒脏 汙污 濁浊 清(same) 澈(same) 濤涛 潤润 澤泽 灣湾 灘滩 潭(same) 溪(same) "
    "澗涧 瀑(same) 泉(same) 湧涌 沉(same) 浮(same) 漂(same) 流(same) 淌(same) 滴(same) "
    "灑洒 濺溅 昇升 陞升 墜坠 墮堕 蟲虫 螞蚂 蟻蚁 蝴(same) 蝶(same) 蜜(same) 蜂(same)"
)

_PUNCT_RE = re.compile(r"[^0-9a-z一-鿿]+")   # non-CJK/alnum -> word boundary


def _build_t2s() -> dict[str, str]:
    table: dict[str, str] = {}
    for pair in _T2S_PAIRS.split():
        if "(same)" in pair or len(pair) < 2:
            continue
        t, s = pair[0], pair[1]
        table[t] = s
    return table


_T2S = _build_t2s()

try:  # prefer OpenCC when the environment has it (full coverage)
    from opencc import OpenCC  # type: ignore
    _CC = OpenCC("t2s")

    def _t2s(text: str) -> str:
        return _CC.convert(text)
except Exception:
    def _t2s(text: str) -> str:
        return "".join(_T2S.get(ch, ch) for ch in text)


def normalize(text: str) -> str:
    """Canonical matching form: NFKC, lowercase, t2s, punctuation removed.

    CJK runs concatenate directly; ASCII words keep a single space between them
    so substring matching can never run across an English word boundary
    ("you're all" must not match "really")."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _t2s(t)
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"(?<=[一-鿿])\s+|\s+(?=[一-鿿])", "", t)   # no space at CJK edges
    return re.sub(r"\s+", " ", t).strip()
