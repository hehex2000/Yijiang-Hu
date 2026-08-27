# -*- coding: utf-8 -*-
"""
chan_lun_core_faithful.py — 缠论(缠中说禅)理论文档的忠实算子化实现
================================================================
目标：把缠论标准定义(分型/笔/线段/中枢/背驰/三类买卖点)忠实代码化，
     使之可被回测检验 —— 重点在"可量化、可被证伪"，不在"证明有效"。

无未来函数约定：本模块所有函数都是"纯函数"，接收"截至 T-1 的序列"，
返回"在 T-1 这一刻能确定的结构"。调用方负责把序列截断到 T-1，
信号在 T 开盘(或 T+lag)执行。本文件本身不读时间、不偷看未来。

严格对齐的理论规则(摘自《教你炒股票》系列)：
  1. 包含关系：相邻K线有包含(一者高≥另一高且低≤另一低)时，按前趋方向合并
     (向上包含→高取max/低取max；向下包含→高取min/低取min)，合并后取较近一根的索引。
  2. 分型：处理后序列上，中间K线高严格大于两侧=顶分型；低严格小于两侧=底分型。
  3. 笔：两反向分型构成；两分型中心至少间隔 MIN_GAP=4 根处理K线(≈≥5根原K线)；
     同向更极值分型延伸笔；反向分型间隔足够即收尾确认(不足间隔忽略，属不成熟)。
  4. 线段：≥3笔严格交替(上升线段=每上升笔高>前高且每下降笔低>前低)；
     线段破坏=反向线段(≥3笔)打穿本线段起点极值。
  5. 中枢：线段内≥3笔价格区间的几何交集 [ZD,ZG]=[三笔低点之最大, 三笔高点之最小]；
     连续重叠则合并为同一中枢。纯几何，与波动率无关。
  6. 背驰：同向两离开段(笔)比较 MACD 力度(面积=∑|DIF|)；价格创新极值但面积更小=背驰。
  7. 三类买卖点(相对真实中枢区间[ZD,ZG])：
     一买=趋势背驰转折点(底背驰末端)；二买=回踩不破一买低点；三买=回抽不进中枢(低>ZG)。
     一卖=顶背驰末端；二卖=反弹不过一卖高点；三卖=回抽不进中枢(高<ZD)。

依赖：numpy / pandas。
"""
import numpy as np
import pandas as pd

MIN_GAP_BI = 4  # 两分型中心最少间隔的处理K线数(≈≥5根原K线)


# =====================================================================
# 1. K线包含关系处理
# =====================================================================
def merge_inclusion(highs, lows):
    """处理后序列：相邻包含K线按前趋方向合并。
    返回 (proc_h, proc_l, proc_idx)：处理后高/低数组 + 每根对应的原始bar索引。
    合并后取较近(较大)的原始索引，作为该处理K线的时间位置(分型在合并末端定型)。
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    n = len(highs)
    if n == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    ph = [highs[0]]
    pl = [lows[0]]
    pi = [0]
    for i in range(1, n):
        h, l = highs[i], lows[i]
        cur_h, cur_l = ph[-1], pl[-1]
        incl = (cur_h >= h and cur_l <= l) or (h >= cur_h and l <= cur_l)
        if not incl:
            ph.append(h); pl.append(l); pi.append(i)
            continue
        # 合并方向由"合并前两根处理K线"的趋势决定
        if len(ph) >= 2:
            up = ph[-1] > ph[-2]
        else:
            up = h >= cur_h
        if up:
            nh = max(cur_h, h); nl = max(cur_l, l)
        else:
            nh = min(cur_h, h); nl = min(cur_l, l)
        ph[-1] = nh; pl[-1] = nl
        pi[-1] = i  # 合并末端索引
    return np.array(ph), np.array(pl), np.array(pi, dtype=int)


# =====================================================================
# 2. 分型(处理后序列)
# =====================================================================
def fractals_processed(h, l, idx):
    """返回 (tops, bots)：每项为 (pos, orig_idx, price)。
    pos=在处理序列中的位置；orig_idx=对应原始bar索引；price=极值。
    """
    tops, bots = [], []
    n = len(h)
    for i in range(1, n - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            tops.append((i, int(idx[i]), float(h[i])))
        if l[i] < l[i - 1] and l[i] < l[i + 1]:
            bots.append((i, int(idx[i]), float(l[i])))
    return tops, bots


# =====================================================================
# 3. 笔
# =====================================================================
def build_bi(tops, bots):
    """由分型构建笔序列。返回 list of dict：
    {dir:'up'/'down', s_pos,s_idx,s_pri, e_pos,e_idx,e_pri}
    up笔: 底分型->顶分型；down笔: 顶分型->底分型。
    规则：两分型中心间隔≥MIN_GAP；同向更极值延伸；反向不破前笔起点不确认。
    """
    allf = [(p, i, pr, 'top') for (p, i, pr) in tops] + \
           [(p, i, pr, 'bot') for (p, i, pr) in bots]
    allf.sort(key=lambda x: x[0])
    bi = []
    pending = None       # 当前未闭合笔的起点分型
    for (pos, oidx, price, kind) in allf:
        if pending is None:
            pending = dict(kind=kind, pos=pos, idx=oidx, price=price)
            continue
        if kind == pending['kind']:
            # 同向：更极值则延伸起点(更新笔起点为更极值者)
            more = (kind == 'top' and price > pending['price']) or \
                   (kind == 'bot' and price < pending['price'])
            if more:
                pending = dict(kind=kind, pos=pos, idx=oidx, price=price)
            continue
        # 反向分型
        if pos - pending['pos'] < MIN_GAP_BI:
            # 间隔不足：忽略该反向分型(不成熟)，保留 pending
            continue
        # 确认一笔 pending->当前（标准缠论笔：反向分型间隔≥MIN_GAP 即收尾确认；
        # 同向更极值已在上文延伸处理；间隔不足的反向分型忽略。此即"反向分型出现即确认"，
        # 与项目报告记载的忠实内核笔数口径一致——4.1 版曾误加"须突破前一笔极值(HH/LL)"的过严
        # 门槛，把 250+ 笔塌缩到个位数、线段塌到 1~2，等价于结构失效，已撤销。）
        d = 'up' if pending['kind'] == 'bot' else 'down'
        bi.append(dict(dir=d, s_pos=pending['pos'], s_idx=pending['idx'], s_pri=pending['price'],
                       e_pos=pos, e_idx=oidx, e_pri=price))
        pending = dict(kind=kind, pos=pos, idx=oidx, price=price)
    # pending 若最后还有一根未闭合，若与已有最后笔构成合规(间隔够)则补一笔
    if pending is not None and bi:
        last = bi[-1]
        if (pending['kind'] != last['dir'][0] and  # 反向
                pending['pos'] - last['e_pos'] >= MIN_GAP_BI):
            d = 'up' if pending['kind'] == 'bot' else 'down'
            bi.append(dict(dir=d, s_pos=last['e_pos'], s_idx=last['e_idx'], s_pri=last['e_pri'],
                           e_pos=pending['pos'], e_idx=pending['idx'], e_pri=pending['price']))
    return bi


# =====================================================================
# 4. 线段(特征序列/3笔严格交替 + 线段破坏)
# =====================================================================
def build_segments(bi):
    """由笔构建线段。返回 list of dict：
    {dir, bi_indices(list), s_idx,e_idx, s_pri,e_pri, zhongshu:[(zd,zg,b0,b1,b2)...]}
    线段破坏(缠论"三笔反向破坏"标准简化)：
      bi 序列方向严格交替，故"反向笔"指与当前线段方向相反之笔。
      反向笔持续累积(跨多次回调不清零)，每当出现第 ≥3 个反向笔，取最近三笔(a,b,c)，
      若 c 的端点较 a 更极端(向下线段 c 低点<a 低点；向上线段 c 高点>a 高点)，
      即判定原线段被该反向走势破坏。新线段自 a 起(含其后的回调笔)。
      —— 此前版本误用"打穿整段起点极值"作破坏门槛，对长期单边标的永不触发(线段=1)，已弃用。
    """
    if len(bi) < 3:
        return []
    segs = []
    cur = [0]  # 当前线段包含的笔索引
    cur_dir = bi[0]['dir']
    opp = []   # 与 cur_dir 反向的笔索引序列(跨多次回调累积)

    for k in range(1, len(bi)):
        nb = bi[k]
        if nb['dir'] == cur_dir:
            # 同向笔：延伸当前线段（不重置 opp，保留跨多次回调的反向累积）
            cur.append(k)
        else:
            opp.append(k)
            if len(opp) >= 3:
                a, b, c = opp[-3], opp[-2], opp[-1]
                rdir = bi[a]['dir']
                if rdir == 'down':
                    broken = bi[c]['e_pri'] < bi[a]['e_pri']
                else:
                    broken = bi[c]['e_pri'] > bi[a]['e_pri']
                if broken:
                    segs.append(_finalize_seg(bi, cur, cur_dir))
                    # 新线段：自三笔破坏起点 a 起，含其后的回调笔
                    cur = list(range(a, k + 1))
                    cur_dir = rdir
                    opp = []
                    continue
            # 未破坏：nb 作为线段内回调笔归属当前线段
            cur.append(k)
    if cur:
        segs.append(_finalize_seg(bi, cur, cur_dir))
    return segs


def _finalize_seg(bi, idx_list, direction):
    sub = [bi[i] for i in idx_list]
    s = sub[0]; e = sub[-1]
    # 中枢检测：线段内≥3笔价格区间几何交集
    zhongshu = _detect_zhongshu(sub, idx_list)
    return dict(dir=direction, bi_indices=list(idx_list),
                s_idx=s['s_idx'], e_idx=e['e_idx'],
                s_pri=s['s_pri'], e_pri=e['e_pri'],
                zhongshu=zhongshu)


def _detect_zhongshu(bi_sub, idx_list):
    """线段内连续3笔重叠检测，连续重叠则合并为同一中枢块。
    返回 [(zd,zg,b0,b1,b2)...]，其中 b0/b1/b2 为全局 bi 索引；
    (zd,zg) 取构成中枢的"真实3笔"边界(不向后延伸)，确保三类买卖点判定无未来函数。
    """
    res = []
    i = 0
    n = len(bi_sub)
    while i + 2 < n:
        za, zb, zc = bi_sub[i], bi_sub[i + 1], bi_sub[i + 2]
        lows = [min(x['s_pri'], x['e_pri']) for x in (za, zb, zc)]
        highs = [max(x['s_pri'], x['e_pri']) for x in (za, zb, zc)]
        zd = max(lows); zg = min(highs)
        if zd <= zg:  # 三笔区间有交集 -> 构成中枢
            # 向后延伸合并(仅用于判定哪些笔同属一块中枢，不影响输出边界)
            j = i + 3
            while j < n:
                nxt = bi_sub[j]
                nl = min(nxt['s_pri'], nxt['e_pri']); nh = max(nxt['s_pri'], nxt['e_pri'])
                if max(zd, nl) <= min(zg, nh):
                    zd, zg = max(zd, nl), min(zg, nh)
                    j += 1
                else:
                    break
            # 输出用"真实3笔"边界(无未来函数) + 回写全局 bi 索引
            zla = min(za['s_pri'], za['e_pri']); zha = max(za['s_pri'], za['e_pri'])
            zlb = min(zb['s_pri'], zb['e_pri']); zhb = max(zb['s_pri'], zb['e_pri'])
            zlc = min(zc['s_pri'], zc['e_pri']); zhc = max(zc['s_pri'], zc['e_pri'])
            zzd = max(zla, zlb, zlc); zzg = min(zha, zhb, zhc)
            res.append((zzd, zzg, idx_list[i], idx_list[i + 1], idx_list[i + 2]))
            i = j  # 跳到合并后的下一笔
        else:
            i += 1
    return res


# =====================================================================
# 5. 背驰(MACD 面积比较)
# =====================================================================
def macd_dif(closes, fast=12, slow=26):
    s = pd.Series(np.asarray(closes, dtype=float))
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    return (ema_f - ema_s).values


def _macd_area(dif, i0, i1):
    """i0..i1(含)区间 MACD 力度面积 = ∑|DIF|。"""
    i0 = max(0, min(i0, i1)); i1 = max(i0, max(i0, i1))
    seg = dif[i0:i1 + 1]
    return float(np.sum(np.abs(seg)))


def detect_beichi(bi, dif, segs):
    """对每笔标注是否背驰。返回 dict: bi_index -> 'bull'/'bear'/None。
    bull(底背驰)=该down笔价格创新低但 MACD面积 < 前一下跌笔面积；
    bear(顶背驰)=该up笔价格创新高但面积 < 前一上涨笔面积。
    仅在同一线段(走势类型)内的同向相邻笔之间比较——跨线段不比较(4.2 修正)。
    """
    bi_seg = {}
    for si, seg in enumerate(segs):
        for bii in seg.get('bi_indices', []):
            bi_seg[bii] = si
    res = {k: None for k in range(len(bi))}
    prev_up = prev_down = None  # (idx, area, price_ext)
    cur_seg = None
    for k in range(len(bi)):
        b = bi[k]
        seg = bi_seg.get(k)
        if seg is None:
            # 过渡笔(不属于任何线段)：不参与背驰比较，也不作为走势类型组，
            # 且重置对照状态，避免连续过渡笔之间互相污染(None==None 不会再次重置)
            prev_up = prev_down = None
            cur_seg = None
            continue
        if seg != cur_seg:
            # 进入新走势类型：重置对照(背驰只在同走势类型内比较)
            prev_up = prev_down = None
            cur_seg = seg
        i0, i1 = b['s_idx'], b['e_idx']
        area = _macd_area(dif, i0, i1)
        if b['dir'] == 'up':
            ext = b['e_pri']
            if prev_up is not None and ext > prev_up[2] and area < prev_up[1]:
                res[k] = 'bear'
            prev_up = (k, area, ext)
        else:
            ext = b['e_pri']
            if prev_down is not None and ext < prev_down[2] and area < prev_down[1]:
                res[k] = 'bull'
            prev_down = (k, area, ext)
    return res


# =====================================================================
# 6. 三类买卖点(相对真实中枢区间 [ZD,ZG])
# =====================================================================
def detect_trade_points(segs, bi, beichi):
    """返回 (buys, sells)：各为 list of (orig_idx, type)。
    买点：b1=底背驰末端(趋势背驰)；b2=回踩不破b1低；b3=回抽不进中枢(低>ZG)。
    卖点：s1=顶背驰末端；s2=反弹不过s1高；s3=回抽不进中枢(高<ZD)。
    """
    buys, sells = [], []
    # 先做笔级买卖点(基于背驰)
    for k in range(len(bi)):
        b = bi[k]
        if beichi.get(k) == 'bull' and b['dir'] == 'down':
            buys.append((b['e_idx'], 'b1'))
        if beichi.get(k) == 'bear' and b['dir'] == 'up':
            sells.append((b['e_idx'], 's1'))
    # b1 之后找 b2(回踩不破b1低) / b3(中枢后回抽不进中枢)
    for (b1_idx, t) in list(buys):
        if t != 'b1':
            continue
        b1_low = bi[_bi_index_at(bi, b1_idx)]['e_pri']
        # 找 b1 之后的 next down 笔(回调)
        for k in range(len(bi)):
            b = bi[k]
            if b['e_idx'] <= b1_idx or b['dir'] != 'down':
                continue
            # b2: 回踩低点 > b1 低点
            if b['e_pri'] > b1_low and (b['e_idx'], 'b2') not in buys:
                buys.append((b['e_idx'], 'b2'))
                break
    # b3/s3: 中枢后第一次回调(三买)/反弹(三卖)不进中枢即触发。
    # 4.3 修正：找中枢后"第一笔"同方向笔(未必紧邻 b2+1)，且 b2 现为全局 bi 索引；
    #         判定用真实3笔中枢边界(zd,zg)，无未来函数。
    for seg in segs:
        for (zd, zg, b0, b1, b2) in seg['zhongshu']:
            # 三买：中枢后第一笔向下回调(down)且低点>ZG
            kk = b2 + 1
            while kk < len(bi):
                nb = bi[kk]
                if nb['dir'] == 'down':
                    if min(nb['s_pri'], nb['e_pri']) > zg:
                        if (nb['e_idx'], 'b3') not in buys:
                            buys.append((nb['e_idx'], 'b3'))
                    break  # 第一笔回调即决定三买，不再往后找
                kk += 1
            # 三卖：中枢后第一笔向上反弹(up)且高点<ZD
            kk = b2 + 1
            while kk < len(bi):
                nb = bi[kk]
                if nb['dir'] == 'up':
                    if max(nb['s_pri'], nb['e_pri']) < zd:
                        if (nb['e_idx'], 's3') not in sells:
                            sells.append((nb['e_idx'], 's3'))
                    break
                kk += 1
    # s2: 一卖之后反弹不过一卖高
    for (s1_idx, t) in list(sells):
        if t != 's1':
            continue
        s1_high = bi[_bi_index_at(bi, s1_idx)]['e_pri']
        for k in range(len(bi)):
            b = bi[k]
            if b['e_idx'] <= s1_idx or b['dir'] != 'up':
                continue
            if b['e_pri'] < s1_high and (b['e_idx'], 's2') not in sells:
                sells.append((b['e_idx'], 's2'))
                break
    # 去重并排序
    buys = sorted(set(buys))
    sells = sorted(set(sells))
    return buys, sells


def _bi_index_at(bi, e_idx):
    for k, b in enumerate(bi):
        if b['e_idx'] == e_idx:
            return k
    return 0


# =====================================================================
# 7. 总装：一次性计算全序列结构
# =====================================================================
def compute_states(highs, lows, closes):
    """输入原始 OHLC 数组(已截断到 T-1)，输出完整缠论状态。
    返回 dict：
      bi, segments, beichi, buys[(idx,type)], sells[(idx,type)],
      last_dir('up'/'down'/'neutral' 基于最后线段方向)
    """
    ph, pl, pidx = merge_inclusion(highs, lows)
    if len(ph) < 3:
        return dict(bi=[], segments=[], beichi={}, buys=[], sells=[], last_dir='neutral')
    tops, bots = fractals_processed(ph, pl, pidx)
    bi = build_bi(tops, bots)
    if len(bi) < 3:
        return dict(bi=bi, segments=[], beichi={}, buys=[], sells=[], last_dir='neutral')
    dif = macd_dif(closes)
    segs = build_segments(bi)
    beichi = detect_beichi(bi, dif, segs)
    buys, sells = detect_trade_points(segs, bi, beichi)
    last_dir = segs[-1]['dir'] if segs else 'neutral'
    return dict(bi=bi, segments=segs, beichi=beichi,
                buys=buys, sells=sells, last_dir=last_dir)


# =====================================================================
# 8. 流式信号发生器（逐 Bar 因果 + 去闪烁，供平台接入复用）
# =====================================================================
class ChanLunStream:
    """逐 Bar 流式缠论买卖点发生器。

    设计目的：让 run_monthly_rebalance.py 等「月度/事件驱动」回测也能安全复用
    本内核的买卖点，而**不重新引入未来函数**。逻辑与 run_chan_lun_faithful.run_backtest
    的逐 Bar 循环逐字一致（SETTLE 稳定化 + done 去重），仅把「一次性全序列扫描」
    改成「喂一根处理一根」的流式接口。

    用法：
        s = ChanLunStream()
        s.seed(highs, lows, closes)        # 灌入截至选股日的全历史，标记已有信号为已消费
        while 有新bar:
            sig = s.feed(h, l, c)          # 喂入当日 H/L/C
            if sig and sig[0] == "BUY":     # 命中买点，于下一根开盘执行
                ...
    无未来函数保证：feed 内部只调用 compute_states(self.H[:t+1], ...)（截至当日），
    信号经 SETTLE 连续前缀门槛 + done 去重，二者皆仅依赖 ≤t 数据。
    """

    def __init__(self, lag=1, warmup=120, SETTLE=2):
        self.lag = lag
        self.warmup = warmup
        self.SETTLE = SETTLE
        self.H, self.L, self.C = [], [], []
        self.persist = {}      # (e_idx,type) -> 连续出现前缀数
        self.done = set()      # 已触发过的信号(防长周期闪烁重复触发)
        self.t = -1

    def seed(self, highs, lows, closes):
        """灌入截至某日的全历史并标记所有「当前已有信号」为已消费(done)，
        使后续只对新出现的买卖点触发。O(n) 一次性，不会重放逐 Bar。"""
        self.H = [float(x) for x in highs]
        self.L = [float(x) for x in lows]
        self.C = [float(x) for x in closes]
        self.t = len(self.H) - 1
        if self.t < self.warmup:
            self.done = set()
            self.persist = {}
            return
        st = compute_states(self.H, self.L, self.C)
        # 历史已出现过的买卖点视为「已消费」，不回放触发
        self.done = set(st["buys"]) | set(st["sells"])
        self.persist = {}

    def feed(self, h, l, c):
        """喂入当日 H/L/C，返回该 bar 确认的买/卖信号 (("BUY"|"SELL", (e_idx,type))) 或 None。
        信号在 t 确认，调用方应于 t+lag 开盘执行。"""
        self.H.append(float(h))
        self.L.append(float(l))
        self.C.append(float(c))
        self.t += 1
        if self.t < self.warmup:
            return None
        st = compute_states(self.H, self.L, self.C)
        cur = set(st["buys"]) | set(st["sells"])
        # 仅当前出现的信号累加前缀；未出现则自然衰减(与逐 Bar 引擎一致)
        new_persist = {}
        for sig in cur:
            new_persist[sig] = self.persist.get(sig, 0) + 1
        self.persist = new_persist
        buys = set(st["buys"])
        fired = None
        for sig in cur:
            if self.persist[sig] >= self.SETTLE and sig not in self.done:
                fired = ("BUY" if sig in buys else "SELL", sig)
                self.done.add(sig)
                break
        return fired


if __name__ == '__main__':
    # 冒烟测试：合成一段 上升-盘整-下跌，验证不崩且结构非空
    np.random.seed(0)
    n = 300
    closes = np.cumsum(np.random.randn(n)) + 100
    highs = closes + np.abs(np.random.randn(n)) * 0.5 + 0.3
    lows = closes - np.abs(np.random.randn(n)) * 0.5 - 0.3
    st = compute_states(highs, lows, closes)
    print("SMOKE bi=%d segments=%d buys=%d sells=%d last_dir=%s"
          % (len(st['bi']), len(st['segments']), len(st['buys']), len(st['sells']), st['last_dir']))
    print("OK" if st['bi'] and st['segments'] else "EMPTY")
