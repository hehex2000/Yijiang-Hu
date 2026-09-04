# -*- coding: utf-8 -*-
"""按 AST 定位的源码区间, 把 argparse help 里的裸 % 转义成 %%

只改 help= 关键字参数的字符串字面量源码区间(含引号), 不碰同行的其他内容。
幂等: 已经是 %% 或 %(name)s 形式的跳过。
"""
import ast, re, subprocess

FILES = sorted(f for f in subprocess.run(
    ["git", "ls-files"], capture_output=True, text=True).stdout.split()
    if f.endswith(".py"))

# %(name)s / %s / %d / %f ... 合法转换符: 原样保留
LEGAL = re.compile(r"%(?:\([^)]*\))?[-#0 +]*[\d*]*(?:\.(\d+|\*))?[diouxXeEfFgGcrsa%]")


def escape_segment(seg):
    """转义 seg 里的裸 %, 返回 (新串, 修改处数)"""
    out, i, n, cnt = [], 0, len(seg), 0
    while i < n:
        ch = seg[i]
        if ch == "%":
            if i + 1 < n and seg[i + 1] == "%":        # 已转义
                out.append("%%"); i += 2; continue
            m = LEGAL.match(seg, i)                     # 合法转换符
            if m:
                out.append(m.group(0)); i = m.end(); continue
            out.append("%%"); cnt += 1; i += 1          # 裸 % -> %%
        else:
            out.append(ch); i += 1
    return "".join(out), cnt


def targets_of(src):
    """返回 help 字符串节点的绝对源码区间 [(start,end,lineno)]"""
    tree = ast.parse(src)
    offs = [0]
    for ln in src.splitlines(keepends=True):
        offs.append(offs[-1] + len(ln))
    res = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            v = kw.value
            # 只处理"求值后直接作为 help"的串; "..." % x 的 BinOp 交给 Python 先求值, 不动
            if isinstance(v, ast.BinOp):
                v = v.left
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                s = offs[v.lineno - 1] + v.col_offset
                e = offs[v.end_lineno - 1] + v.end_col_offset
                res.append((s, e, v.lineno))
            elif isinstance(v, ast.JoinedStr):
                s = offs[v.lineno - 1] + v.col_offset
                e = offs[v.end_lineno - 1] + v.end_col_offset
                res.append((s, e, v.lineno))
    return res


total_files, total_fix = 0, 0
for f in FILES:
    try:
        src = open(f, encoding="utf-8").read()
    except Exception:
        continue
    try:
        tg = targets_of(src)
    except SyntaxError:
        continue
    if not tg:
        continue
    edits = []
    for s, e, ln in tg:
        seg = src[s:e]
        new_seg, cnt = escape_segment(seg)
        if cnt:
            edits.append((s, e, seg, new_seg, ln, cnt))
    if not edits:
        continue
    # 从后往前应用, 避免偏移错乱
    for s, e, seg, new_seg, ln, cnt in sorted(edits, reverse=True):
        src = src[:s] + new_seg + src[e:]
    open(f, "w", encoding="utf-8", newline="").write(src)
    total_files += 1
    for s, e, seg, new_seg, ln, cnt in sorted(edits):
        total_fix += cnt
        print(f"{f}:{ln}  x{cnt}")
        print(f"    - {seg}")
        print(f"    + {new_seg}")

print()
print(f"共修改 {total_files} 个文件, {total_fix} 处 %")
