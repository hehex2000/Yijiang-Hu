# -*- coding: utf-8 -*-
"""AST 精确扫描: argparse add_argument(help=...) 里未转义的 %

argparse 在 HelpFormatter._expand_help 里对 help 做 `help % params`。
所以 help 求值后的字符串里, 任何非 %% / 非 %(name)s 的裸 % 都会让 --help 崩溃。

两类情况:
  A. help 是普通串/f-string -> 求值后是最终 help, 裸 % 会被 argparse 格式化 -> 崩
  B. help 是 `"...%s" % x` 的 BinOp -> Python 层先求值, 再交给 argparse 二次格式化
     - left 里 %s/%d/%(k)s 是合法转换符, 求值后消失 -> 安全
     - left 里若有 `30%中` 这种非法转换符 -> Python 求值当场就崩 -> 也是 bug
"""
import ast, io, os, re, subprocess, sys

FILES = sorted(f for f in subprocess.run(
    ["git", "ls-files"], capture_output=True, text=True).stdout.split()
    if f.endswith(".py"))

VALID_CONV = set("diouxXeEfFgGcrsa%")   # % 后可接的合法转换符/标记
tokens = re.compile(r"%(?:\(([^)]*)\))?([-#0 +]*)([\d*]*)?(?:\.(\d+|\*))?([diouxXeEfFgGcrsa%]?)")


def scan_const_str(s):
    """返回常量字符串里非法 % 的位置列表 (index, 原文片段)"""
    bad = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "%":
            i += 1
            continue
        m = tokens.match(s, i)
        if m and m.group(5):          # 有合法转换符
            i = m.end()
            continue
        bad.append((i, s[i:i + 12]))
        i += 1
    return bad


def nodes_to_check(help_node):
    """返回 [(node, kind)] kind: 'direct' | 'binop_left'"""
    if isinstance(help_node, ast.BinOp) and isinstance(help_node.op, ast.Mod):
        return [(help_node.left, "binop_left")]
    if isinstance(help_node, (ast.Constant, ast.JoinedStr)):
        return [(help_node, "direct")]
    return []


results = []
for f in FILES:
    try:
        src = open(f, encoding="utf-8", errors="ignore").read()
        tree = ast.parse(src)
    except Exception:
        continue
    lines = src.splitlines()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            for sub, kind in nodes_to_check(kw.value):
                if kind == "binop_left":
                    if not (isinstance(sub, ast.Constant) and isinstance(sub.value, str)):
                        continue
                    bad = scan_const_str(sub.value)
                    note = "BinOp左串(Python求值即崩)"
                elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    bad = scan_const_str(sub.value)
                    note = "普通串"
                elif isinstance(sub, ast.JoinedStr):
                    # 只看 f-string 里的常量片段, 跳过 {...} 表达式
                    txt = "".join(v.value for v in sub.values
                                  if isinstance(v, ast.Constant) and isinstance(v.value, str))
                    bad = scan_const_str(txt)
                    note = "f-string常量片段"
                else:
                    continue
                if bad:
                    seg = ast.get_source_segment(src, kw.value) or ""
                    results.append((f, kw.value.lineno, note, seg[:100],
                                    [b[1] for b in bad]))

print("AST 扫描文件数: %d" % len(FILES))
print("含非法 %% 的 help: %d 处, 分布 %d 个文件\n" %
      (len(results), len(set(r[0] for r in results))))
for f, ln, note, seg, bads in results:
    print(f"{f}:{ln}  [{note}]")
    print(f"    片段: {seg}")
    print(f"    非法%: {bads}\n")

# 待修清单写到文件, 供替换脚本使用
with open("_pct_targets.txt", "w", encoding="utf-8") as fh:
    for f, ln, note, seg, bads in results:
        fh.write("%s\t%d\t%s\n" % (f, ln, note))
