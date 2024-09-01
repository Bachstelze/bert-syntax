import sys
from collections import *

script, result_path, model_name = sys.argv

files = [(model_name,  result_path)]

table_layout = """
\\begin{{table}}[H]
    \\begin{{tabular}}{{|l|c|c|}}
        \\hline Condition & Value & Test cases\\
{lines}
        \\hline \\end{{tabular}}
        \\caption{{Evaluation of {model_name} with an average of {mean}}}
    \\label{model_name}
\\end{{table}}
"""

line_layout = "        \\hline {condition} & {value} & {test_count}\\\\ \n"

by_model={}
conditions=set()
for title,fname in files:
    lines = open(fname)
    results=defaultdict(Counter)
    by_model[title]=results
    skipped = set()
    for line in lines:
        if line.startswith("Better speed"): continue
        if line.startswith("skipping"):
            skipped.add(line.split()[1])
            next(lines)
            continue
        res,c1,c2,w1,w2,s = line.split(None, 5)
        c1 = c1.replace("inanim","anim")
        conditions.add(c1)
        results[c1][res]+=1

print("skipped:",skipped)

results = ""
result_sum = 0.0
sum_counter = 0
for cond in conditions:
    ro = by_model[model_name][cond]
    if sum(ro.values())==0:
        so = "-"
    else:
        sum_counter += 1
        so = "%.2f" % (ro['True']/(ro['True']+ro['False']))
    result_sum += float(so)
    results += line_layout.format(condition=cond, value=so, test_count= sum(ro.values()))

print(table_layout.format(lines=results, model_name=model_name, mean=result_sum/sum_counter))
