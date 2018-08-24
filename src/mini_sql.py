import os
import sys
import csv
import numpy as np
import re
import pprint

# globals
DB_DIR = "../files/"
META_FILE = "../files/metadata.txt"
AGGREGATE = ["min", "max", "sum", "avg", "count"]

schema = {}

"""
def log_error(s):
    print("ERROR : {}".format(s))
    exit(-1)
def assertit(x, s):
    # assert x, s
    if not x:
        print("ERROR : {}".format(s))
        exit(-1)
"""

def errorif(x, s):
    if x:
        print("ERROR : {}".format(s))
        exit(-1)

def init_metadata():
    with open(META_FILE, "r") as f:
        contents = f.readlines()
    contents = [t.strip() for t in contents if t.strip()]

    table_name = None
    for t in contents:
        t = t.lower()
        if t == "<begin_table>": attrs, table_name = [], None
        elif t == "<end_table>": pass
        elif not table_name: table_name, schema[t] = t, []
        else: schema[table_name].append(t)

def load_table(fname):
    # ll = list(csv.reader(open(fname, "r")))
    # return list(map(lambda x : list(map(int, x)), ll))
    return np.genfromtxt(fname, dtype=int, delimiter=',')


def recursion(qdict, inter_table, till_row, idx):
    if idx == len(qdict['tables']):
        # check conditions
        inter_table.append(till_row)
        return

    for row in qdict['loaded_tables'][idx]:
        recursion(qdict, inter_table, till_row + row[qdict['inter_cols'][idx]].tolist(), idx+1)

def get_inter_table(qdict):

    temp_qdict = qdict.copy()
    qdict['loaded_tables'] = [
        load_table(os.path.join(DB_DIR, "{}.csv".format(qdict['alias2tb'][t]))) for t in qdict['tables']
    ]

    inter_table = []
    recursion(qdict, inter_table, [], 0)
    inter_header = [tname+"."+schema[qdict['alias2tb'][tname]][tc] for tname, tcols in \
        zip(qdict['tables'], qdict['inter_cols']) for tc in tcols]

    return inter_header, inter_table


def parse_query(q):
    toks = q.lower().split()
    # toks = q.split()

    # check the structure of select, from and where
    # ----------------------------------------------
    if toks[0] != "select":
        log_error("only select is allowed")

    select_idx = [idx for idx, t in enumerate(toks) if t == "select"]
    from_idx = [idx for idx, t in enumerate(toks) if t == "from"]
    where_idx = [idx for idx, t in enumerate(toks) if t == "where"]

    errorif(len(select_idx) != 1 or len(from_idx) != 1 or len(where_idx) > 1, "invalid query")

    raw_cols = toks[select_idx[0]+1:from_idx[0]]
    if len(where_idx):
        raw_tables = toks[from_idx[0]+1:where_idx[0]]
        raw_condition = toks[where_idx[0]+1:]
    else:
        raw_tables = toks[from_idx[0]+1:]
        raw_condition = []

    errorif(len(raw_tables) == 0, "no tables after 'from'")
    errorif(len(where_idx) != 0 and len(raw_condition) == 0, "no conditions after 'where'")
    # ----------------------------------------------

    # all joined tables
    # ----------------------------------------------
    raw_tables = " ".join(raw_tables).split(",")
    tables = []
    alias2tb = {}
    for rt in raw_tables:
        t = rt.split()
        errorif(not(len(t) == 1 or (len(t) == 3 and t[1] == "as")), "invalid table spacification '{}'".format(rt))
        if len(t) == 1: tb_name, tb_alias = t[0], t[0]
        else: tb_name, _, tb_alias = t

        errorif(tb_name not in schema.keys(), "no table name '{}'".format(tb_name))
        errorif(tb_alias in alias2tb.keys(), "not unique table/alias '{}'".format(tb_alias))

        tables.append(tb_alias)
        alias2tb[tb_alias] = tb_name
    # ----------------------------------------------


    # projection columns : columns to output
    # ----------------------------------------------
    raw_cols = "".join(raw_cols).split(",")
    proj_cols = []
    for rc in raw_cols:
        # match for aggregate function
        regmatch = re.match("(.+)\((.+)\)", rc)
        if regmatch: aggr, rc = regmatch.groups()
        else: aggr = None

        # either one of these two : col or table.col
        errorif("." in rc and len(rc.split(".")) != 2, "invalid column name '{}'".format(rc))

        # get table name and column name
        tname = None
        if "." in rc:
            tname, cname = rc.split(".")
            errorif(tname not in alias2tb.keys(), "unknown field : '{}'".format(rc))
        else:
            cname = rc
            if cname != "*":
                tname = [t for t in tables if rc in schema[alias2tb[t]]]
                errorif(len(tname) > 1, "not unique field : '{}'".format(rc))
                errorif(len(tname) == 0, "unknown field : '{}'".format(rc))
                tname = tname[0]

        # add all columns if *
        if cname == "*":
            errorif(aggr is not None, "can't use aggregate '{}'".format(aggr))
            if tname is not None:
                proj_cols.extend([(tname, c, aggr) for c in schema[alias2tb[tname]]])
            else:
                for t in tables:
                    proj_cols.extend([(t, c, aggr) for c in schema[alias2tb[t]]])
        else:
            errorif(cname not in schema[alias2tb[tname]], "unknown field : '{}'".format(rc))
            proj_cols.append((tname, cname, aggr))

    # either all columns without aggregate or all columns with aggregate
    s = [a for t, c, a in proj_cols]
    errorif(all(s) ^ any(s), "aggregated and nonaggregated columns are not allowed simultaneously")
    # ----------------------------------------------

    # parse conditions
    # ----------------------------------------------
    if raw_condition:
        raw_condition = " ".join(raw_condition)

        if " or " in raw_condition: op = " or "
        elif " and " in raw_condition: op = " and "
        else: op = None

        if op: raw_condition = raw_condition.split(op)
        else: raw_condition = [raw_condition]

        for cond in raw_condition:
            

    # ----------------------------------------------

    inter_cols = []
    return {'inter_cols':inter_cols, 'proj_cols':proj_cols, 'tables':tables, 'raw_condition':raw_condition, 'alias2tb':alias2tb}


def print_table(header, table):
    print(",".join(map(str, header)))
    for row in table:
        print(",".join(map(str, row)))


def main():
    init_metadata()
    if len(sys.argv) != 2:
        print("ERROR : invalid args")
        print("USAGE : python {} '<sql query>'".format(sys.argv[0]))
        exit(-1)
    q = sys.argv[1]
    qdict = parse_query(q)
    pprint.pprint(qdict)
    # inter_header, inter_table = get_inter_table(qdict)
    # print_table(inter_header, inter_table)


if __name__ == "__main__":
    main()
