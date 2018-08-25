import os
import sys
import csv
import numpy as np
import re
import pprint
import itertools

# globals
DB_DIR = "../files/"
META_FILE = "../files/metadata.txt"
AGGREGATE = ["min", "max", "sum", "avg", "count", "distinct"]
RELATE_OPS = ["<", ">", "<=", ">=", "=", "<>"]
LITERAL = "<literal>"

schema = {}

def errorif(x, s):
    if x:
        # assert not x, s
        print("ERROR : {}".format(s))
        exit(-1)

def isint(s):
    try:
        _ = int(s)
        return True
    except:
        return False

def get_relate_op(cond):
    if "<=" in cond: op = "<="
    elif ">=" in cond: op = ">="
    elif "<>" in cond: op = "<>"
    elif ">" in cond: op = ">"
    elif "<" in cond: op = "<"
    elif "=" in cond: op = "="
    else : errorif(True, "invalid condition : '{}'".format(cond))

    errorif(cond.count(op) != 1, "invalid condition : '{}'".format(cond))
    l, r = cond.split(op)
    l = l.strip()
    r = r.strip()
    return op, l, r


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


"""
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
"""

def get_output_table(qdict):
    # pprint.pprint(qdict)
    alias2tb = qdict['alias2tb']
    inter_cols = qdict['inter_cols']
    tables = qdict['tables']
    conditions = qdict['conditions']
    cond_op = qdict['cond_op']
    proj_cols = qdict['proj_cols']

    # load all tables and retain only necessary columns
    # also decide the indexes of intermediate table columns
    colidx = {}
    cnt = 0
    all_tables = []
    for t in tables:
        lt = load_table(os.path.join(DB_DIR, "{}.csv".format( alias2tb[t] )))

        idxs = [schema[alias2tb[t]].index(cname) for cname in inter_cols[t]]
        lt = lt[:, idxs]
        all_tables.append(lt.tolist())

        colidx[t] = {cname: cnt+i for i, cname in enumerate(inter_cols[t])}
        cnt += len(inter_cols[t])

    # cartesian product of all tables
    inter_table = [[i for tup in r for i in list(tup)] for r in itertools.product(*all_tables)]
    inter_table = np.array(inter_table)

    # check for conditions and get reduced table
    if len(conditions):
        totake = np.ones((inter_table.shape[0],len(conditions)), dtype=bool)

        for idx, (op, left, right) in enumerate(conditions):
            cols = []
            for tname, cname in [left, right]:
                if tname == LITERAL: cols.append(np.full((inter_table.shape[0]), int(cname)))
                else: cols.append(inter_table[:, colidx[tname][cname]])

            if op=="<=": totake[:, idx] = (cols[0] <= cols[1])
            if op==">=": totake[:, idx] = (cols[0] >= cols[1])
            if op=="<>": totake[:, idx] = (cols[0] != cols[1])
            if op=="<": totake[:, idx] = (cols[0] < cols[1])
            if op==">": totake[:, idx] = (cols[0] > cols[1])
            if op=="=": totake[:, idx] = (cols[0] == cols[1])

        if cond_op == " or ": final_take = (totake[:, 0] |  totake[:, 1])
        elif cond_op == " and ": final_take = (totake[:, 0] & totake[:, 1])
        else: final_take = totake[:, 0]
        inter_table = inter_table[final_take]

    select_idxs = [colidx[tn][cn] for tn, cn, aggr in proj_cols]
    inter_table = inter_table[:, select_idxs]

    # process for aggregate function
    if proj_cols[0][2]:
        out_table = []
        disti = False
        for idx, (tn, cn, aggr) in enumerate(proj_cols):
            col = inter_table[:, idx]
            if aggr == "min": out_table.append(min(col))
            elif aggr == "max": out_table.append(max(col))
            elif aggr == "sum": out_table.append(sum(col))
            elif aggr == "avg": out_table.append(sum(col)/col.shape[0])
            elif aggr == "count": out_table.append(col.shape[0])
            elif aggr == "distinct":
                seen = set()
                out_table = [x for x in col.tolist() if not (x in seen or seen.add(x) )]
                disti = True
            else: errorif(True, "invalid aggregate")
        out_table = np.array([out_table])
        if disti: out_table = np.array(out_table).T
        out_header = ["{}({}.{})".format(aggr, tn, cn) for tn, cn, aggr in proj_cols]
    else:
        out_table = inter_table
        out_header = ["{}.{}".format(tn, cn) for tn, cn, aggr in proj_cols]
    return out_header, out_table.tolist()

def break_query(q):
    # check the structure of select, from and where
    # ----------------------------------------------
    toks = q.lower().split()
    if toks[0] != "select":
        log_error("only select is allowed")

    select_idx = [idx for idx, t in enumerate(toks) if t == "select"]
    from_idx = [idx for idx, t in enumerate(toks) if t == "from"]
    where_idx = [idx for idx, t in enumerate(toks) if t == "where"]

    errorif((len(select_idx) != 1) or (len(from_idx) != 1) or (len(where_idx) > 1), "invalid query")
    select_idx, from_idx = select_idx[0], from_idx[0]
    where_idx = where_idx[0] if len(where_idx) == 1 else None
    errorif(from_idx <= select_idx, "invalid query")
    if where_idx: errorif(where_idx <= from_idx, "invalid query")

    raw_cols = toks[select_idx+1:from_idx]
    if where_idx:
        raw_tables = toks[from_idx+1:where_idx]
        raw_condition = toks[where_idx+1:]
    else:
        raw_tables = toks[from_idx+1:]
        raw_condition = []

    errorif(len(raw_tables) == 0, "no tables after 'from'")
    errorif(where_idx != None and len(raw_condition) == 0, "no conditions after 'where'")
    # ----------------------------------------------
    return raw_tables, raw_cols, raw_condition

def parse_tables(raw_tables):
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
    return tables, alias2tb

def parse_proj_cols(raw_cols, tables, alias2tb):
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
                tname = [t for t in tables if cname in schema[alias2tb[t]]]
                errorif(len(tname) > 1, "not unique field : '{}'".format(rc))
                errorif(len(tname) == 0, "unknown field : '{}'".format(rc))
                tname = tname[0]

        # add all columns if *
        if cname == "*":
            errorif(aggr != None, "can't use aggregate '{}'".format(aggr))
            if tname != None:
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
    errorif(any([(a=="distinct") for a in s]) and len(s)!=1, "distinct can only be used alone")
    # ----------------------------------------------
    return proj_cols

def parse_conditions(raw_condition, tables, alias2tb):
    # parse conditions
    # ----------------------------------------------
    conditions = []
    cond_op = None
    if raw_condition:
        raw_condition = " ".join(raw_condition)

        if " or " in raw_condition: cond_op = " or "
        elif " and " in raw_condition: cond_op = " and "

        if cond_op: raw_condition = raw_condition.split(cond_op)
        else: raw_condition = [raw_condition]

        for cond in raw_condition:
            relate_op, left, right = get_relate_op(cond)
            parsed_cond = [relate_op]
            for idx, rc in enumerate([left, right]):
                if isint(rc):
                    parsed_cond.append((LITERAL, rc))
                    continue

                if "." in rc:
                    tname, cname = rc.split(".")
                else:
                    cname = rc
                    tname = [t for t in tables if rc in schema[alias2tb[t]]]
                    errorif(len(tname) > 1, "not unique field : '{}'".format(rc))
                    errorif(len(tname) == 0, "unknown field : '{}'".format(rc))
                    tname = tname[0]
                errorif((tname not in alias2tb.keys()) or (cname not in schema[alias2tb[tname]]),
                    "unknown field : '{}'".format(rc))
                parsed_cond.append((tname, cname))
            conditions.append(parsed_cond)
    # ----------------------------------------------
    return conditions, cond_op


def parse_query(q):

    # break query
    raw_tables, raw_cols, raw_condition = break_query(q)
    # get tables
    tables, alias2tb = parse_tables(raw_tables)
    # get columns to be projected
    proj_cols = parse_proj_cols(raw_cols, tables, alias2tb)
    # get conditions
    conditions, cond_op = parse_conditions(raw_condition, tables, alias2tb)

    # decide all needed columns for each table
    # ----------------------------------------------
    inter_cols = {t : set() for t in tables}
    for tn, cn, _ in proj_cols: inter_cols[tn].add(cn)
    for cond in conditions:
        for tn, cn in cond[1:]:
            if tn == LITERAL: continue
            inter_cols[tn].add(cn)

    for t in tables: inter_cols[t] = list(inter_cols[t])
    # ----------------------------------------------

    return {
        'tables':tables,
        'alias2tb':alias2tb,
        'proj_cols':proj_cols,
        'conditions':conditions,
        'cond_op':cond_op,
        'inter_cols':inter_cols,
    }


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
    out_header, out_table = get_output_table(qdict)

    print_table(out_header, out_table)
    # inter_header, inter_table = get_inter_table(qdict)


if __name__ == "__main__":
    main()
