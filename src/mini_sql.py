import os
import sys
import csv
import numpy as np

# globals
DB_DIR = "../files/"
META_FILE = "../files/metadata.txt"
schema = {}

def init_metadata():
    with open(META_FILE, "r") as f:
        contents = f.readlines()
    contents = [t.strip() for t in contents if t.strip()]

    table_name = None
    for t in contents:
        if t == "<begin_table>": attrs, table_name = [], None
        elif t == "<end_table>": pass
        elif not table_name: table_name, schema[t] = t, []
        else: schema[table_name].append(t)

def load_table(fname):
    ll = list(csv.reader(open(fname, "r")))
    return list(map(lambda x : list(map(int, x)), ll))
    # return np.genfromtxt(fname, dtype=int, delimiter=',')


def recursion(tables, conditions, inter_table, till_row, idx):
    if idx == len(tables):
        # check conditions
        inter_table.append(till_row)
        return

    for row in tables[idx]:
        recursion(tables, conditions, inter_table, till_row + row, idx+1)

def get_inter_table(tb_names, conditions):

    tables = [load_table(os.path.join(DB_DIR, "{}.csv".format(t))) for t in tb_names]

    inter_table = []
    recursion(tables, conditions, inter_table, [], 0)

    inter_header = []
    for t in tb_names: inter_header.extend(["{}.{}".format(t, a) for a in schema[t]])

    return inter_header, inter_table


def parse_query(q):
    toks = q.lower().split()

    if toks[0] != "select":
        print("ERROR : only select is allowed")
        exit(-1)

    select_idx = [idx for idx, t in enumerate(toks) if t == "select"]
    from_idx = [idx for idx, t in enumerate(toks) if t == "from"]
    where_idx = [idx for idx, t in enumerate(toks) if t == "where"]

    if len(select_idx) != 1 or len(from_idx) != 1 or len(where_idx) > 1:
        print("ERROR : invalid query")
        exit(-1)

    proj_cols = toks[select_idx[0]+1:from_idx[0]]
    if len(where_idx):
        tables = toks[from_idx[0]+1:where_idx[0]]
        conditions = toks[where_idx[0]+1:]
    else:
        tables = toks[from_idx[0]+1:]
        conditions = []

    if len(tables) == 0:
        print("ERROR : invalid query")
        print("ERROR : no tables after from")
        exit(-1)

    if len(where_idx) != 0 and len(conditions) == 0:
        print("ERROR : invalid query")
        print("ERROR : no conditions after where")
        exit(-1)

    tables = "".join(tables).split(",")
    for t in tables:
        if t not in schema.keys():
            print("ERROR : invalid query")
            print("ERROR : no table name '{}'".format(t))
            exit(-1)

    return {'proj_cols':proj_cols, 'tables':tables, 'conditions':conditions}


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
    qd = parse_query(q)
    inter_header, inter_table = get_inter_table(qd['tables'], qd['conditions'])
    print_table(inter_header, inter_table)


if __name__ == "__main__":
    main()
