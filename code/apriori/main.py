from load_data import *
from utils import *
from apriori import apriori
from association_rules import gen_rules

if __name__ == '__main__':
    data = read_data()
    # cs = create_C1(data)
    # ret, support = scan_D(data,cs,0.5)
    ret, support = apriori(data,0.5)
    rules = gen_rules(ret,support,0.7)
    print(rules)
