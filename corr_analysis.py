import numpy as np
import sys

from model_fit import get_inp

def main():
    x_tr, x_te, y_tr, y_te, i_ref, o_ref = get_inp(sys.argv[1])
    x_tr = np.append(x_tr, np.array([y_tr]).T, axis=1)
    count = 0
    for c in np.corrcoef(x_tr, rowvar=False)[-1]:
        print(c)
        if abs(c) > 0.5:
            count += 1

    print("count:",count)

if __name__ == '__main__':
    main()
