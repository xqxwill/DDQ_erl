import os

import torch


def print_log(variable_name, variable):
    base_path = "/home/x/P/DDQ/mylog/test/"
    variable_path = variable_name + ".txt"
    real_path = os.path.join(base_path, variable_path)
    with open(real_path, "w") as file:
        file.write(str(variable))
    print "{}.shape::::{}".format(variable_name, variable.shape)
    print "{}.type::::{}".format(variable_name, type(variable))

t = torch.randint(0, 10, (4, 4))
print_log('t', t)
