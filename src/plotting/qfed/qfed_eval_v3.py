import os
import numpy as np

global_loss = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived\loss_global"
local_loss = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived\loss_local"

global_loss_path = sorted(os.listdir(global_loss))
for loss_path_ in global_loss_path:
    loss = np.load(os.path.join(global_loss, loss_path_))
    print(loss_path_)
    print("var", np.var(loss),"\n")

print("**********************************************************************")
print("**********************************************************************")
local_loss_path = sorted(os.listdir(local_loss))
for loss_path_ in local_loss_path:
    loss = np.load(os.path.join(local_loss, loss_path_))
    print(loss_path_)
    print("var", np.var(loss), "\n")