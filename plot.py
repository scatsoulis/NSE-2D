from libraries import *

fig_ts, ax_ts = plt.subplots()
ax_ts.loglog([0.1, 0.01], [495038, 3681], '-x')
plt.grid(True, which="both")
plt.xlabel("Space-step")
plt.ylabel("Error")
plt.title("Order analysis")
plt.savefig("../hw2/order.pdf")
