"""
plot txt files from ./output
"""
import numpy as np
import matplotlib.pylab as plt


def open_data(str):
    with open(str, "r")as f:
        lines = f.readlines()
    x_list: list[int] = []
    y_list: list[float] = []
    for line in lines:
        x, y = line.split(' ')
        x_list.append(int(x))
        y_list.append(float(y))
    return x_list, y_list



if __name__ == '__main__':

    filename_out1 = "brute"
    filename_out2 = "approx"
    filename_out3 = "random"

    dataname = 'd_sentiment'

    x_list, y_list = open_data("./output/" + dataname + "_0.9acc+"+ filename_out1 + "(k=1)_s.txt")
    x_list1, y_list1 = open_data("./output/" + dataname + "_0.9acc+"+ filename_out2 + "(k=1)_s.txt")
    x_list2, y_list2 = open_data("./output/" + dataname + "_0.9acc+"+ filename_out3 + "(k=1)_s.txt")

    plt.figure(figsize=(10,6), dpi=200, facecolor='w', edgecolor='k')

    plt.plot(x_list, y_list, "r->", linewidth=1,label='OPT       K=1')
    plt.plot(x_list1, y_list1, "y-x", linewidth=1, label='Approx   K=1')
    plt.plot(x_list2, y_list2, "g-d", linewidth=1, label='Random K=1')
 

    # 设置坐标轴范围
    plt.xlim((0, 2000))
    plt.ylim((-300, 0))
    # 设置坐标轴名称
    plt.xlabel('Budgets',fontsize=26, fontproperties='Times New Roman', weight='bold')
    plt.ylabel('Quality',fontsize=26, fontproperties='Times New Roman', weight='bold')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 2001, 200)
    my_y_ticks = np.arange(-300, 1, 50)
    plt.xticks(my_x_ticks, size=18)
    plt.yticks(my_y_ticks, size=18)
    plt.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
    plt.legend(loc='lower right',prop={'family' : 'Arial', 'size': 13})
    plt.savefig('./outputpdf/'+dataname+'_diff_method_k=1.pdf', dpi=800, bbox_inches='tight')
    plt.show()
