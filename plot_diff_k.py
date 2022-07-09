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

k=6


if __name__ == '__main__':

    # filename_out1 = 'Brute'
    filename_out2 = "approx"
    # filename_out3 = "Random"

    dataname = 'd_sentiment'

    x_list, y_list = open_data("./output/" + dataname + "_0.9acc+approx(k=1)EBCC_compare.txt")
    x_list1, y_list1 = open_data("./output/" + dataname + "_0.9acc+approx(k=2)EBCC_compare.txt")
    x_list2, y_list2 = open_data("./output/" + dataname + "_0.9acc+approx(k=3)EBCC_compare.txt")
    x_list3, y_list3 = open_data("./output/" + dataname + "_0.9acc+approx(k=4)EBCC_compare.txt")
    x_list4, y_list4 = open_data("./output/" + dataname + "_0.9acc+approx(k=5)EBCC_compare.txt")
    x_list5, y_list5 = open_data("./output/" + dataname + "_0.9acc+approx(k=6)EBCC_compare.txt")
    plt.figure(figsize=(10,6), dpi=200, facecolor='w', edgecolor='k')

    plt.plot(x_list, y_list, "c-+", linewidth=1, label='K=1')
    plt.plot(x_list1, y_list1, "r-o", linewidth=1, label='K=2')
    plt.plot(x_list2, y_list2, "y-*", linewidth=1, label='K=3')
    plt.plot(x_list3, y_list3, "m-x", linewidth=1, label='K=4')
    plt.plot(x_list4, y_list4, "g-s", linewidth=1, label='K=5')
    plt.plot(x_list5, y_list5, "b-x", linewidth=1, label='K=6')

    # 设置坐标轴范围
    plt.xlim((0, 2000))
    plt.ylim((-280, 0))
    # 设置坐标轴名称
    plt.xlabel('Budgets',fontsize=26, fontproperties='Times New Roman', weight='bold')
    plt.ylabel('Quality',fontsize=26, fontproperties='Times New Roman', weight='bold')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 2001, 200)
    my_y_ticks = np.arange(-280, 1, 40)
    plt.xticks(my_x_ticks, size=18)
    plt.yticks(my_y_ticks, size=18)
    plt.tick_params(top=True, bottom=True, left=True, right=True,direction='in')
    plt.legend(loc='lower right',prop={'family' : 'Arial', 'size': 13})
    plt.savefig('./outputpdf/'+dataname+'_Quality_diff_k.pdf', dpi=800, bbox_inches='tight')
    plt.show()
